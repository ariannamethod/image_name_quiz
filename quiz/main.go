// main.go — Image Name Quiz, Go + Ebiten rewrite (Arianna Method fork).
// Pure-local: no Python, no cloud. The "Describe -> Name" mode is powered by the
// SmolVLM engine in ../engine (captions are pre-computed into a cache).
package main

import (
	"fmt"
	"image/color"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"log"
	"os"
	"path/filepath"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
	"github.com/hajimehoshi/ebiten/v2/inpututil"
	"golang.org/x/image/font"
	"golang.org/x/image/font/gofont/goregular"
	"golang.org/x/image/font/opentype"
)

// scenes
const (
	sceneMenu       = "menu"
	sceneScoreboard = "scoreboard"
)

var answerCountCycle = []string{"4", "6", "8", "10", "All"}

type Game struct {
	model *Model

	faceBig, faceMid, faceSmall font.Face

	scene    string
	winW     int
	winH     int
	imgCache map[string]*ebiten.Image

	// per-frame widget list (immediate-mode: built in Update, drawn in Draw)
	buttons []*Button

	alert      string
	alertStyle string

	mx, my  int
	clicked bool

	// gameplay session state
	mode       string
	ms         map[string]*modeState
	score      int
	total      int
	awaiting   bool   // an answer was given; waiting for Next
	guessName  string // mode1/mode4 last guess
	guessPath  string // mode2 last guess
	correct    string // correct basename for the current question
	cur        *question
	mode2cells []imgCell

	// reveal-grid (mode 3) state
	revealCols int
	revealRows int
	revealed   []bool

	// mode 4 (Describe -> Name) caption cache
	caps *captionStore
}

func newGame(m *Model) *Game {
	g := &Game{
		model:    m,
		scene:    sceneMenu,
		winW:     m.Settings.Width,
		winH:     m.Settings.Height,
		imgCache: map[string]*ebiten.Image{},
		ms:       map[string]*modeState{},
		caps:     newCaptionStore(m.BaseDir),
	}
	g.loadFonts()
	return g
}

func (g *Game) showAlert(msg, style string) { g.alert, g.alertStyle = msg, style }
func (g *Game) clearAlert()                  { g.alert, g.alertStyle = "", "secondary" }

func (g *Game) loadFonts() {
	tt, err := opentype.Parse(goregular.TTF)
	if err != nil {
		log.Fatalf("font parse: %v", err)
	}
	mk := func(sz float64) font.Face {
		f, err := opentype.NewFace(tt, &opentype.FaceOptions{Size: sz, DPI: 72, Hinting: font.HintingFull})
		if err != nil {
			log.Fatalf("font face: %v", err)
		}
		return f
	}
	g.faceBig = mk(24)
	g.faceMid = mk(16)
	g.faceSmall = mk(12)
}

func (g *Game) image(path string) *ebiten.Image {
	if path == "" {
		return nil
	}
	if img, ok := g.imgCache[path]; ok {
		return img
	}
	img, _, err := ebitenutil.NewImageFromFile(path)
	if err != nil {
		g.imgCache[path] = nil
		return nil
	}
	g.imgCache[path] = img
	return img
}

// cached returns an already-loaded image (or nil) without disk I/O or mutation,
// so Draw stays side-effect-free; the Update path preloads via image().
func (g *Game) cached(path string) *ebiten.Image {
	return g.imgCache[path]
}

// ── Ebiten interface ────────────────────────────────────────────────────────

func (g *Game) Layout(outsideW, outsideH int) (int, int) {
	if outsideW != g.winW || outsideH != g.winH {
		g.winW, g.winH = outsideW, outsideH
		g.model.Settings.Width = outsideW
		g.model.Settings.Height = outsideH
		g.model.SaveSettings()
	}
	return outsideW, outsideH
}

func (g *Game) Update() error {
	g.mx, g.my = ebiten.CursorPosition()
	g.clicked = inpututil.IsMouseButtonJustPressed(ebiten.MouseButtonLeft)
	g.buttons = g.buttons[:0]

	switch g.scene {
	case sceneMenu:
		g.updateMenu()
	case sceneScoreboard:
		g.updateScoreboard()
	default:
		g.updateMode()
	}
	return nil
}

func (g *Game) Draw(screen *ebiten.Image) {
	screen.Fill(colBG)
	switch g.scene {
	case sceneMenu:
		g.drawMenu(screen)
	case sceneScoreboard:
		g.drawScoreboard(screen)
	default:
		g.drawMode(screen)
	}
	for _, b := range g.buttons {
		b.draw(screen, g.faceMid)
	}
}

// addButton registers a button, sets hover from the cursor, and returns true on click.
func (g *Game) addButton(b *Button) bool {
	b.hovered = b.hit(g.mx, g.my)
	g.buttons = append(g.buttons, b)
	return !b.Disabled && b.hovered && g.clicked
}

// ── Menu scene ────────────────────────────────────────────────────────────────

func (g *Game) updateMenu() {
	cx := float64(g.winW) / 2
	bw, bh := 360.0, 48.0
	x := cx - bw/2
	y := 150.0
	gap := 14.0

	modes := []struct {
		label string
		mode  string
	}{
		{"Mode 1: Image → Name", ModeImageToName},
		{"Mode 2: Name → Image", ModeNameToImage},
		{"Mode 3: Reveal Grid", ModeRevealGrid},
		{"Mode 4: Describe → Name (AI)", ModeDescribe},
	}
	for _, mo := range modes {
		if g.addButton(&Button{X: x, Y: y, W: bw, H: bh, Label: mo.label, Style: "primary"}) {
			g.startMode(mo.mode)
		}
		y += bh + gap
	}
	if g.addButton(&Button{X: x, Y: y, W: bw, H: bh, Label: "Scoreboard", Style: "info"}) {
		g.scene = sceneScoreboard
	}
	y += bh + gap

	// answer-count cycler
	if g.addButton(&Button{X: x, Y: y, W: bw, H: bh,
		Label: "Answer choices: " + g.model.Settings.AnswerCount, Style: "secondary"}) {
		g.cycleAnswerCount()
	}
}

func (g *Game) cycleAnswerCount() {
	cur := g.model.Settings.AnswerCount
	idx := 0
	for i, v := range answerCountCycle {
		if v == cur {
			idx = i
			break
		}
	}
	g.model.Settings.AnswerCount = answerCountCycle[(idx+1)%len(answerCountCycle)]
	g.model.SaveSettings()
}

func (g *Game) drawMenu(screen *ebiten.Image) {
	cx := float64(g.winW) / 2
	title := "Image Name Quiz"
	drawText(screen, g.faceBig, title, cx-float64(textWidth(g.faceBig, title))/2, 60, colText)
	sub := "Choose a mode"
	drawText(screen, g.faceMid, sub, cx-float64(textWidth(g.faceMid, sub))/2, 105, colPrimary)
}

// startMode enters a mode: resets state if the answer-count changed, then either
// re-shows the last question or picks a new one. Mirrors image_quiz.py _start_mode.
func (g *Game) startMode(mode string) {
	g.mode = mode
	g.scene = mode
	g.clearAlert()
	st := g.modeStateFor(mode)
	cur := g.model.Settings.AnswerCount
	if st.lastCount != cur {
		st.history, st.future, st.recent = nil, nil, nil
		st.used = map[string]bool{}
		st.lastCount = cur
	}
	g.awaiting = false
	g.guessName, g.guessPath = "", ""
	switch mode {
	case ModeImageToName, ModeNameToImage:
		st.recentBuf = 5 + g.model.rng.Intn(3) // 5..7
		if len(st.history) > 0 {
			g.restoreCurrent(st.history[len(st.history)-1])
		} else {
			g.newQuestion()
		}
	case ModeRevealGrid:
		g.revealNext()
	case ModeDescribe:
		st.recentBuf = 5 + g.model.rng.Intn(3) // 5..7
		if g.caps.haveAll(g.model.ImagePaths) {
			if len(st.history) > 0 {
				g.restoreCurrent(st.history[len(st.history)-1])
			} else {
				g.newQuestion()
			}
		} else {
			g.caps.buildAsync(g.model.BaseDir, g.model.ImagePaths) // build cache, then play
			g.cur = nil
		}
	}
}

func (g *Game) updateMode() {
	g.updateTopBar()
	switch g.mode {
	case ModeImageToName:
		g.updateMode1()
	case ModeNameToImage:
		g.updateMode2()
	case ModeRevealGrid:
		g.updateMode3()
	case ModeDescribe:
		g.updateMode4()
	}
}

func (g *Game) drawMode(screen *ebiten.Image) {
	switch g.mode {
	case ModeImageToName:
		g.drawMode1(screen)
	case ModeNameToImage:
		g.drawMode2(screen)
	case ModeRevealGrid:
		g.drawMode3(screen)
	case ModeDescribe:
		g.drawMode4(screen)
	default:
		g.drawPlaceholder(screen, "")
	}
}

// ── Scoreboard scene ───────────────────────────────────────────────────────────

func (g *Game) updateScoreboard() {
	if g.addButton(&Button{X: 20, Y: 20, W: 100, H: 36, Label: "Menu", Style: "secondary"}) {
		g.scene = sceneMenu
	}
}

func (g *Game) drawScoreboard(screen *ebiten.Image) {
	title := "Scoreboard"
	drawText(screen, g.faceBig, title, 140, 24, colText)
	y := 80.0
	type row struct {
		name  string
		score int
	}
	var rows []row
	for _, p := range g.model.SortedImagesByScore() {
		n := filepath.Base(p)
		rows = append(rows, row{DisplayName(n), g.model.Scores[n]})
	}
	for _, r := range rows {
		line := fmt.Sprintf("%4d   %s", r.score, r.name)
		drawText(screen, g.faceMid, line, 60, y, colText)
		y += float64(textHeight(g.faceMid)) + 6
		if y > float64(g.winH)-30 {
			break
		}
	}
}

var _ = color.Black // keep image/color import meaningful if unused elsewhere

func main() {
	base, err := os.Getwd()
	if err != nil {
		log.Fatal(err)
	}
	// If run from quiz/, the images live one level up (repo root /img).
	if _, statErr := os.Stat(filepath.Join(base, "img")); statErr != nil {
		if _, up := os.Stat(filepath.Join(base, "..", "img")); up == nil {
			base = filepath.Clean(filepath.Join(base, ".."))
		}
	}
	m, err := NewModel(base, nil)
	if err != nil {
		log.Fatalf("model: %v", err)
	}

	// `quizgame -captions` pre-builds the SmolVLM caption cache and exits
	// (headless; the engine + GGUF models must be present under engine/).
	if len(os.Args) > 1 && (os.Args[1] == "-captions" || os.Args[1] == "--captions") {
		cs := newCaptionStore(m.BaseDir)
		fmt.Printf("building captions for %d images via engine...\n", len(m.ImagePaths))
		cs.build(m.BaseDir, m.ImagePaths)
		if cs.errMsg != "" {
			log.Fatal(cs.errMsg)
		}
		done, total := cs.prog()
		fmt.Printf("done: %d/%d captions cached -> %s\n", done, total, cs.path)
		return
	}

	g := newGame(m)
	ebiten.SetWindowSize(m.Settings.Width, m.Settings.Height)
	ebiten.SetWindowTitle("Image Name Quiz")
	ebiten.SetWindowResizingMode(ebiten.WindowResizingModeEnabled)
	if err := ebiten.RunGame(g); err != nil {
		log.Fatal(err)
	}
}
