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

	mx, my       int
	clicked      bool
	currentImage string // path of the image currently shown (placeholder use in F0)
}

func newGame(m *Model) *Game {
	g := &Game{
		model:        m,
		scene:        sceneMenu,
		winW:         m.Settings.Width,
		winH:         m.Settings.Height,
		imgCache:     map[string]*ebiten.Image{},
		currentImage: firstOr(m.ImagePaths, ""),
	}
	g.loadFonts()
	return g
}

func firstOr(s []string, def string) string {
	if len(s) > 0 {
		return s[0]
	}
	return def
}

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
		g.updateMenu()
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
		g.drawMenu(screen)
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

// startMode is the dispatch point; real per-mode logic lands in F2-F4.
func (g *Game) startMode(mode string) {
	// Placeholder for F0: just remember the mode and show menu. F2 wires screens.
	g.scene = sceneMenu
	_ = mode
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
	g := newGame(m)
	ebiten.SetWindowSize(m.Settings.Width, m.Settings.Height)
	ebiten.SetWindowTitle("Image Name Quiz")
	ebiten.SetWindowResizingMode(ebiten.WindowResizingModeEnabled)
	if err := ebiten.RunGame(g); err != nil {
		log.Fatal(err)
	}
}
