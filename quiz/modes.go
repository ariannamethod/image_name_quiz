// modes.go — per-mode session state + gameplay for the four quiz modes.
// Faithful to image_quiz.py: question history/future per mode, used-correct set,
// recent buffer, +1/-1 scoring, correct/wrong highlight, Back/Next navigation.
package main

import (
	"math"
	"path/filepath"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/vector"
)

type question struct {
	path      string   // the image being asked about
	names     []string // mode1/mode4: name choices (basenames)
	paths     []string // mode2: image-path choices
	answered  bool     // whether this question was already answered (per-question, not global)
	guessName string   // the name picked (mode1/mode4), for restoring the highlight
	guessPath string   // the image picked (mode2), for restoring the highlight
}

type modeState struct {
	history   []*question
	future    []*question
	used      map[string]bool // images answered correctly this session
	recent    []string        // recent image paths (anti-repeat buffer)
	recentBuf int             // 5..7
	lastCount string          // answer-count when this state was built
}

type rect struct{ x, y, w, h float64 }

// imgCell is a clickable image thumbnail (mode2).
type imgCell struct {
	r    rect
	path string
}

// ── session helpers ────────────────────────────────────────────────────────

func (g *Game) modeStateFor(mode string) *modeState {
	if g.ms == nil {
		g.ms = map[string]*modeState{}
	}
	st := g.ms[mode]
	if st == nil {
		st = &modeState{used: map[string]bool{}}
		g.ms[mode] = st
	}
	return st
}

func (g *Game) setCurrent(q *question) {
	g.cur = q
	g.correct = filepath.Base(q.path)
}

// restoreCurrent shows a question reached by navigation (Back/Next/re-enter),
// reinstating its per-question answered state and guess highlight so an
// unanswered question stays playable and an answered one shows its result.
func (g *Game) restoreCurrent(q *question) {
	g.setCurrent(q)
	g.awaiting = q.answered
	g.guessName = q.guessName
	g.guessPath = q.guessPath
}

func (g *Game) newQuestion() {
	st := g.modeStateFor(g.mode)
	g.awaiting = false
	g.guessName, g.guessPath = "", ""
	path := g.model.PickNextImage(st.used, st.recent)
	if path == "" {
		g.showAlert("No more images left in this mode. Return to menu to restart.", "secondary")
		g.cur = nil
		return
	}
	q := &question{path: path}
	switch g.mode {
	case ModeImageToName, ModeDescribe:
		q.names = g.model.PickNameChoices(filepath.Base(path))
	case ModeNameToImage:
		q.paths = g.model.PickImageChoices(path)
	}
	st.history = append(st.history, q)
	st.future = nil
	g.setCurrent(q)
	st.recent = append(st.recent, path)
	if len(st.recent) > st.recentBuf {
		st.recent = st.recent[len(st.recent)-st.recentBuf:]
	}
}

func (g *Game) answerName(name string) {
	if g.awaiting || g.cur == nil {
		return
	}
	g.guessName = name
	ok := name == g.correct
	if ok {
		g.modeStateFor(g.mode).used[g.cur.path] = true
	}
	g.scoreDelta(g.cur.path, ok)
	g.awaiting = true
	g.cur.answered = true
	g.cur.guessName = name
}

func (g *Game) answerImage(path string) {
	if g.awaiting || g.cur == nil {
		return
	}
	g.guessPath = path
	ok := path == g.cur.path
	if ok {
		g.modeStateFor(g.mode).used[g.cur.path] = true
	}
	g.scoreDelta(g.cur.path, ok)
	g.awaiting = true
	g.cur.answered = true
	g.cur.guessPath = path
}

func (g *Game) scoreDelta(path string, ok bool) {
	delta := -1
	if ok {
		delta = 1
	}
	g.model.UpdateImageScore(path, delta)
	g.total++
	if ok {
		g.score++
		g.showAlert("Correct!", "success")
	} else {
		g.showAlert("Incorrect. Correct answer: "+DisplayName(g.correct), "danger")
	}
}

func (g *Game) goForward() {
	st := g.modeStateFor(g.mode)
	if g.mode == ModeRevealGrid {
		if len(st.future) > 0 {
			q := st.future[len(st.future)-1]
			st.future = st.future[:len(st.future)-1]
			st.history = append(st.history, q)
			g.setCurrent(q)
			g.buildRevealGrid()
		} else {
			g.revealNext()
		}
		return
	}
	// Prefer replaying the future stack (set by Back); only generate a new
	// question when at the tip and the current one has been answered.
	if len(st.future) > 0 {
		q := st.future[len(st.future)-1]
		st.future = st.future[:len(st.future)-1]
		st.history = append(st.history, q)
		g.restoreCurrent(q)
		return
	}
	if g.awaiting {
		g.newQuestion()
	}
}

func (g *Game) goBack() {
	st := g.modeStateFor(g.mode)
	if len(st.history) <= 1 {
		return
	}
	cur := st.history[len(st.history)-1]
	st.history = st.history[:len(st.history)-1]
	st.future = append(st.future, cur)
	prev := st.history[len(st.history)-1]
	g.restoreCurrent(prev)
	if g.mode == ModeRevealGrid {
		g.buildRevealGrid()
	}
}

// ── Mode 3: Reveal Grid ──────────────────────────────────────────────────────
// A study mode: the name is shown as the title, the image sits under a grid of
// opaque cells; click a cell to reveal that patch. Images are walked in
// score-ascending order (weakest first). No scoring. Mirrors image_quiz.py mode3.

func (g *Game) buildRevealGrid() {
	count := g.model.AnswerCount()
	if count == 0 { // "All" -> a sensible default grid like the Python version
		count = 12
	}
	cols := int(math.Sqrt(float64(count)))
	if cols < 2 {
		cols = 2
	}
	rows := (count + cols - 1) / cols
	if rows < 2 {
		rows = 2
	}
	g.revealCols, g.revealRows = cols, rows
	g.revealed = make([]bool, cols*rows)
}

func (g *Game) revealNext() {
	st := g.modeStateFor(ModeRevealGrid)
	var path string
	for _, p := range g.model.SortedImagesByScore() {
		if !st.used[p] {
			path = p
			break
		}
	}
	if path == "" {
		g.showAlert("No more images left in this mode. Return to menu to restart.", "secondary")
		g.cur = nil
		return
	}
	st.used[path] = true
	q := &question{path: path}
	st.history = append(st.history, q)
	st.future = nil
	g.setCurrent(q)
	g.buildRevealGrid()
	g.clearAlert()
}

// revealImgRect is the on-screen rect of the image (aspect-fit), recomputed each
// frame so the cell grid stays aligned across window resizes.
func (g *Game) revealImgRect() rect {
	img := g.cached(g.cur.path)
	if img == nil {
		return rect{}
	}
	ax, ay := 10.0, 110.0
	aw, ah := float64(g.winW)-20, float64(g.winH)-ay-12
	iw, ih := float64(img.Bounds().Dx()), float64(img.Bounds().Dy())
	if iw == 0 || ih == 0 {
		return rect{}
	}
	scale := min3(aw/iw, ah/ih, 1.0)
	dw, dh := iw*scale, ih*scale
	return rect{x: ax + (aw-dw)/2, y: ay + (ah-dh)/2, w: dw, h: dh}
}

func (g *Game) updateMode3() {
	if g.cur == nil || g.revealCols == 0 || g.revealRows == 0 {
		return
	}
	g.image(g.cur.path) // preload so Draw only reads the cache
	if !g.clicked {
		return
	}
	r := g.revealImgRect()
	if r.w == 0 || !hitRect(r, g.mx, g.my) {
		return
	}
	cw := r.w / float64(g.revealCols)
	ch := r.h / float64(g.revealRows)
	c := int((float64(g.mx) - r.x) / cw)
	rr := int((float64(g.my) - r.y) / ch)
	if c >= 0 && c < g.revealCols && rr >= 0 && rr < g.revealRows {
		idx := rr*g.revealCols + c
		if idx >= 0 && idx < len(g.revealed) {
			g.revealed[idx] = true
		}
	}
}

func (g *Game) drawMode3(screen *ebiten.Image) {
	g.drawTopBar(screen)
	if g.cur == nil {
		return
	}
	title := DisplayName(g.correct)
	cx := float64(g.winW) / 2
	drawText(screen, g.faceBig, title, cx-float64(textWidth(g.faceBig, title))/2, 80, colText)
	img := g.cached(g.cur.path)
	if img == nil {
		return
	}
	r := g.revealImgRect()
	iw, ih := float64(img.Bounds().Dx()), float64(img.Bounds().Dy())
	op := &ebiten.DrawImageOptions{}
	op.GeoM.Scale(r.w/iw, r.h/ih)
	op.GeoM.Translate(r.x, r.y)
	op.Filter = ebiten.FilterLinear
	screen.DrawImage(img, op)
	// opaque cover over un-revealed cells
	cw := r.w / float64(g.revealCols)
	ch := r.h / float64(g.revealRows)
	for rr := 0; rr < g.revealRows; rr++ {
		for c := 0; c < g.revealCols; c++ {
			if g.revealed[rr*g.revealCols+c] {
				continue
			}
			x := float32(r.x + float64(c)*cw)
			y := float32(r.y + float64(rr)*ch)
			vector.DrawFilledRect(screen, x, y, float32(cw), float32(ch), colGrid, false)
			vector.StrokeRect(screen, x, y, float32(cw), float32(ch), 1, colBG, false)
		}
	}
}

// ── Mode 4: Describe → Name (VLM) ────────────────────────────────────────────
// Shows ONLY the local SmolVLM caption (image hidden until answered); guess the
// name from the same choice grid as Mode 1. Captions are pre-built into a cache.

func (g *Game) updateMode4() {
	if g.caps.isBuilding() {
		return // building cache; just the top bar is interactive
	}
	if g.cur == nil {
		if g.caps.haveAll(g.model.ImagePaths) {
			g.newQuestion() // cache ready -> first question
		}
		return
	}
	g.image(g.cur.path) // preload (revealed after answering) so Draw only reads the cache
	ay := float64(g.winH) * 0.52
	g.updateNameButtons(10, ay, float64(g.winW)-20, float64(g.winH)-ay-12)
}

func (g *Game) drawMode4(screen *ebiten.Image) {
	g.drawTopBar(screen)
	cx := float64(g.winW) / 2
	if g.caps.isBuilding() {
		done, total := g.caps.prog()
		msg := "Building local descriptions (SmolVLM): " + itoa(done) + "/" + itoa(total)
		drawText(screen, g.faceMid, msg, cx-float64(textWidth(g.faceMid, msg))/2, float64(g.winH)/2, colInfo)
		return
	}
	if g.caps.errMsg != "" {
		for i, line := range wrapText(g.faceMid, g.caps.errMsg, float64(g.winW)*0.7) {
			drawText(screen, g.faceMid, line, cx-float64(textWidth(g.faceMid, line))/2,
				float64(g.winH)/2+float64(i*(textHeight(g.faceMid)+4)), colDanger)
		}
		return
	}
	if g.cur == nil {
		return
	}
	head := "AI description — who is this?"
	drawText(screen, g.faceMid, head, cx-float64(textWidth(g.faceMid, head))/2, 84, colPrimary)
	cap := g.caps.get(g.correct)
	if cap == "" {
		cap = "(no description)"
	}
	y := 130.0
	for _, line := range wrapText(g.faceBig, "“"+cap+"”", float64(g.winW)*0.8) {
		drawText(screen, g.faceBig, line, cx-float64(textWidth(g.faceBig, line))/2, y, colText)
		y += float64(textHeight(g.faceBig)) + 6
	}
	// reveal the actual image once answered
	if g.awaiting {
		if img := g.cached(g.cur.path); img != nil {
			drawImageFit(screen, img, cx-110, y+10, 220, float64(g.winH)*0.52-y-20)
		}
	}
}

// ── top bar (shared) ──────────────────────────────────────────────────────

func (g *Game) updateTopBar() {
	bw, bh, pad := 84.0, 36.0, 8.0
	x := float64(g.winW) - bw - 12
	if g.addButton(&Button{X: x, Y: 12, W: bw, H: bh, Label: "Menu", Style: "secondary"}) {
		g.scene = sceneMenu
		g.clearAlert()
	}
	x -= bw + pad
	if g.addButton(&Button{X: x, Y: 12, W: bw, H: bh, Label: "Next", Style: "secondary"}) {
		g.goForward()
	}
	x -= bw + pad
	if g.addButton(&Button{X: x, Y: 12, W: bw, H: bh, Label: "Back", Style: "secondary"}) {
		g.goBack()
	}
}

func (g *Game) drawTopBar(screen *ebiten.Image) {
	drawText(screen, g.faceMid, scoreText(g.score, g.total), 20, 18, colText)
	if g.alert != "" {
		clr := styleColor(g.alertStyle, false)
		cx := float64(g.winW) / 2
		drawText(screen, g.faceMid, g.alert, cx-float64(textWidth(g.faceMid, g.alert))/2, 60, clr)
	}
}

func scoreText(s, t int) string {
	return "Score: " + itoa(s) + "/" + itoa(t)
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		b[i] = '-'
	}
	return string(b[i:])
}

// ── layout helpers ─────────────────────────────────────────────────────────

func colsFor(areaW float64, n, per int) int {
	cols := int(areaW) / per
	if cols < 2 {
		cols = 2
	}
	if cols > 4 {
		cols = 4
	}
	if cols > n {
		cols = n
	}
	if cols < 1 {
		cols = 1
	}
	return cols
}

func gridLayout(ax, ay, aw, ah float64, n, cols int, pad float64) []rect {
	if n == 0 || cols < 1 {
		return nil
	}
	rows := (n + cols - 1) / cols
	cw := (aw - pad*float64(cols+1)) / float64(cols)
	ch := (ah - pad*float64(rows+1)) / float64(rows)
	out := make([]rect, n)
	for i := 0; i < n; i++ {
		r, c := i/cols, i%cols
		out[i] = rect{
			x: ax + pad + float64(c)*(cw+pad),
			y: ay + pad + float64(r)*(ch+pad),
			w: cw,
			h: ch,
		}
	}
	return out
}

func (g *Game) choiceStyleName(name string) (style string, disabled bool) {
	if !g.awaiting {
		return "primary", false
	}
	switch {
	case name == g.correct:
		return "success", true
	case name == g.guessName:
		return "danger", true
	default:
		return "secondary", true
	}
}

// ── Mode 1: Image → Name ─────────────────────────────────────────────────────

// updateNameButtons lays out the name-choice grid in the given area (shared by
// Mode 1 and Mode 4, which differ only in what is shown above the choices).
func (g *Game) updateNameButtons(ax, ay, aw, ah float64) {
	cols := colsFor(aw, len(g.cur.names), 220)
	rects := gridLayout(ax, ay, aw, ah, len(g.cur.names), cols, 10)
	for i, name := range g.cur.names {
		style, dis := g.choiceStyleName(name)
		r := rects[i]
		b := &Button{X: r.x, Y: r.y, W: r.w, H: r.h, Label: DisplayName(name), Style: style, Disabled: dis}
		if g.addButton(b) {
			g.answerName(name)
		}
	}
}

func (g *Game) updateMode1() {
	if g.cur == nil {
		return
	}
	g.image(g.cur.path) // preload so Draw only reads the cache
	ay := float64(g.winH) * 0.55
	g.updateNameButtons(10, ay, float64(g.winW)-20, float64(g.winH)-ay-12)
}

func (g *Game) drawMode1(screen *ebiten.Image) {
	g.drawTopBar(screen)
	if g.cur == nil {
		return
	}
	if img := g.cached(g.cur.path); img != nil {
		drawImageFit(screen, img, 10, 84, float64(g.winW)-20, float64(g.winH)*0.55-94)
	}
}

// ── Mode 2: Name → Image ─────────────────────────────────────────────────────

func (g *Game) updateMode2() {
	if g.cur == nil {
		g.mode2cells = nil
		return
	}
	ax, ay := 10.0, 120.0
	aw, ah := float64(g.winW)-20, float64(g.winH)-ay-12
	cols := colsFor(aw, len(g.cur.paths), 260)
	rects := gridLayout(ax, ay, aw, ah, len(g.cur.paths), cols, 12)
	g.mode2cells = g.mode2cells[:0]
	for i, p := range g.cur.paths {
		g.image(p) // preload so Draw only reads the cache
		g.mode2cells = append(g.mode2cells, imgCell{r: rects[i], path: p})
		if !g.awaiting && g.clicked && hitRect(rects[i], g.mx, g.my) {
			g.answerImage(p)
		}
	}
}

func (g *Game) drawMode2(screen *ebiten.Image) {
	g.drawTopBar(screen)
	if g.cur == nil {
		return
	}
	label := "Pick the image for: " + DisplayName(g.correct)
	cx := float64(g.winW) / 2
	drawText(screen, g.faceBig, label, cx-float64(textWidth(g.faceBig, label))/2, 84, colText)
	for _, cell := range g.mode2cells {
		// border color signals state after answering
		border := colPrimary
		if g.awaiting {
			switch {
			case cell.path == g.cur.path:
				border = colSuccess
			case cell.path == g.guessPath:
				border = colDanger
			default:
				border = colSecondary
			}
		} else if hitRect(cell.r, g.mx, g.my) {
			border = colPrimaryHi
		}
		vector.StrokeRect(screen, float32(cell.r.x), float32(cell.r.y), float32(cell.r.w), float32(cell.r.h), 3, border, true)
		if img := g.cached(cell.path); img != nil {
			drawImageFit(screen, img, cell.r.x+5, cell.r.y+5, cell.r.w-10, cell.r.h-10)
		}
	}
}

func hitRect(r rect, mx, my int) bool {
	x, y := float64(mx), float64(my)
	return x >= r.x && x <= r.x+r.w && y >= r.y && y <= r.y+r.h
}

// ── generic placeholder (fallback for an unknown scene) ──────────────────────

func (g *Game) drawPlaceholder(screen *ebiten.Image, msg string) {
	g.drawTopBar(screen)
	cx := float64(g.winW) / 2
	drawText(screen, g.faceMid, msg, cx-float64(textWidth(g.faceMid, msg))/2, float64(g.winH)/2, colSecondary)
}
