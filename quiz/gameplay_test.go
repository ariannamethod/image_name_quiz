package main

import (
	"math/rand"
	"path/filepath"
	"testing"
)

// newTestGame builds a Game with no renderer/fonts (Draw is never called),
// rooted at a tmp scores file so scoring can persist without touching the repo.
func newTestGame(t *testing.T, paths []string) *Game {
	t.Helper()
	m := testModel(paths, map[string]int{})
	m.scoresPath = filepath.Join(t.TempDir(), "scores.json")
	m.rng = rand.New(rand.NewSource(7))
	return &Game{model: m, ms: map[string]*modeState{}, scene: sceneMenu}
}

func sixImages() []string {
	return []string{
		"/img/Alex Cox.jpg", "/img/Rob Rabbit.jpg", "/img/Jessica Grey.jpg",
		"/img/Maud Blunder.jpg", "/img/Una Thomas.jpg", "/img/Jack Ball.jpg",
	}
}

func TestGameplayMode1CorrectThenWrong(t *testing.T) {
	g := newTestGame(t, sixImages())
	g.startMode(ModeImageToName)
	if g.cur == nil {
		t.Fatal("startMode did not produce a question")
	}
	if len(g.cur.names) != 4 {
		t.Fatalf("expected 4 name choices, got %d", len(g.cur.names))
	}
	if !contains(g.cur.names, g.correct) {
		t.Fatalf("correct answer %q not among choices %v", g.correct, g.cur.names)
	}

	// Answer correctly.
	correctPath := g.cur.path
	g.answerName(g.correct)
	if g.score != 1 || g.total != 1 {
		t.Fatalf("after correct: score/total = %d/%d, want 1/1", g.score, g.total)
	}
	if !g.awaiting {
		t.Fatal("expected awaiting=true after answering")
	}
	if !g.ms[ModeImageToName].used[correctPath] {
		t.Fatal("correct image not added to used set")
	}
	if g.model.Scores[filepath.Base(correctPath)] != 1 {
		t.Fatalf("persisted score = %d, want +1", g.model.Scores[filepath.Base(correctPath)])
	}

	// A second answer while awaiting must be ignored.
	g.answerName(g.cur.names[0])
	if g.total != 1 {
		t.Fatal("answer while awaiting should be a no-op")
	}

	// Next advances to a fresh question (awaiting reset).
	g.goForward()
	if g.awaiting {
		t.Fatal("goForward (advance) should clear awaiting")
	}
	if g.cur == nil {
		t.Fatal("no question after advance")
	}

	// Answer wrong: pick any choice that isn't correct.
	var wrong string
	for _, n := range g.cur.names {
		if n != g.correct {
			wrong = n
			break
		}
	}
	wrongPath := g.cur.path
	g.answerName(wrong)
	if g.score != 1 || g.total != 2 {
		t.Fatalf("after wrong: score/total = %d/%d, want 1/2", g.score, g.total)
	}
	if g.model.Scores[filepath.Base(wrongPath)] != -1 {
		t.Fatalf("persisted score for wrong = %d, want -1", g.model.Scores[filepath.Base(wrongPath)])
	}
}

func TestGameplayMode2PickImage(t *testing.T) {
	g := newTestGame(t, sixImages())
	g.startMode(ModeNameToImage)
	if g.cur == nil || len(g.cur.paths) != 4 {
		t.Fatalf("mode2 question malformed: cur=%v", g.cur)
	}
	if !contains(g.cur.paths, g.cur.path) {
		t.Fatal("correct image path not among choices")
	}
	g.answerImage(g.cur.path) // correct
	if g.score != 1 || g.total != 1 || !g.awaiting {
		t.Fatalf("mode2 correct: score/total/awaiting = %d/%d/%v", g.score, g.total, g.awaiting)
	}
}

func TestGameplayBackForward(t *testing.T) {
	g := newTestGame(t, sixImages())
	g.startMode(ModeImageToName)
	q1 := g.cur
	g.answerName(g.correct)
	g.goForward() // new question q2
	q2 := g.cur
	if q1 == q2 {
		t.Fatal("advance did not change the question")
	}
	g.goBack() // back to q1 (answered) -> should restore as awaiting (view its result)
	if g.cur != q1 {
		t.Fatal("goBack did not restore the previous question")
	}
	if !g.awaiting {
		t.Fatal("answered q1 should restore with awaiting=true")
	}
	g.goForward() // forward to q2 (unanswered) -> should be playable again
	if g.cur != q2 {
		t.Fatal("goForward did not restore the next question from future")
	}
	if g.awaiting {
		t.Fatal("unanswered q2 must restore as playable (awaiting=false)")
	}
	// q2 is genuinely answerable after the round-trip.
	g.answerName(g.correct)
	if g.total != 2 {
		t.Fatalf("answering restored q2 should score: total=%d, want 2", g.total)
	}
}

func TestGameplayRevealGrid(t *testing.T) {
	g := newTestGame(t, sixImages())
	// Make "Alex Cox" the clear weakest -> it must be revealed first.
	g.model.Scores = map[string]int{"Alex Cox.jpg": -5}
	g.startMode(ModeRevealGrid)
	if g.cur == nil {
		t.Fatal("reveal mode produced no image")
	}
	if filepath.Base(g.cur.path) != "Alex Cox.jpg" {
		t.Fatalf("first reveal = %q, want lowest-score Alex Cox.jpg", filepath.Base(g.cur.path))
	}
	if g.revealCols < 2 || g.revealRows < 2 || len(g.revealed) != g.revealCols*g.revealRows {
		t.Fatalf("grid malformed: %dx%d, revealed=%d", g.revealCols, g.revealRows, len(g.revealed))
	}
	for i, r := range g.revealed {
		if r {
			t.Fatalf("cell %d should start covered", i)
		}
	}
	if !g.ms[ModeRevealGrid].used[g.cur.path] {
		t.Fatal("revealed image not marked used")
	}
	first := g.cur.path
	g.revealNext()
	if g.cur.path == first {
		t.Fatal("revealNext did not advance to a new image")
	}
	if g.total != 0 || g.score != 0 {
		t.Fatalf("reveal mode must not score: total/score = %d/%d", g.total, g.score)
	}
}

func TestStartModeResetsOnAnswerCountChange(t *testing.T) {
	g := newTestGame(t, sixImages())
	g.startMode(ModeImageToName)
	g.answerName(g.correct)
	g.goForward()
	if len(g.ms[ModeImageToName].history) < 2 {
		t.Fatal("expected history to accumulate")
	}
	// Change answer count, re-enter: history must reset.
	g.model.Settings.AnswerCount = "6"
	g.startMode(ModeImageToName)
	if len(g.ms[ModeImageToName].history) != 1 {
		t.Fatalf("history not reset on answer-count change: %d", len(g.ms[ModeImageToName].history))
	}
}
