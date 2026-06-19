package main

import (
	"math/rand"
	"path/filepath"
	"reflect"
	"sort"
	"testing"
)

func testModel(paths []string, scores map[string]int) *Model {
	if scores == nil {
		scores = map[string]int{}
	}
	return &Model{
		ImagePaths: paths,
		Scores:     scores,
		Settings:   Settings{AnswerCount: "4"},
		rng:        rand.New(rand.NewSource(42)),
	}
}

func names(paths []string) []string {
	out := make([]string, len(paths))
	for i, p := range paths {
		out[i] = filepath.Base(p)
	}
	return out
}

func TestNameTokens(t *testing.T) {
	cases := map[string][]string{
		"Alex Cox.jpg":          {"alex", "cox"},
		"Mike_Butterscotch.png": {"mike", "butterscotch"},
		"Rob-Rabbit.jpeg":       {"rob", "rabbit"},
	}
	for in, want := range cases {
		if got := nameTokens(in); !reflect.DeepEqual(got, want) {
			t.Errorf("nameTokens(%q) = %v, want %v", in, got, want)
		}
	}
}

func TestDisplayName(t *testing.T) {
	if got := DisplayName("Jessica Grey.jpg"); got != "Jessica Grey" {
		t.Errorf("DisplayName = %q, want %q", got, "Jessica Grey")
	}
}

func TestAnswerCount(t *testing.T) {
	cases := map[string]int{"All": 0, "all": 0, "8": 8, "4": 4, "": 4, "xx": 4}
	for in, want := range cases {
		m := &Model{Settings: Settings{AnswerCount: in}}
		if got := m.AnswerCount(); got != want {
			t.Errorf("AnswerCount(%q) = %d, want %d", in, got, want)
		}
	}
}

func TestSimilarNamesPrefersSharedTokens(t *testing.T) {
	// "Jack Ball" shares "ball" with "Lucy Ball" and "jack" with "Jack Sparrow".
	all := []string{"Jack Ball.jpg", "Lucy Ball.jpg", "Jack Sparrow.jpg", "Una Thomas.jpg", "Maud Blunder.jpg"}
	m := testModel(nil, nil)
	got := m.similarNames("Jack Ball.jpg", all)
	if len(got) != 3 {
		t.Fatalf("similarNames returned %d, want 3", len(got))
	}
	// The two token-sharing names must be present (they rank above the rest).
	set := map[string]bool{}
	for _, n := range got {
		set[n] = true
	}
	if !set["Lucy Ball.jpg"] || !set["Jack Sparrow.jpg"] {
		t.Errorf("similarNames = %v, expected the token-sharing names present", got)
	}
	if set["Jack Ball.jpg"] {
		t.Errorf("similarNames must not include the target itself: %v", got)
	}
}

func TestPickNameChoicesCountAndCorrect(t *testing.T) {
	all := []string{"A One.jpg", "B Two.jpg", "C Three.jpg", "D Four.jpg", "E Five.jpg", "F Six.jpg"}
	m := testModel(all, nil)
	m.Settings.AnswerCount = "4"
	choices := m.PickNameChoices("A One.jpg")
	if len(choices) != 4 {
		t.Fatalf("got %d choices, want 4", len(choices))
	}
	if !contains(choices, "A One.jpg") {
		t.Errorf("correct answer missing from choices: %v", choices)
	}
	if hasDup(choices) {
		t.Errorf("duplicate choices: %v", choices)
	}
}

func TestPickNameChoicesAll(t *testing.T) {
	all := []string{"A.jpg", "B.jpg", "C.jpg"}
	m := testModel(all, nil)
	m.Settings.AnswerCount = "All"
	choices := m.PickNameChoices("B.jpg")
	if len(choices) != 3 {
		t.Fatalf("All: got %d, want 3", len(choices))
	}
	sort.Strings(choices)
	if !reflect.DeepEqual(choices, []string{"A.jpg", "B.jpg", "C.jpg"}) {
		t.Errorf("All choices = %v", choices)
	}
}

func TestPickImageChoices(t *testing.T) {
	all := []string{"/img/A.jpg", "/img/B.jpg", "/img/C.jpg", "/img/D.jpg", "/img/E.jpg"}
	m := testModel(all, nil)
	m.Settings.AnswerCount = "4"
	choices := m.PickImageChoices("/img/C.jpg")
	if len(choices) != 4 {
		t.Fatalf("got %d, want 4", len(choices))
	}
	if !contains(choices, "/img/C.jpg") {
		t.Errorf("correct path missing: %v", choices)
	}
	if hasDup(choices) {
		t.Errorf("duplicate choices: %v", choices)
	}
}

func TestPickNextImageLowestScore(t *testing.T) {
	all := []string{"/img/A.jpg", "/img/B.jpg", "/img/C.jpg"}
	scores := map[string]int{"A.jpg": 5, "B.jpg": -2, "C.jpg": 3} // B is lowest
	m := testModel(all, scores)
	got := m.PickNextImage(map[string]bool{}, nil)
	if filepath.Base(got) != "B.jpg" {
		t.Errorf("PickNextImage = %q, want B.jpg (lowest score)", got)
	}
}

func TestPickNextImageExcludesUsedAndRecent(t *testing.T) {
	all := []string{"/img/A.jpg", "/img/B.jpg", "/img/C.jpg"}
	scores := map[string]int{"A.jpg": 0, "B.jpg": 0, "C.jpg": 0}
	m := testModel(all, scores)
	// A used-correct, B recent -> only C can be picked.
	used := map[string]bool{"/img/A.jpg": true}
	got := m.PickNextImage(used, []string{"/img/B.jpg"})
	if filepath.Base(got) != "C.jpg" {
		t.Errorf("PickNextImage = %q, want C.jpg (A used, B recent)", got)
	}
}

func TestPickNextImageEmpty(t *testing.T) {
	all := []string{"/img/A.jpg"}
	m := testModel(all, nil)
	used := map[string]bool{"/img/A.jpg": true}
	if got := m.PickNextImage(used, nil); got != "" {
		t.Errorf("expected empty when all used, got %q", got)
	}
}

func TestSortedImagesByScore(t *testing.T) {
	all := []string{"/img/A.jpg", "/img/B.jpg", "/img/C.jpg"}
	scores := map[string]int{"A.jpg": 2, "B.jpg": -1, "C.jpg": 2}
	m := testModel(all, scores)
	got := names(m.SortedImagesByScore())
	want := []string{"B.jpg", "A.jpg", "C.jpg"} // score asc, then name asc for ties
	if !reflect.DeepEqual(got, want) {
		t.Errorf("SortedImagesByScore = %v, want %v", got, want)
	}
}

func TestUpdateImageScoreNoPanic(t *testing.T) {
	m := testModel([]string{"/img/A.jpg"}, nil)
	m.scoresPath = filepath.Join(t.TempDir(), "scores.json")
	m.UpdateImageScore("/img/A.jpg", 1)
	m.UpdateImageScore("/img/A.jpg", -1)
	if m.Scores["A.jpg"] != 0 {
		t.Errorf("score = %d, want 0", m.Scores["A.jpg"])
	}
}

func contains(s []string, v string) bool {
	for _, x := range s {
		if x == v {
			return true
		}
	}
	return false
}

func hasDup(s []string) bool {
	seen := map[string]bool{}
	for _, x := range s {
		if seen[x] {
			return true
		}
		seen[x] = true
	}
	return false
}
