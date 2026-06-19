// model.go — pure game logic for the image-name quiz (no UI, fully testable).
// Faithful port of image_quiz.py: per-image scores (+1/-1), settings, the
// "similar names" distractor logic, and the lowest-score / recent-buffer
// question picker. Randomness is injected so the logic is deterministic in tests.
package main

import (
	"encoding/json"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Mode identifiers (match the Python keys so scores/settings stay compatible).
const (
	ModeImageToName = "image_to_name"
	ModeNameToImage = "name_to_image"
	ModeRevealGrid  = "reveal_grid"
	ModeDescribe    = "describe_to_name" // new: VLM caption -> name
)

var supportedExts = map[string]bool{
	".png": true, ".jpg": true, ".jpeg": true, ".gif": true, ".bmp": true,
}

// Settings persists window size + answer-count choice (mirrors image_quiz_settings.json).
type Settings struct {
	Width       int    `json:"width"`
	Height      int    `json:"height"`
	AnswerCount string `json:"answer_count"` // "4","6","8","10","All"
}

// Model holds everything the game needs that is independent of the renderer.
type Model struct {
	BaseDir    string
	ImageDir   string
	ImagePaths []string       // absolute, sorted
	Scores     map[string]int // basename -> cumulative score
	Settings   Settings

	scoresPath   string
	settingsPath string
	rng          *rand.Rand
}

// NewModel loads images, scores and settings rooted at baseDir.
// rng is injectable; pass nil for a time-seeded default.
func NewModel(baseDir string, rng *rand.Rand) (*Model, error) {
	if rng == nil {
		rng = rand.New(rand.NewSource(rand.Int63()))
	}
	m := &Model{
		BaseDir:      baseDir,
		ImageDir:     filepath.Join(baseDir, "img"),
		scoresPath:   filepath.Join(baseDir, "image_quiz_scores.json"),
		settingsPath: filepath.Join(baseDir, "image_quiz_settings.json"),
		rng:          rng,
	}
	if err := os.MkdirAll(m.ImageDir, 0o755); err != nil {
		return nil, err
	}
	m.ImagePaths = loadImages(m.ImageDir)
	m.Settings = m.loadSettings()
	m.Scores = m.loadScores()
	return m, nil
}

func loadImages(dir string) []string {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	var paths []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		ext := strings.ToLower(filepath.Ext(e.Name()))
		if supportedExts[ext] {
			paths = append(paths, filepath.Join(dir, e.Name()))
		}
	}
	sort.Strings(paths)
	return paths
}

func (m *Model) loadSettings() Settings {
	s := Settings{Width: 900, Height: 700, AnswerCount: "4"}
	data, err := os.ReadFile(m.settingsPath)
	if err == nil {
		_ = json.Unmarshal(data, &s) // best-effort, keep defaults on bad fields
	}
	if s.Width <= 0 {
		s.Width = 900
	}
	if s.Height <= 0 {
		s.Height = 700
	}
	if s.AnswerCount == "" {
		s.AnswerCount = "4"
	}
	return s
}

// SaveSettings writes the current window size + answer-count to disk.
func (m *Model) SaveSettings() {
	data, err := json.MarshalIndent(m.Settings, "", "  ")
	if err != nil {
		return
	}
	_ = os.WriteFile(m.settingsPath, data, 0o644)
}

func (m *Model) loadScores() map[string]int {
	scores := map[string]int{}
	data, err := os.ReadFile(m.scoresPath)
	if err == nil {
		raw := map[string]int{}
		if json.Unmarshal(data, &raw) == nil {
			scores = raw
		}
	}
	// Drop entries for images that no longer exist (mirrors Python).
	valid := map[string]bool{}
	for _, p := range m.ImagePaths {
		valid[filepath.Base(p)] = true
	}
	for k := range scores {
		if !valid[k] {
			delete(scores, k)
		}
	}
	return scores
}

// SaveScores persists the score map (sorted keys for a stable diff).
func (m *Model) SaveScores() {
	data, err := json.MarshalIndent(m.Scores, "", "  ")
	if err != nil {
		return
	}
	_ = os.WriteFile(m.scoresPath, data, 0o644)
}

// AnswerCount returns the configured number of choices, or 0 meaning "All".
func (m *Model) AnswerCount() int {
	v := strings.ToLower(strings.TrimSpace(m.Settings.AnswerCount))
	if v == "all" {
		return 0
	}
	n := 0
	for _, c := range v {
		if c < '0' || c > '9' {
			return 4 // matches Python's ValueError fallback
		}
		n = n*10 + int(c-'0')
	}
	if n == 0 {
		return 4
	}
	return n
}

// DisplayName strips the extension (the on-screen / answer label).
func DisplayName(name string) string {
	return strings.TrimSuffix(name, filepath.Ext(name))
}

// nameTokens splits a filename (sans ext) on space/-/_ into lowercase words.
func nameTokens(name string) []string {
	base := DisplayName(name)
	base = strings.NewReplacer("-", " ", "_", " ").Replace(base)
	var out []string
	for _, p := range strings.Fields(base) {
		p = strings.ToLower(strings.TrimSpace(p))
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

// similarNames returns up to 3 names sharing tokens with target (desc overlap,
// then name asc), padded with random others to 3. Mirrors Python _similar_names.
func (m *Model) similarNames(target string, allNames []string) []string {
	targetTok := map[string]bool{}
	for _, t := range nameTokens(target) {
		targetTok[t] = true
	}
	type sc struct {
		score int
		name  string
	}
	var scored []sc
	for _, name := range allNames {
		if name == target {
			continue
		}
		overlap := 0
		for _, t := range nameTokens(name) {
			if targetTok[t] {
				overlap++
			}
		}
		scored = append(scored, sc{overlap, name})
	}
	sort.Slice(scored, func(i, j int) bool {
		if scored[i].score != scored[j].score {
			return scored[i].score > scored[j].score // desc overlap
		}
		return scored[i].name < scored[j].name // then name asc
	})
	var top []string
	for _, s := range scored {
		if s.score > 0 && len(top) < 3 {
			top = append(top, s.name)
		}
	}
	if len(top) < 3 {
		inTop := map[string]bool{}
		for _, n := range top {
			inTop[n] = true
		}
		var remaining []string
		for _, s := range scored {
			if !inTop[s.name] {
				remaining = append(remaining, s.name)
			}
		}
		need := 3 - len(top)
		top = append(top, m.sample(remaining, need)...)
	}
	return top
}

// PickNameChoices builds the choice list for Image->Name (correct + distractors).
func (m *Model) PickNameChoices(correctName string) []string {
	allNames := make([]string, len(m.ImagePaths))
	for i, p := range m.ImagePaths {
		allNames[i] = filepath.Base(p)
	}
	count := m.AnswerCount()
	var choices []string
	if count == 0 { // All
		choices = append(choices, allNames...)
	} else {
		if count > len(allNames) {
			count = len(allNames)
		}
		distractors := m.similarNames(correctName, allNames)
		needed := count - 1 - len(distractors)
		if needed > 0 {
			inD := map[string]bool{}
			for _, d := range distractors {
				inD[d] = true
			}
			var remaining []string
			for _, n := range allNames {
				if !inD[n] && n != correctName {
					remaining = append(remaining, n)
				}
			}
			distractors = append(distractors, m.sample(remaining, needed)...)
		}
		// [correct] + distractors[:count-1]
		take := count - 1
		if take > len(distractors) {
			take = len(distractors)
		}
		choices = append([]string{correctName}, distractors[:take]...)
	}
	m.shuffleStrings(choices)
	return choices
}

// PickImageChoices builds the choice list for Name->Image (correct path + random others).
func (m *Model) PickImageChoices(correctPath string) []string {
	count := m.AnswerCount()
	var choices []string
	if count == 0 { // All
		choices = append(choices, m.ImagePaths...)
	} else {
		if count > len(m.ImagePaths) {
			count = len(m.ImagePaths)
		}
		choices = []string{correctPath}
		var others []string
		for _, p := range m.ImagePaths {
			if p != correctPath {
				others = append(others, p)
			}
		}
		if count > 1 {
			choices = append(choices, m.sample(others, count-1)...)
		}
	}
	m.shuffleStrings(choices)
	return choices
}

// ScoreFor returns the cumulative score for an image path.
func (m *Model) ScoreFor(path string) int {
	return m.Scores[filepath.Base(path)]
}

// SortedImagesByScore sorts by (score asc, name asc) — the practice order.
func (m *Model) SortedImagesByScore() []string {
	paths := make([]string, len(m.ImagePaths))
	copy(paths, m.ImagePaths)
	sort.Slice(paths, func(i, j int) bool {
		si, sj := m.ScoreFor(paths[i]), m.ScoreFor(paths[j])
		if si != sj {
			return si < sj
		}
		return strings.ToLower(filepath.Base(paths[i])) < strings.ToLower(filepath.Base(paths[j]))
	})
	return paths
}

// PickNextImage chooses the next image for a question: among images not yet
// answered correctly this session, prefer ones outside the recent buffer, then
// the lowest cumulative score, random among ties. Mirrors Python _pick_next_image.
func (m *Model) PickNextImage(usedCorrect map[string]bool, recent []string) string {
	var remaining []string
	for _, p := range m.ImagePaths {
		if !usedCorrect[p] {
			remaining = append(remaining, p)
		}
	}
	if len(remaining) == 0 {
		return ""
	}
	recentSet := map[string]bool{}
	for _, p := range recent {
		recentSet[p] = true
	}
	var candidates []string
	for _, p := range remaining {
		if !recentSet[p] {
			candidates = append(candidates, p)
		}
	}
	if len(candidates) == 0 {
		candidates = remaining
	}
	minScore := m.ScoreFor(candidates[0])
	for _, p := range candidates {
		if s := m.ScoreFor(p); s < minScore {
			minScore = s
		}
	}
	var lowest []string
	for _, p := range candidates {
		if m.ScoreFor(p) == minScore {
			lowest = append(lowest, p)
		}
	}
	return lowest[m.rng.Intn(len(lowest))]
}

// UpdateImageScore adds delta to an image's score and persists.
func (m *Model) UpdateImageScore(path string, delta int) {
	key := filepath.Base(path)
	m.Scores[key] += delta
	m.SaveScores()
}

// sample returns up to k random distinct elements of pool (order randomized),
// matching random.sample semantics (no repeats, k capped at len).
func (m *Model) sample(pool []string, k int) []string {
	if k <= 0 || len(pool) == 0 {
		return nil
	}
	if k > len(pool) {
		k = len(pool)
	}
	idx := m.rng.Perm(len(pool))[:k]
	out := make([]string, k)
	for i, j := range idx {
		out[i] = pool[j]
	}
	return out
}

func (m *Model) shuffleStrings(s []string) {
	m.rng.Shuffle(len(s), func(i, j int) { s[i], s[j] = s[j], s[i] })
}
