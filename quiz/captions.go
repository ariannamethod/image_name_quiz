// captions.go — local SmolVLM caption cache for the "Describe -> Name" mode.
// Runs the pure-C engine (../engine/smolvlm) once per image via os/exec and
// caches the one-sentence descriptions to quiz/captions.json. During play the
// game reads the cache only — the engine is not needed at runtime.
package main

import (
	"bufio"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
)

const captionPrompt = "Describe this image in one sentence."

type captionStore struct {
	mu       sync.Mutex
	m        map[string]string // basename -> caption
	path     string
	progress int32 // atomic: images processed this build
	total    int32 // atomic: images to process
	building int32 // atomic bool
	errMsg   string
}

func newCaptionStore(baseDir string) *captionStore {
	cs := &captionStore{m: map[string]string{}, path: filepath.Join(baseDir, "quiz", "captions.json")}
	cs.load()
	return cs
}

func (cs *captionStore) load() {
	data, err := os.ReadFile(cs.path)
	if err != nil {
		return
	}
	raw := map[string]string{}
	if json.Unmarshal(data, &raw) == nil {
		cs.mu.Lock()
		cs.m = raw
		cs.mu.Unlock()
	}
}

func (cs *captionStore) save() {
	cs.mu.Lock()
	data, err := json.MarshalIndent(cs.m, "", "  ")
	cs.mu.Unlock()
	if err == nil {
		_ = os.WriteFile(cs.path, data, 0o644)
	}
}

func (cs *captionStore) get(name string) string {
	cs.mu.Lock()
	defer cs.mu.Unlock()
	return cs.m[name]
}

func (cs *captionStore) haveAll(paths []string) bool {
	cs.mu.Lock()
	defer cs.mu.Unlock()
	for _, p := range paths {
		if _, ok := cs.m[filepath.Base(p)]; !ok {
			return false
		}
	}
	return len(paths) > 0
}

func (cs *captionStore) isBuilding() bool { return atomic.LoadInt32(&cs.building) == 1 }
func (cs *captionStore) prog() (int, int) {
	return int(atomic.LoadInt32(&cs.progress)), int(atomic.LoadInt32(&cs.total))
}

// findEngine locates the engine binary + the two GGUF files relative to baseDir.
func findEngine(baseDir string) (bin, mainGGUF, mmproj string, ok bool) {
	eng := filepath.Join(baseDir, "engine")
	bin = filepath.Join(eng, "smolvlm")
	if !fileExists(bin) {
		return "", "", "", false
	}
	entries, err := os.ReadDir(filepath.Join(eng, "models"))
	if err != nil {
		return "", "", "", false
	}
	for _, e := range entries {
		n := e.Name()
		if !strings.HasSuffix(n, ".gguf") {
			continue
		}
		full := filepath.Join(eng, "models", n)
		if strings.HasPrefix(n, "mmproj") {
			mmproj = full
		} else {
			mainGGUF = full
		}
	}
	ok = mainGGUF != "" && mmproj != ""
	return bin, mainGGUF, mmproj, ok
}

func fileExists(p string) bool {
	info, err := os.Stat(p)
	return err == nil && !info.IsDir()
}

// build runs the engine over every image lacking a cached caption (synchronous).
// Safe to call from a goroutine; the map + file writes are mutex-guarded.
func (cs *captionStore) build(baseDir string, images []string) {
	if !atomic.CompareAndSwapInt32(&cs.building, 0, 1) {
		return // already building
	}
	defer atomic.StoreInt32(&cs.building, 0)

	bin, mainGGUF, mmproj, ok := findEngine(baseDir)
	if !ok {
		cs.errMsg = "engine not found — build engine/smolvlm and fetch the GGUF models first"
		return
	}
	cs.errMsg = ""
	atomic.StoreInt32(&cs.total, int32(len(images)))
	atomic.StoreInt32(&cs.progress, 0)
	for _, img := range images {
		name := filepath.Base(img)
		if cs.get(name) == "" {
			if cap := runCaption(bin, mainGGUF, mmproj, img); cap != "" {
				cs.mu.Lock()
				cs.m[name] = cap
				cs.mu.Unlock()
				cs.save()
			}
		}
		atomic.AddInt32(&cs.progress, 1)
	}
}

func (cs *captionStore) buildAsync(baseDir string, images []string) {
	if cs.isBuilding() {
		return
	}
	go cs.build(baseDir, images)
}

// runCaption invokes the engine on one image and extracts the OURS: "..." line.
func runCaption(bin, mainGGUF, mmproj, img string) string {
	cmd := exec.Command(bin, mainGGUF, "--image", img, "--mmproj", mmproj, "-p", captionPrompt, "-n", "48")
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	sc := bufio.NewScanner(strings.NewReader(string(out)))
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if strings.HasPrefix(line, "OURS:") {
			return strings.TrimSpace(parseQuoted(line))
		}
	}
	return ""
}

// parseQuoted returns the text between the first and last double quote.
func parseQuoted(s string) string {
	i := strings.Index(s, "\"")
	j := strings.LastIndex(s, "\"")
	if i >= 0 && j > i {
		return s[i+1 : j]
	}
	return ""
}
