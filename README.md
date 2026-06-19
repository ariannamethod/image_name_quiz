# Image Name Quiz — Arianna Method

A study game that connects names to faces, rebuilt around a question the original never asked: *can the machine actually see the picture?* In this version it can, and it does it entirely on your own machine — a pure-C vision-language model on our [notorch](https://github.com/ariannamethod/notorch) runtime, with no cloud, no API, and no Python in the loop. You show it a face, it says what it sees in plain English, and you guess the name from its description alone.

This project began as [anniebelkin/image_name_quiz](https://github.com/anniebelkin/image_name_quiz), a desktop study tool where the answer to each question is simply the image's filename. That original is the seed everything here grew from; the Arianna Method version keeps the idea and replaces the machinery.

## The engine — SmolVLM-256M in pure C

`engine/` is a from-scratch C implementation of **SmolVLM-256M** (`HuggingFaceTB/SmolVLM-256M-Instruct`) running on notorch. Image in, sentence out. A picture is normalized to a single 512×512 frame, passed through a 12-layer SigLIP vision tower that turns 16×16 patches into 1024 vectors, compressed by a pixel-shuffle connector into 64 visual tokens projected into the text model's dimension, and spliced onto the `<image>` placeholders of a SmolLM2-135M llama decoder, which then writes the description one word at a time. Everything runs through portable BLAS (Accelerate on macOS, OpenBLAS on Linux); weights are held as f16 and dequantized to f32 only in small slices at multiply time, so the whole model lives in roughly 744 MB of RAM. The text decoder matches `llama.cpp` token-for-token on plain text; on the 14 sample faces, 6 captions match the `llama-mtmd-cli` reference exactly and all 14 are coherent and correct, the rest differing by a phrase from the f32-vs-f16 numerics in the vision path.

The model weights are not committed — they are fetched from `ggml-org/SmolVLM-256M-Instruct-GGUF` into `engine/models/`.

```bash
cd engine
clang -O2 -DUSE_BLAS -DACCELERATE -o smolvlm smolvlm.c notorch.c gguf.c bpe.c vision.c -lm -framework Accelerate
./smolvlm models/SmolVLM-256M-Instruct-f16.gguf \
  --image "../img/Alex Cox.jpg" --mmproj models/mmproj-SmolVLM-256M-Instruct-f16.gguf \
  -p "Describe this image in one sentence." -n 48
```

## The quiz — Go + Ebiten

`quiz/` is the game itself, rewritten in Go on the Ebiten game engine — a single self-contained binary, no Python and no external runtime. It plays four modes:

- **Image → Name** — see the face, pick the name.
- **Name → Image** — see the name, pick the face.
- **Reveal Grid** — uncover a face patch by patch, weakest-remembered faces first.
- **Describe → Name** — see only the engine's local description of a hidden face and name it from that alone. This is where the model earns its place: the captions come from `engine/smolvlm`, cached once into `quiz/captions.json`, so play itself is instant.

Scoring, distractor choice (similar names share name tokens), and the lowest-score / recent-buffer question picker are carried over faithfully from the original; scores and window settings persist to JSON next to the game.

```bash
cd quiz
go run . -captions   # build the description cache once (needs the engine + models)
go run .             # play
```

## Status

The engine is end-to-end working; the quiz plays all four modes and its game logic is covered by Go tests. The original `image_quiz.py` is kept in the tree as the upstream reference. Build logs and the phase-by-phase record live with the engine.

## Credit

Original concept and the first implementation: **anniebelkin** — [image_name_quiz](https://github.com/anniebelkin/image_name_quiz). The sample faces in `img/` are AI-generated (from `thispersondoesnotexist.com`) with names from `name-generator.org.uk`, used for testing. The vision-language engine, the Go rewrite, and the notorch runtime are Arianna Method.

## License

See [LICENSE](LICENSE).
