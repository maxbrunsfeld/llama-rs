# llama-rs

Rust bindings to llama.cpp, for macOS, with metal support, for testing and
evaluating whether it would be worthwhile to run an Llama model locally in
a Rust app.

### Setup

1. Clone llama.cpp into `vendor/llama.cpp`
2. Build llama.cpp: `LLAMA_METAL=1 make`
3. Download a llama2 model: https://huggingface.co/TheBloke/Llama-2-7B-GGML/tree/main
4. Convert the model to llama.cpp's GGUF format using the script in the `llama.cpp` repo.
