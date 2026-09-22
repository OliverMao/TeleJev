# Repository instructions

- Run commands from the repository root. Dependencies: `pip install -r requirements.txt`.
- The library lives in `src/telejev/` (import `telejev`). Entries: `python run.py` (JSONL
  scorer) and `python serve.py` (HTTP server).
- Two inference backends: local PyTorch/CUDA (`--backend local`) and an OpenAI-compatible
  SGLang/vLLM server (`--backend sglang|vllm --server-url ...`). Expose exactly one CUDA GPU
  per local scorer process.
- Non-package material: `demo/` (front-end pages), `benchmarks/` (comparison scripts),
  `examples/` (client examples), `tests/` (offline interface test).
- Validate with `python tests/test_api.py` (no GPU needed).
- Do not commit model weights or caches.
- Outputs are create-only; use a new output path per run.