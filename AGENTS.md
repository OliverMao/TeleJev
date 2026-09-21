# Repository instructions

- Run commands from the repository root: install dependencies with `pip install -r requirements.txt` and run the scorer directly via `python run.py`.
- This repository contains CUDA (PyTorch) inference only. Expose exactly one CUDA GPU per scorer process.
- Do not commit model weights or caches.
- Outputs are create-only; use a new output path per run.