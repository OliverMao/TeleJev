"""TeleJev: runtime-defined semantic decisions with open models.

Two ways to decide, both reading the model directly:

- **Jev-style** (``score_direct`` / ``score_batch`` / ``VLLMBackend.score_batch``):
  one prefill-only forward, read the declared option logits. No token generation.
- **Autoregressive** (``generate_answers`` / ``VLLMBackend.generate``): a greedy
  generation baseline that emits the final ``{"has_person", "violations"}`` object.

Typical use::

    from telejev import load_causal_model, score_direct

    model, tokenizer, processor, metadata = load_causal_model("Qwen/Qwen3.5-4B")
    result = score_direct(model, tokenizer, row, metadata, processor=processor)

Serving::

    python serve.py --model Qwen/Qwen3.5-4B          # local torch backend
    python serve.py --backend vllm --server-url ...  # vLLM backend

Everything here is importable without loading torch; heavy imports happen lazily
inside the functions that need them.
"""

from __future__ import annotations

from .api import DecisionService, help_document, serve
from .autoregressive import build_messages, generate_answers, parse_output
from .batch import score_batch
from .core import LETTERS, load_causal_model, load_image, validate_row
from .direct import score as score_direct
from .prompt import (
    DIRECT_SYSTEM,
    GENERATION_SYSTEM,
    GLOBAL_RULES,
    build_generation_text,
    task_standard,
)
from .shared import score_shared
from .vllm_backend import VLLMBackend, VLLMError

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # model + input
    "load_causal_model",
    "load_image",
    "validate_row",
    "LETTERS",
    # local readout / generation
    "score_direct",
    "score_shared",
    "score_batch",
    "generate_answers",
    "parse_output",
    "build_messages",
    # OpenAI-compatible backend (vLLM)
    "VLLMBackend",
    "VLLMError",
    # serving
    "DecisionService",
    "serve",
    "help_document",
    # prompts
    "DIRECT_SYSTEM",
    "GENERATION_SYSTEM",
    "GLOBAL_RULES",
    "build_generation_text",
    "task_standard",
]