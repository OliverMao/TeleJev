"""Shared-prefix batch scoring: one image/state, many criteria.

Instead of running one full forward per criterion (re-encoding the image every
time), this pre-fills the shared state/image prefix once and then evaluates all
criteria suffixes against that cached prefix in a single batched forward: two
model forwards total, regardless of how many criteria there are.
"""

from __future__ import annotations

import inspect
import json
import time

from .core import direct_messages, load_image, softmax, validate_row
from .direct import PROMPT_VERSION, _apply_chat_template, encode_prompt
from .shared import _state_prefix, _suffix_layout


def _prefix_text(apply, state: str, has_image: bool) -> str:
    """Render the chat template truncated to the start of the evidence value."""
    row = {
        "id": "prefix-only",
        "state": state,
        "question": "prefix boundary placeholder",
        "options": [{"id": "yes", "description": "Yes"}, {"id": "no", "description": "No"}],
    }
    base = direct_messages(row)
    payload = base[-1]["content"]
    if has_image:
        messages = base[:-1] + [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": payload}]}
        ]
    else:
        messages = base
    text = _apply_chat_template(apply, messages)
    if text.count(payload) != 1:
        raise ValueError("Cannot locate the unmodified evidence payload in the chat template")
    evidence = json.dumps({"evidence": state}, ensure_ascii=False)[:-1]
    if not payload.startswith(evidence):
        raise ValueError("Evidence serialization changed")
    return text[: text.index(payload)] + evidence


def _image_prefix(processor, state: str, image):
    """Return (prefix_ids, prefix_extras) for an image+state prefix.

    The final boundary token is dropped so the suffix can be split cleanly, and
    every per-token tensor the processor returns (``input_ids``,
    ``attention_mask``, ``mm_token_type_ids``/``token_type_ids``, ...) is
    truncated by the same amount, keeping all sequence lengths aligned.
    """
    text = _prefix_text(processor.apply_chat_template, state, True)
    extras = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    full = extras["input_ids"].shape[-1]
    keep = full - 1
    if keep <= 0:
        raise ValueError("Empty image prefix")
    sliced = {}
    for key, value in extras.items():
        is_per_token = (
            hasattr(value, "shape")
            and hasattr(value, "dim")
            and value.dim() >= 2
            and value.shape[-1] == full
        )
        sliced[key] = value[..., :keep] if is_per_token else value
    return sliced["input_ids"][0].tolist(), sliced


def _repeat_cache(cache, count: int, device) -> None:
    """Duplicate a single-item cache into ``count`` identical branches."""
    import torch

    reorder = getattr(cache, "reorder_cache", None)
    if callable(reorder):
        reorder(torch.zeros(count, dtype=torch.long, device=device))
        return
    batched = getattr(cache, "batch_repeat_interleave", None)
    if callable(batched):
        batched(count)
        return
    raise RuntimeError("Native cache cannot be duplicated for shared batch scoring")


def _forward_signature(model):
    parameters = inspect.signature(model.forward).parameters
    if "logits_to_keep" not in parameters and hasattr(model, "get_base_model"):
        parameters = inspect.signature(model.get_base_model().forward).parameters
    return parameters


def _prefill(model, prefix_ids, prefix_extras, device):
    import torch

    if prefix_extras is not None:
        prefill = {
            key: (value.to(device) if hasattr(value, "to") else value)
            for key, value in prefix_extras.items()
        }
    else:
        prefill = {}
    prefill["input_ids"] = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    prefill["attention_mask"] = torch.ones((1, len(prefix_ids)), dtype=torch.long, device=device)
    sync = lambda: torch.cuda.synchronize(device) if device.type == "cuda" else None
    sync()
    mark = time.perf_counter()
    with torch.inference_mode():
        output = model(**prefill, use_cache=True, return_dict=True, logits_to_keep=1)
    cache = output.past_key_values
    del output
    sync()
    prefill_seconds = time.perf_counter() - mark
    if cache is None or cache.get_seq_length() != len(prefix_ids):
        raise RuntimeError("Invalid native prefix cache")
    return cache, prefill_seconds


def _result_row(row, slots, prompt_hash, selected, has_image, model_metadata, serving_config):
    return {
        "id": row["id"],
        "option_ids": [option["id"] for option in row["options"]],
        "probabilities": softmax(selected),
        "option_logits": selected,
        "has_image": has_image,
        "input_tokens": None,  # filled by the caller
        "prompt_sha256": prompt_hash,
        "prompt_version": PROMPT_VERSION,
        "model": {**model_metadata, "serving_config": serving_config},
        "readout": "native shared-prefix criteria logits",
        "probability_status": "conditional option score; uncalibrated as decision confidence",
    }


def _shared_suffix(model, rows, encoded, prefix_ids, pad, cache, device, prefill_seconds, has_image, metadata):
    import torch

    layout, ends = _suffix_layout([ids[len(prefix_ids):] for ids, _, _, _ in encoded], len(prefix_ids), pad)
    selected_positions = sorted(set(ends))
    _repeat_cache(cache, len(rows), device)
    inputs = {key: torch.tensor(value, dtype=torch.long, device=device) for key, value in layout.items()}
    sync = lambda: torch.cuda.synchronize(device) if device.type == "cuda" else None
    sync()
    mark = time.perf_counter()
    with torch.inference_mode():
        output = model(
            **inputs,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
            logits_to_keep=torch.tensor(selected_positions, dtype=torch.long, device=device),
        )
    sync()
    suffix_seconds = time.perf_counter() - mark
    results = []
    for index, (row, (ids, slots, prompt_hash, _)) in enumerate(zip(rows, encoded)):
        vocabulary = output.logits[index, selected_positions.index(ends[index]), :].float()
        selected = vocabulary[slots].cpu().tolist()
        result = _result_row(row, slots, prompt_hash, selected, has_image, metadata, "native-shared-prefix-batch-v1")
        result["input_tokens"] = len(ids)
        results.append(result)
    del output
    timing = {
        "batch_size": len(rows),
        "prefix_tokens": len(prefix_ids),
        "true_suffix_tokens": sum(len(ids) - len(prefix_ids) for ids, _, _, _ in encoded),
        "padded_suffix_tokens": len(rows) * len(layout["input_ids"][0]),
        "prefill_passes": 1,
        "suffix_passes": 1,
        "forward_passes": 2,
    }
    return results, suffix_seconds, timing


def score_batch(model, tokenizer, metadata, state, image_ref, criteria, max_tokens=4096, processor=None):
    """Score many criteria over one shared state/image, returning (results, timing)."""
    import torch

    started = time.perf_counter()
    if not isinstance(criteria, list) or not criteria:
        raise ValueError("criteria must be a nonempty list")
    if not isinstance(state, str) or not state:
        raise ValueError("state must be a nonempty string")
    image_started = time.perf_counter()
    image = load_image(image_ref) if image_ref else None
    image_seconds = time.perf_counter() - image_started
    if image is not None and processor is None:
        raise ValueError("Image input requires a multimodal processor for this model")

    rows = []
    for index, criterion in enumerate(criteria):
        if not isinstance(criterion, dict):
            raise ValueError("Each criterion must be an object")
        row = {
            "id": criterion.get("id") or f"criterion-{index}",
            "state": state,
            "question": criterion["question"],
            "options": criterion["options"],
        }
        validate_row(row)
        rows.append(row)
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Criterion IDs must be unique")

    encode_started = time.perf_counter()
    encoded = [encode_prompt(tokenizer, row, max_tokens, processor, image) for row in rows]
    if image is not None:
        prefix_ids, prefix_extras = _image_prefix(processor, state, image)
    else:
        prefix_ids, prefix_extras = _state_prefix(tokenizer, state), None
    if not prefix_ids or any(
        ids[: len(prefix_ids)] != prefix_ids or len(ids) <= len(prefix_ids)
        for ids, _, _, _ in encoded
    ):
        raise ValueError("The shared state/image prefix does not match every criterion prompt")

    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad is None:
        raise ValueError("Tokenizer requires a padding or EOS token")
    if "logits_to_keep" not in _forward_signature(model):
        raise RuntimeError("Model lacks selective-position logits needed for batch scoring")
    encode_seconds = time.perf_counter() - encode_started

    device = next(model.parameters()).device
    model.eval()
    cache, prefill_seconds = _prefill(model, prefix_ids, prefix_extras, device)

    try:
        results, suffix_total, detail = _shared_suffix(
            model, rows, encoded, prefix_ids, pad, cache, device, prefill_seconds, image is not None, metadata
        )
    finally:
        del cache

    timing = {
        "inference_seconds": time.perf_counter() - started,
        "image_seconds": image_seconds,
        "encode_seconds": encode_seconds,
        "prefill_seconds": prefill_seconds,
        "suffix_seconds": suffix_total,
        "image": image is not None,
        **detail,
    }
    return results, timing