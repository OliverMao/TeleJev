"""Autoregressive generative baseline for comparison with direct readout.

Uses ``model.generate`` (greedy decoding) to emit the whole answer sequence token
by token. There is intentionally no manual decoding loop here -- ``generate``
performs the autoregressive steps internally.
"""

from __future__ import annotations

import json
import re
import time

from .core import load_image
from .direct import _apply_chat_template

SYSTEM_PROMPT = (
    "You are a surveillance analyst. For every criterion, decide whether the "
    "attached image contains that behaviour. Answer for each criterion, in the "
    "given order, with only a JSON array of \"yes\"/\"no\" strings and no "
    "explanation or reasoning."
)

_ANSWER_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def build_messages(state, criteria, has_image: bool) -> list[dict]:
    """One prompt that asks the model to answer every criterion in order."""
    listing = [
        {"id": criterion.get("id", f"criterion-{index}"), "criterion": criterion["question"]}
        for index, criterion in enumerate(criteria)
    ]
    text = (
        "Evidence: " + json.dumps(state, ensure_ascii=False) + "\n"
        "Criteria (answer in this exact order):\n"
        + json.dumps(listing, ensure_ascii=False)
        + f'\nReturn a JSON array of {len(criteria)} strings, each "yes" or "no".'
    )
    content = [{"type": "image"}, {"type": "text", "text": text}] if has_image else text
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def generate_answers(model, tokenizer, processor, state, criteria, image_ref=None, max_new_tokens=None):
    """Run a full greedy generation and return decoded text, labels, and timing."""
    import torch

    if not isinstance(criteria, list) or not criteria:
        raise ValueError("criteria must be a nonempty list")
    started = time.perf_counter()
    image = load_image(image_ref) if image_ref else None
    has_image = image is not None
    if has_image and processor is None:
        raise ValueError("Image input requires a multimodal processor for this model")

    messages = build_messages(state, criteria, has_image)
    apply = processor.apply_chat_template if has_image else tokenizer.apply_chat_template
    text = _apply_chat_template(apply, messages)

    if has_image:
        inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    else:
        encoded = tokenizer(text, return_tensors="pt")
        inputs = {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}

    device = next(model.parameters()).device
    inputs = {key: (value.to(device) if hasattr(value, "to") else value) for key, value in inputs.items()}
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    limit = max_new_tokens or (8 * len(criteria) + 16)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    encode_seconds = time.perf_counter() - started

    sync = lambda: torch.cuda.synchronize(device) if device.type == "cuda" else None
    model.eval()
    with torch.inference_mode():
        sync()
        mark = time.perf_counter()
        output = model.generate(
            **inputs,
            max_new_tokens=limit,
            do_sample=False,          # greedy, deterministic
            use_cache=True,
            pad_token_id=pad,
        )
        sync()
        generate_seconds = time.perf_counter() - mark

    new_ids = output[0, prompt_tokens:]
    new_tokens = int(new_ids.shape[-1])
    generated = tokenizer.decode(new_ids, skip_special_tokens=True)
    answers = [match.group(1).lower() for match in _ANSWER_RE.finditer(generated)][: len(criteria)]
    return {
        "text": generated,
        "answers": answers,
        "prompt_tokens": prompt_tokens,
        "new_tokens": new_tokens,
        "encode_seconds": encode_seconds,
        "generate_seconds": generate_seconds,
        "total_seconds": time.perf_counter() - started,
        "tokens_per_second": (new_tokens / generate_seconds) if generate_seconds > 0 else None,
        "has_image": has_image,
    }