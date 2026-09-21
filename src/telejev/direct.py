"""Direct categorical decision readout from native next-token logits.

Text-only rows score from the chat prompt. When a row carries an optional
``image`` reference and the loaded model exposes a multimodal processor, the
image is attached to the user turn and the same letter logits are read.
"""

from __future__ import annotations

import inspect
import time

from .core import LETTERS, digest, direct_messages, load_image, softmax, validate_row

PROMPT_VERSION = "direct-options-v1"


def _slot_ids(tokenizer, count: int) -> list[int]:
    result = []
    for letter in LETTERS[:count]:
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) != 1 or tokenizer.decode(encoded) != letter:
            raise ValueError(f"Answer slot {letter!r} is not one exact round-trip token")
        result.append(encoded[0])
    if len(result) != len(set(result)):
        raise ValueError("Answer-slot tokens collide")
    return result


def _apply_chat_template(apply, messages: list[dict]) -> str:
    try:
        return apply(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return apply(messages, tokenize=False, add_generation_prompt=True)


def _image_messages(row: dict) -> list[dict]:
    messages = direct_messages(row)
    return messages[:-1] + [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": messages[-1]["content"]},
            ],
        }
    ]


def _forward(model, inputs):
    parameters = inspect.signature(model.forward).parameters
    kwargs = dict(inputs, use_cache=False, return_dict=True)
    if "logits_to_keep" in parameters:
        kwargs["logits_to_keep"] = 1
    return model(**kwargs).logits[:, -1, :]


def encode_prompt(tokenizer, row: dict, max_tokens: int, processor=None, image=None):
    """Return ``(ids, slots, prompt_hash, extras)``.

    ``extras`` is ``None`` for text-only rows, otherwise a processor batch
    (``input_ids``/``pixel_values``/...) that must be fed to the model verbatim.
    """
    validate_row(row)
    if image is not None:
        if processor is None:
            raise ValueError("Image input requires a multimodal processor for this model")
        text = _apply_chat_template(processor.apply_chat_template, _image_messages(row))
        extras = processor(text=[text], images=[image], return_tensors="pt", padding=True)
        ids = extras["input_ids"][0].tolist()
        if not ids or len(ids) > max_tokens:
            raise ValueError(f"Row {row['id']}: {len(ids)} input tokens exceed limit {max_tokens}")
        return ids, _slot_ids(tokenizer, len(row["options"])), digest(text), extras

    prompt = _apply_chat_template(tokenizer.apply_chat_template, direct_messages(row))
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not ids or len(ids) > max_tokens:
        raise ValueError(f"Row {row['id']}: {len(ids)} input tokens exceed limit {max_tokens}; no truncation allowed")
    slots = _slot_ids(tokenizer, len(row["options"]))
    for letter, token in zip(LETTERS, slots):
        if tokenizer.encode(prompt + letter, add_special_tokens=False) != ids + [token]:
            raise ValueError(f"Answer boundary changes tokenization for slot {letter}")
    return ids, slots, digest(prompt), None


def score(model, tokenizer, row: dict, metadata: dict, max_tokens: int = 4096, processor=None) -> dict:
    import torch

    started = time.perf_counter()
    image = load_image(row["image"]) if row.get("image") else None
    ids, slots, prompt_hash, extras = encode_prompt(tokenizer, row, max_tokens, processor, image)
    device = next(model.parameters()).device
    if extras is not None:
        inputs = {key: (value.to(device) if hasattr(value, "to") else value) for key, value in extras.items()}
    else:
        inputs = {
            "input_ids": torch.tensor([ids], dtype=torch.long, device=device),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long, device=device),
        }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    forward_start = time.perf_counter()
    with torch.inference_mode():
        vocabulary = _forward(model, inputs)[0].float()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    selected = vocabulary[slots].cpu().tolist()
    return {
        "id": row["id"],
        "option_ids": [option["id"] for option in row["options"]],
        "probabilities": softmax(selected),
        "option_logits": selected,
        "has_image": image is not None,
        "input_tokens": len(ids),
        "forward_seconds": time.perf_counter() - forward_start,
        "total_seconds": time.perf_counter() - started,
        "prompt_sha256": prompt_hash,
        "prompt_version": PROMPT_VERSION,
        "model": metadata,
        "readout": "native full-vocabulary last-position logits restricted to declared answer slots",
        "probability_status": "conditional option score; uncalibrated as decision confidence",
    }