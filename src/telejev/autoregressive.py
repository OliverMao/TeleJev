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
    "You are a surveillance analyst. Decide from the attached image whether any "
    "person is present and which of the listed behaviours are present. Answer "
    "with only a JSON object, no explanation, no markdown."
)


def _behavior_labels(criteria: list[dict]) -> list[str]:
    return [criterion.get("label") or criterion.get("id") for criterion in criteria if criterion.get("id") != "person"]


def build_prompt_text(state, criteria) -> str:
    """Prompt text asking for the final {has_person, violations} object."""
    lines = ["Evidence: " + json.dumps(state, ensure_ascii=False), "Decide from the image:"]
    for criterion in criteria:
        label = criterion.get("label") or criterion.get("id")
        field = "has_person (0 or 1)" if criterion.get("id") == "person" else f'violation "{label}"'
        lines.append(f"- {field}: {criterion['question']}")
    allowed = ", ".join(json.dumps(label, ensure_ascii=False) for label in _behavior_labels(criteria))
    lines.append(f"Allowed violations: [{allowed}]")
    lines.append(
        'Respond with ONLY a JSON object exactly like {"has_person": 0, "violations": ["..."]}. '
        "has_person=1 if any person is present; list only the violation names that are present."
    )
    return "\n".join(lines)


def build_messages(state, criteria, has_image: bool) -> list[dict]:
    """One prompt that asks for the final {has_person, violations} object."""
    text = build_prompt_text(state, criteria)
    content = [{"type": "image"}, {"type": "text", "text": text}] if has_image else text
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def parse_output(text: str, criteria: list[dict]) -> tuple[dict, list[str]]:
    """Parse the generated JSON into {has_person, violations} plus per-criterion yes/no."""
    has_person = 0
    violations: list[str] = []
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            try:
                has_person = 1 if int(data.get("has_person", 0)) else 0
            except (TypeError, ValueError):
                has_person = 0
            allowed = _behavior_labels(criteria)
            raw = data.get("violations", [])
            if isinstance(raw, list):
                violations = [item for item in raw if item in allowed]
    output = {"has_person": has_person, "violations": violations}
    answers = []
    for criterion in criteria:
        if criterion.get("id") == "person":
            answers.append("yes" if has_person else "no")
        else:
            answers.append("yes" if (criterion.get("label") or criterion.get("id")) in violations else "no")
    return output, answers


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
    parsed, answers = parse_output(generated, criteria)
    return {
        "text": generated,
        "output": parsed,
        "answers": answers,
        "prompt_tokens": prompt_tokens,
        "new_tokens": new_tokens,
        "encode_seconds": encode_seconds,
        "generate_seconds": generate_seconds,
        "total_seconds": time.perf_counter() - started,
        "tokens_per_second": (new_tokens / generate_seconds) if generate_seconds > 0 else None,
        "prefill_passes": 1,
        "decode_passes": new_tokens,
        "forward_passes": 1 + new_tokens,
        "has_image": has_image,
    }