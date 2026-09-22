"""OpenAI-compatible wrapper around the Jev-style option readout.

The client sends a normal chat request (``messages`` with text and/or an image)
that already instructs the model which answer to give, plus the set of allowed
answers. Instead of generating tokens we run one prefill-only logprob read over
those answers and return a normal ``chat.completion`` whose assistant content is
the structured decision.

How the allowed answers are supplied (either works):

- ``response_format`` json_schema with one ``enum`` property::

      {"type": "json_schema", "json_schema": {"name": "decision", "strict": true,
       "schema": {"type": "object", "properties": {"department": {"enum": ["shipping", "billing"]}},
                  "required": ["department"]}}}

  The property name becomes the key of the returned object.

- a top-level ``options`` list::

      {"messages": [...], "options": ["yes", "no"]}

  The returned object uses the key ``"choice"``.

The response also carries ``telejev.probabilities`` (extra fields are ignored by
OpenAI clients) and the per-label logprobs under ``choices[0].logprobs``.
"""

from __future__ import annotations

import hashlib
import json
import time

DEFAULT_KEY = "choice"


def extract_labels(body: dict) -> tuple[list[str], str]:
    """Return (allowed labels, output key) from the request body."""
    response_format = body.get("response_format")
    if isinstance(response_format, dict) and response_format.get("type") == "json_schema":
        json_schema = response_format.get("json_schema") or {}
        schema = json_schema.get("schema") or {}
        properties = schema.get("properties") or {}
        for name, prop in properties.items():
            enum = (prop or {}).get("enum")
            if isinstance(enum, list) and enum:
                return [str(item) for item in enum], str(name)

    for field in ("options", "labels"):
        value = body.get(field)
        if isinstance(value, list) and value:
            return [str(item) for item in value], DEFAULT_KEY

    raise ValueError(
        "Provide allowed answers via response_format json_schema enum, or a nonempty 'options' list"
    )


def build_completion(model: str, result: dict, key: str, labels: list[str]) -> dict:
    """Wrap a label readout into an OpenAI chat.completion with structured content."""
    probabilities = result.get("probabilities", {})
    logprobs = result.get("logprobs", {})
    chosen = max(probabilities, key=probabilities.get) if probabilities else None
    content = json.dumps({key: chosen}, ensure_ascii=False, sort_keys=False)
    top_logprobs = []
    for label in labels:
        value = logprobs.get(label)
        top_logprobs.append(
            {"token": label, "logprob": float(value) if value is not None else float("-inf")}
        )
    chosen_logprob = logprobs.get(chosen)
    usage = {
        "prompt_tokens": int(result.get("prompt_tokens", 0) or 0),
        "completion_tokens": 0,
        "total_tokens": int(result.get("prompt_tokens", 0) or 0),
    }
    return {
        "id": "chatcmpl-" + hashlib.sha256(content.encode()).hexdigest()[:24],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
                "logprobs": {
                    "content": [
                        {
                            "token": chosen or "",
                            "logprob": float(chosen_logprob) if chosen_logprob is not None else 0.0,
                            "top_logprobs": top_logprobs,
                        }
                    ]
                },
            }
        ],
        "usage": usage,
        "telejev": {"probabilities": probabilities, "chosen": chosen, "key": key},
    }