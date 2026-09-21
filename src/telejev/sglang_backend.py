"""SGLang HTTP backend: Jev-style logit readout and generation on SGLang.

Talks to a running SGLang server (OpenAI-compatible ``/v1/chat/completions``) so
that both the direct option readout and the autoregressive path use the same
optimized runtime (fused kernels, CUDA graphs, Radix Cache). Jev-style readout is
prefill-only: ``max_tokens=1`` plus ``logprobs`` over the option labels. Repeated
image/state prefixes are reused by SGLang's Radix Cache.

Based on the approach used by https://github.com/Yinsongxu/LLM2Jev
(SGLang ``score`` with ``label_token_ids`` + shared-prefix staging).
"""

from __future__ import annotations

import base64
import json
import mimetypes
import time
import urllib.error
import urllib.request
from pathlib import Path

from .autoregressive import SYSTEM_PROMPT, _ANSWER_RE
from .core import LETTERS, direct_messages

DEFAULT_SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)


class SGLangError(RuntimeError):
    """Raised when the SGLang server errors or returns an unusable response."""


def _image_url(reference: str) -> str:
    """Turn a local path / URL / data URI into an OpenAI image_url value."""
    if reference.startswith(("data:", "http://", "https://")):
        return reference
    path = Path(reference)
    if not path.is_file():
        raise ValueError(f"Image file not found: {reference}")
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def _map_label_logprobs(top_logprobs: list[dict], labels: list[str]) -> dict[str, float | None]:
    """Match decoded label tokens (e.g. ``" A"``) to the requested labels."""
    observed: dict[str, float] = {}
    for item in top_logprobs:
        key = str(item.get("token", "")).strip().upper()
        value = float(item.get("logprob", float("-inf")))
        if key and (key not in observed or value > observed[key]):
            observed[key] = value
    return {label: observed.get(label.strip().upper()) for label in labels}


class SGLangBackend:
    """Direct option readout and generation against one SGLang server."""

    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.requests = 0

    def _post(self, path: str, payload: dict) -> tuple[dict, float]:
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(self.base_url + path, data=data, headers=headers)
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read())
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:500]
            raise SGLangError(f"SGLang HTTP {error.code}: {detail}") from error
        elapsed = time.perf_counter() - started
        self.requests += 1
        return body, elapsed

    def _content(self, text: str, image: str | None):
        if not image:
            return text
        return [
            {"type": "image_url", "image_url": {"url": _image_url(image)}},
            {"type": "text", "text": text},
        ]

    # ---- Jev-style readout -------------------------------------------------
    def score_options(self, state, question: str, options: list[dict], image: str | None = None):
        """Prefill-only readout of option-letter logprobs for one decision."""
        messages = direct_messages({"id": "x", "state": state, "question": question, "options": options})
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": DEFAULT_SYSTEM},
                {"role": "user", "content": self._content(messages[-1]["content"], image)},
            ],
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": 20,
        }
        body, elapsed = self._post("/v1/chat/completions", payload)
        try:
            content = body["choices"][0]["logprobs"]["content"][0]
        except (KeyError, IndexError, TypeError) as error:
            raise SGLangError("SGLang response did not include first-token logprobs") from error
        letters = LETTERS[: len(options)]
        mapped = _map_label_logprobs(content.get("top_logprobs", []), list(letters))
        # Include the chosen token itself in case it is absent from top_logprobs.
        chosen = str(content.get("token", "")).strip().upper()
        if chosen in mapped and mapped[chosen] is None:
            mapped[chosen] = float(content.get("logprob", float("-inf")))
        available = {label: value for label, value in mapped.items() if value is not None}
        if not available:
            raise SGLangError("None of the option letters appeared in SGLang top_logprobs")
        maximum = max(available.values())
        weights = {label: pow(2.718281828459045, value - maximum) for label, value in available.items()}
        total = sum(weights.values())
        probabilities = {label: weights[label] / total for label in letters if label in weights}
        usage = body.get("usage", {}) or {}
        return {
            "letters": letters,
            "probabilities": probabilities,
            "logprobs": {label: mapped[label] for label in letters},
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "seconds": elapsed,
        }

    def score_row(self, row: dict) -> dict:
        scored = self.score_options(row["state"], row["question"], row["options"], row.get("image"))
        option_ids = [option["id"] for option in row["options"]]
        probabilities = [scored["probabilities"].get(letter, 0.0) for letter in scored["letters"]]
        return {
            "id": row["id"],
            "option_ids": option_ids,
            "probabilities": probabilities,
            "option_logits": [scored["logprobs"].get(letter) for letter in scored["letters"]],
            "has_image": bool(row.get("image")),
            "image_tokens": 0,
            "input_tokens": scored["prompt_tokens"],
            "prompt_version": "sglang-prefill-logp",
            "probability_status": "conditional option score from SGLang top_logprobs",
        }

    def score_batch(self, state, image, criteria, **_ignored):
        """Score many criteria, reusing SGLang's Radix Cache for the shared prefix."""
        if not isinstance(criteria, list) or not criteria:
            raise ValueError("criteria must be a nonempty list")
        started = time.perf_counter()
        results = []
        first_seconds = 0.0
        rest_seconds = 0.0
        input_tokens = 0
        for index, criterion in enumerate(criteria):
            scored = self.score_options(state, criterion["question"], criterion["options"], image)
            results.append(
                {
                    "id": criterion.get("id") or f"criterion-{index}",
                    "option_ids": [option["id"] for option in criterion["options"]],
                    "probabilities": [scored["probabilities"].get(letter, 0.0) for letter in scored["letters"]],
                    "option_logits": [scored["logprobs"].get(letter) for letter in scored["letters"]],
                    "has_image": bool(image),
                    "image_tokens": 0,
                    "input_tokens": scored["prompt_tokens"],
                    "prompt_version": "sglang-prefill-logp",
                    "probability_status": "conditional option score from SGLang top_logprobs",
                }
            )
            input_tokens += scored["prompt_tokens"]
            if index == 0:
                first_seconds = scored["seconds"]
            else:
                rest_seconds += scored["seconds"]
        timing = {
            "total_seconds": time.perf_counter() - started,
            "prefill_seconds": first_seconds,
            "suffix_seconds": rest_seconds,
            "encode_seconds": 0.0,
            "image_seconds": 0.0,
            "batch_size": len(results),
            "prefix_tokens": results[0]["input_tokens"] if results else 0,
            "true_suffix_tokens": 0,
            "prefill_passes": 1,
            "suffix_passes": max(len(results) - 1, 0),
            "forward_passes": len(results),
            "requests": self.requests,
            "image": bool(image),
            "backend": "sglang",
            "input_tokens": input_tokens,
        }
        return results, timing

    # ---- generation --------------------------------------------------------
    def generate(self, state, image, criteria, max_new_tokens=None):
        if not isinstance(criteria, list) or not criteria:
            raise ValueError("criteria must be a nonempty list")
        listing = [
            {"id": c.get("id", f"criterion-{i}"), "criterion": c["question"]}
            for i, c in enumerate(criteria)
        ]
        text = (
            "Evidence: " + json.dumps(state, ensure_ascii=False) + "\n"
            "Criteria (answer in this exact order):\n"
            + json.dumps(listing, ensure_ascii=False)
            + f'\nReturn a JSON array of {len(criteria)} strings, each "yes" or "no".'
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self._content(text, image)},
            ],
            "max_tokens": max_new_tokens or (8 * len(criteria) + 16),
            "temperature": 0.0,
        }
        body, elapsed = self._post("/v1/chat/completions", payload)
        choice = body["choices"][0]
        generated = choice["message"].get("content", "") or ""
        usage = body.get("usage", {}) or {}
        new_tokens = int(usage.get("completion_tokens", 0) or 0)
        answers = [m.group(1).lower() for m in _ANSWER_RE.finditer(generated)][: len(criteria)]
        return {
            "text": generated,
            "answers": answers,
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "new_tokens": new_tokens,
            "encode_seconds": 0.0,
            "generate_seconds": elapsed,
            "total_seconds": elapsed,
            "tokens_per_second": (new_tokens / elapsed) if elapsed > 0 else None,
            "prefill_passes": 1,
            "decode_passes": new_tokens,
            "forward_passes": 1 + new_tokens,
            "has_image": bool(image),
            "backend": "sglang",
        }