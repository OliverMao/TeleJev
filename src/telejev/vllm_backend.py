"""OpenAI-compatible backend (vLLM): Jev-style readout and generation.

Talks to a running vLLM OpenAI-compatible server (``/v1/chat/completions``) so
that both the direct option readout and the autoregressive path use the same
optimized runtime (fused kernels, CUDA graphs, prefix cache). Jev-style readout
is prefill-only: ``max_tokens=1`` plus ``logprobs`` over the option labels.
Repeated image/state prefixes are reused by vLLM's automatic prefix caching
(``--enable-prefix-caching``).

Based on the approach used by https://github.com/Yinsongxu/LLM2Jev
(shared-prefix staging for prefill-only label scoring).
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .autoregressive import parse_output
from .core import LETTERS, direct_messages
from .prompt import DIRECT_SYSTEM, GENERATION_SYSTEM, build_generation_text, task_standard

logger = logging.getLogger("telejev.vllm")


class VLLMError(RuntimeError):
    """Raised when the vLLM server errors or returns an unusable response."""


class BatchNotSupportedError(VLLMError):
    """The vLLM server has no native batched chat endpoint; callers may fan out."""

    batch_not_supported = True


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


class VLLMBackend:
    """Direct option readout and generation against one vLLM server."""

    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.requests = 0
        self._count_lock = threading.Lock()

    def _post(self, path: str, payload: dict, unsupported: tuple[int, ...] = ()) -> tuple[dict, float]:
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
            logger.warning("POST %s -> HTTP %s: %s", path, error.code, detail[:200])
            if error.code in unsupported:
                raise BatchNotSupportedError(f"vLLM HTTP {error.code} for {path}: {detail}") from error
            raise VLLMError(f"vLLM HTTP {error.code}: {detail}") from error
        elapsed = time.perf_counter() - started
        logger.debug("POST %s payload=%dB -> %.4fs", path, len(data), elapsed)
        with self._count_lock:
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
    def score_options(self, state, question: str, options: list[dict], image: str | None = None, standard: str | None = None):
        """Prefill-only readout of option-letter logprobs for one decision."""
        messages = direct_messages(
            {"id": "x", "state": state, "question": question, "options": options, "standard": standard}
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": DIRECT_SYSTEM},
                {"role": "user", "content": self._content(messages[-1]["content"], image)},
            ],
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        body, elapsed = self._post("/v1/chat/completions", payload)
        try:
            content = body["choices"][0]["logprobs"]["content"][0]
        except (KeyError, IndexError, TypeError) as error:
            raise VLLMError("vLLM response did not include first-token logprobs") from error
        letters = LETTERS[: len(options)]
        mapped = _map_label_logprobs(content.get("top_logprobs", []), list(letters))
        # Include the chosen token itself in case it is absent from top_logprobs.
        chosen = str(content.get("token", "")).strip().upper()
        if chosen in mapped and mapped[chosen] is None:
            mapped[chosen] = float(content.get("logprob", float("-inf")))
        available = {label: value for label, value in mapped.items() if value is not None}
        if not available:
            raise VLLMError("None of the option letters appeared in vLLM top_logprobs")
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
        scored = self.score_options(row["state"], row["question"], row["options"], row.get("image"), row.get("standard"))
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
            "prompt_version": "vllm-prefill-logp",
            "probability_status": "conditional option score from vLLM top_logprobs",
        }

    def score_batch(self, state, image, criteria, **_ignored):
        """Score many criteria with one batched upstream request when possible.

        ``/v1/chat/completions/batch`` carries every criterion's prompt in a single
        HTTP request; the shared image/state prefix is still prefill-cached by the
        server. Older servers without the batched endpoint fall back to concurrent
        per-criterion requests, which the server still continuous-batches.
        """
        if not isinstance(criteria, list) or not criteria:
            raise ValueError("criteria must be a nonempty list")
        started = time.perf_counter()
        conversations = []
        letters_per_criterion = []
        for criterion in criteria:
            messages = direct_messages(
                {"id": "x", "state": state, "question": criterion["question"],
                 "options": criterion["options"], "standard": task_standard(criterion)}
            )
            conversations.append([
                {"role": "system", "content": DIRECT_SYSTEM},
                {"role": "user", "content": self._content(messages[-1]["content"], image)},
            ])
            letters_per_criterion.append(list(LETTERS[: len(criterion["options"])]))

        used_batch = False
        workers = min(len(criteria), 32)
        try:
            readouts = self.score_labels_batch(conversations, letters_per_criterion)
            used_batch = True
        except BatchNotSupportedError:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                scored = list(pool.map(
                    lambda criterion: self.score_options(
                        state, criterion["question"], criterion["options"], image, task_standard(criterion)
                    ),
                    criteria,
                ))
        if used_batch:
            scored = [
                {
                    "letters": letters,
                    "probabilities": readout["probabilities"],
                    "logprobs": readout["logprobs"],
                    "prompt_tokens": readout.get("prompt_tokens", 0),
                    "seconds": readout.get("seconds", 0.0),
                    "batch_usage": readout.get("batch_usage"),
                }
                for readout, letters in zip(readouts, letters_per_criterion)
            ]
        total_seconds = time.perf_counter() - started
        results = []
        input_tokens = 0
        request_seconds = []
        batch_usage = None
        for index, (criterion, item) in enumerate(zip(criteria, scored)):
            results.append(
                {
                    "id": criterion.get("id") or f"criterion-{index}",
                    "option_ids": [option["id"] for option in criterion["options"]],
                    "probabilities": [item["probabilities"].get(letter, 0.0) for letter in item["letters"]],
                    "option_logits": [item["logprobs"].get(letter) for letter in item["letters"]],
                    "has_image": bool(image),
                    "image_tokens": 0,
                    "input_tokens": item["prompt_tokens"],
                    "prompt_version": "vllm-prefill-logp",
                    "probability_status": "conditional option score from vLLM top_logprobs",
                }
            )
            input_tokens += int(item["prompt_tokens"] or 0)
            request_seconds.append(float(item["seconds"] or 0.0))
        if used_batch:
            batch_usage = next((item.get("batch_usage") for item in scored if item.get("batch_usage")), None)
            if batch_usage:
                input_tokens += int(batch_usage.get("prompt_tokens", 0) or 0)
        timing = {
            "total_seconds": total_seconds,
            "prefill_seconds": total_seconds,
            "suffix_seconds": 0.0,
            "encode_seconds": 0.0,
            "image_seconds": 0.0,
            "batch_size": len(results),
            "prefix_tokens": results[0]["input_tokens"] if results else 0,
            "true_suffix_tokens": 0,
            "prefill_passes": 1,
            "suffix_passes": 0,
            "forward_passes": len(results),
            "requests": 1 if used_batch else len(results),
            "concurrency": 1 if used_batch else workers,
            "sum_request_seconds": sum(request_seconds),
            "max_request_seconds": max(request_seconds) if request_seconds else 0.0,
            "batched": True,
            "mode": "batch" if used_batch else "fanout",
            "cached_tokens": int((batch_usage or {}).get("cached_tokens", 0) or 0),
            "image": bool(image),
            "backend": "vllm",
            "input_tokens": input_tokens,
        }
        if batch_usage is not None:
            timing["batch_seconds"] = float(batch_usage.get("seconds", 0.0) or 0.0)
        return results, timing

    # ---- arbitrary messages + label readout (OpenAI-compatible Jev) ----------
    @staticmethod
    def _label_readout(content: dict, labels: list[str]) -> dict:
        """Normalise one first-token logprob entry into label probabilities."""
        mapped = _map_label_logprobs(content.get("top_logprobs", []), labels)
        chosen = str(content.get("token", "")).strip().upper()
        for label in labels:
            if label.strip().upper() == chosen and mapped.get(label) is None:
                mapped[label] = float(content.get("logprob", float("-inf")))
        available = {label: value for label, value in mapped.items() if value is not None}
        if not available:
            raise VLLMError("None of the labels appeared in vLLM top_logprobs")
        maximum = max(available.values())
        weights = {label: pow(2.718281828459045, value - maximum) for label, value in available.items()}
        total = sum(weights.values())
        probabilities = {label: weights[label] / total for label in labels if label in weights}
        return {"probabilities": probabilities, "logprobs": mapped, "raw_token": content.get("token")}

    def score_labels(self, messages: list[dict], labels: list[str], top_logprobs: int = 20):
        """Prefill-only logprob readout over caller-supplied labels for raw messages."""
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a nonempty list")
        if not isinstance(labels, list) or not labels:
            raise ValueError("labels must be a nonempty list")
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": min(max(top_logprobs, len(labels) + 2), 20),
            "chat_template_kwargs": {"enable_thinking": False},
        }
        body, elapsed = self._post("/v1/chat/completions", payload)
        try:
            content = body["choices"][0]["logprobs"]["content"][0]
        except (KeyError, IndexError, TypeError) as error:
            raise VLLMError("vLLM response did not include first-token logprobs") from error
        readout = self._label_readout(content, labels)
        usage = body.get("usage", {}) or {}
        details = usage.get("prompt_tokens_details") or {}
        readout.update(
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            cached_tokens=int(details.get("cached_tokens", 0) or 0),
            seconds=elapsed,
        )
        return readout

    def score_labels_batch(self, conversations: list[list[dict]], labels, top_logprobs: int = 20):
        """Read labels for many conversations in ONE request via vLLM's batch endpoint.

        ``/v1/chat/completions/batch`` (recent vLLM) processes N conversations
        from a single HTTP request and returns one choice per conversation, so the
        whole judgment (person + all tasks) is one upstream HTTP round trip.

        ``labels`` is either one label list shared by every conversation or one
        list per conversation (when criteria declare different option counts).
        Raises :class:`BatchNotSupportedError` when the endpoint is missing, so
        callers can fall back to concurrent per-conversation requests.
        """
        if not isinstance(conversations, list) or not conversations:
            raise ValueError("conversations must be a nonempty list")
        if not isinstance(labels, list) or not labels:
            raise ValueError("labels must be a nonempty list")
        per_conversation = (
            [list(item) for item in labels]
            if isinstance(labels[0], list)
            else [list(labels) for _ in conversations]
        )
        if len(per_conversation) != len(conversations):
            raise ValueError("labels must provide one list per conversation")
        payload = {
            "model": self.model,
            "messages": conversations,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": min(max(top_logprobs, max(len(item) for item in per_conversation) + 2), 20),
            "chat_template_kwargs": {"enable_thinking": False},
        }
        body, elapsed = self._post("/v1/chat/completions/batch", payload, unsupported=(404, 405))
        logger.debug("batched label readout: %d conversations -> %.4fs", len(conversations), elapsed)
        choices = body.get("choices")
        if not isinstance(choices, list) or len(choices) != len(conversations):
            raise VLLMError("vLLM batch response did not include one choice per conversation")
        usage = body.get("usage", {}) or {}
        details = usage.get("prompt_tokens_details") or {}
        batch_usage = {
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "cached_tokens": int(details.get("cached_tokens", 0) or 0),
            "seconds": elapsed,
        }
        results = []
        for index, choice in enumerate(sorted(choices, key=lambda item: item.get("index", 0))):
            try:
                content = choice["logprobs"]["content"][0]
            except (KeyError, IndexError, TypeError) as error:
                raise VLLMError("vLLM batch response did not include first-token logprobs") from error
            readout = self._label_readout(content, per_conversation[index])
            readout.update(
                prompt_tokens=0,
                cached_tokens=0,
                seconds=elapsed / len(conversations),
                batch_usage=batch_usage,
            )
            results.append(readout)
        return results

    # ---- generation --------------------------------------------------------
    def generate(self, state, image, criteria, max_new_tokens=None):
        if not isinstance(criteria, list) or not criteria:
            raise ValueError("criteria must be a nonempty list")
        text = build_generation_text(state, criteria)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": GENERATION_SYSTEM},
                {"role": "user", "content": self._content(text, image)},
            ],
            "max_tokens": max_new_tokens or (16 * len(criteria) + 32),
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        body, elapsed = self._post("/v1/chat/completions", payload)
        choice = body["choices"][0]
        generated = choice["message"].get("content", "") or ""
        usage = body.get("usage", {}) or {}
        new_tokens = int(usage.get("completion_tokens", 0) or 0)
        parsed, answers = parse_output(generated, criteria)
        return {
            "text": generated,
            "output": parsed,
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
            "backend": "vllm",
        }
