"""Native HTTP interface for TeleJev decisions.

The output format is fixed by construction: the server computes the decision and
serialises it with a frozen key order, so callers never need JSON repair.

Run with a real model (single visible CUDA GPU)::

    python serve.py --model Qwen/Qwen3.5-4B

Run the interface without a GPU (deterministic stub scorer)::

    python serve.py --fake

Endpoints
---------
GET  /health        liveness probe
POST /decide        one decision row -> fixed decision object
POST /decide-batch  one state/image + many criteria, sharing one image prefill

``image`` is optional and may be a base64 ``data:`` URI, an ``http(s)`` URL, or a
local file path. It is only used when the loaded model exposes a multimodal
processor.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .core import LETTERS, validate_row

ROW_KEYS = ("id", "state", "question", "options", "image")


def canonical_decision(raw: dict) -> dict:
    """Freeze a scorer result into the stable public decision object."""
    option_ids = list(raw["option_ids"])
    probabilities = list(raw["probabilities"])
    logits = list(raw["option_logits"])
    if not (len(option_ids) == len(probabilities) == len(logits)):
        raise ValueError("Scorer returned misaligned option arrays")
    if not option_ids:
        raise ValueError("Scorer returned no options")
    best = max(range(len(probabilities)), key=probabilities.__getitem__)
    decision = {
        "option_id": option_ids[best],
        "letter": LETTERS[best],
        "probabilities": {oid: float(p) for oid, p in zip(option_ids, probabilities)},
        "option_logits": {oid: float(v) for oid, v in zip(option_ids, logits)},
        "has_image": bool(raw.get("has_image", False)),
        "image_tokens": int(raw.get("image_tokens", 0)),
        "input_tokens": int(raw.get("input_tokens", 0)),
        "prompt_version": raw.get("prompt_version"),
        "probability_status": raw.get("probability_status"),
    }
    for key in ("prefill_seconds", "suffix_seconds", "forward_seconds", "total_seconds"):
        if key in raw:
            decision[key] = float(raw[key])
    return decision


class DecisionService:
    """Thread-safe wrapper around single-row and batch scorers."""

    def __init__(self, score_fn, model_name: str, batch_fn=None):
        self._score = score_fn
        self._batch = batch_fn
        self.model_name = model_name
        self._lock = threading.Lock()

    def decide(self, body: dict) -> dict:
        row = {key: body[key] for key in ROW_KEYS if key in body}
        row.setdefault("id", "request")
        validate_row(row)
        with self._lock:
            raw = self._score(row)
        return canonical_decision(raw)

    def decide_batch(self, body: dict) -> dict:
        if self._batch is None:
            raise ValueError("Batch scoring is not available on this server")
        started = time.perf_counter()
        state = body.get("state")
        image = body.get("image")
        criteria = body.get("criteria")
        with self._lock:
            results, timing = self._batch(state, image, criteria)
        timing["total_seconds"] = time.perf_counter() - started
        return {"results": [canonical_decision(result) for result in results], "timing": timing}


def load_model_scorers(model: str, max_tokens: int = 4096):
    """Load one CUDA model and return (single scorer, batch scorer, metadata)."""
    from .batch import score_batch
    from .core import load_causal_model
    from .direct import score

    loaded_model, tokenizer, processor, metadata = load_causal_model(model)

    def score_fn(row: dict) -> dict:
        return score(loaded_model, tokenizer, row, metadata, max_tokens, processor)

    def batch_fn(state: str, image, criteria):
        return score_batch(
            loaded_model, tokenizer, metadata, state, image, criteria, max_tokens, processor
        )

    return score_fn, batch_fn, metadata


def fake_scorer():
    """Deterministic scorer for interface tests; requires no model or GPU."""

    def score_fn(row: dict) -> dict:
        import math

        logits = [float(len(option["description"]) % 7 + index) for index, option in enumerate(row["options"])]
        maximum = max(logits)
        weights = [math.exp(value - maximum) for value in logits]
        total = sum(weights)
        return {
            "id": row["id"],
            "option_ids": [option["id"] for option in row["options"]],
            "probabilities": [weight / total for weight in weights],
            "option_logits": logits,
            "has_image": bool(row.get("image")),
            "image_tokens": 0,
            "input_tokens": 42,
            "prompt_version": "fake-v1",
            "probability_status": "fake scorer for interface tests",
        }

    return score_fn


def fake_batch_scorer():
    """Deterministic batch scorer mirroring the fake single scorer shape."""

    def batch_fn(state: str, image, criteria):
        import math

        if not isinstance(criteria, list) or not criteria:
            raise ValueError("criteria must be a nonempty list")
        started = time.perf_counter()
        results = []
        for index, criterion in enumerate(criteria):
            options = criterion["options"]
            logits = [float(len(option["description"]) % 7 + position) for position, option in enumerate(options)]
            maximum = max(logits)
            weights = [math.exp(value - maximum) for value in logits]
            total = sum(weights)
            results.append(
                {
                    "id": criterion.get("id") or f"criterion-{index}",
                    "option_ids": [option["id"] for option in options],
                    "probabilities": [weight / total for weight in weights],
                    "option_logits": logits,
                    "has_image": bool(image),
                    "image_tokens": 0,
                    "input_tokens": 42,
                    "prompt_version": "fake-v1",
                    "probability_status": "fake scorer for interface tests",
                }
            )
        timing = {
            "inference_seconds": time.perf_counter() - started,
            "image_seconds": 0.0002,
            "encode_seconds": 0.0005,
            "prefill_seconds": 0.002,
            "suffix_seconds": 0.001 * len(results),
            "image": bool(image),
            "batch_size": len(results),
            "prefix_tokens": 32,
            "true_suffix_tokens": 16 * len(results),
        }
        return results, timing

    return batch_fn


class _Handler(BaseHTTPRequestHandler):
    service: DecisionService = None  # type: ignore[assignment]

    def log_message(self, *args) -> None:  # silence default stderr logging
        pass

    def _send(self, status: int, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send(200, {"status": "ok", "model": self.service.model_name})
        else:
            self._send(404, {"error": {"message": f"unknown path {self.path}", "type": "invalid_request_error"}})

    def do_POST(self) -> None:
        if self.path not in {"/decide", "/decide-batch"}:
            self._send(404, {"error": {"message": f"unknown path {self.path}", "type": "invalid_request_error"}})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/decide":
                self._send(200, self.service.decide(body))
            else:
                self._send(200, self.service.decide_batch(body))
        except Exception as error:  # surface a stable error envelope
            self._send(400, {"error": {"message": str(error), "type": "invalid_request_error"}})


def serve(service: DecisionService, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    """Build (but do not start) an HTTP server bound to the given service."""
    handler = type("BoundHandler", (_Handler,), {"service": service})
    return ThreadingHTTPServer((host, port), handler)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B", help="Model source or local path")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--fake", action="store_true", help="Serve a stub scorer; no GPU or model required")
    args = parser.parse_args()

    if args.fake:
        service = DecisionService(fake_scorer(), "telejev-fake", fake_batch_scorer())
    else:
        score_fn, batch_fn, metadata = load_model_scorers(args.model, args.max_tokens)
        print(f"Loaded model: multimodal={metadata['multimodal']}", flush=True)
        service = DecisionService(score_fn, args.model, batch_fn)

    httpd = serve(service, args.host, args.port)
    print(f"Serving '{service.model_name}' on http://{args.host}:{args.port}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()