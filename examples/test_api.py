#!/usr/bin/env python3
"""Offline test for the TeleJev HTTP interface.

Uses the built-in stub scorer, so no GPU, model weights, or network are needed::

    python examples/test_api.py
"""

import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from telejev.api import DecisionService, fake_batch_scorer, fake_generate, fake_scorer, serve  # noqa: E402

# 1x1 transparent PNG, used to exercise the optional image field.
TINY_PNG = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M8AAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)

ROW = {
    "id": "route-1",
    "state": "Customer cannot access an account after a password reset.",
    "question": "Which queue should handle this request?",
    "options": [
        {"id": "access", "description": "Account access support."},
        {"id": "billing", "description": "Billing support."},
    ],
}

BATCH = {
    "state": "监控画面截图。",
    "image": TINY_PNG,
    "criteria": [
        {
            "id": "fight",
            "question": "画面中是否有人正在打架？",
            "options": [
                {"id": "yes", "description": "画面中有人正在打架。"},
                {"id": "no", "description": "画面中没有人正在打架。"},
            ],
        },
        {
            "id": "fall",
            "question": "画面中是否有人正在摔倒？",
            "options": [
                {"id": "yes", "description": "画面中有人正在摔倒。"},
                {"id": "no", "description": "画面中没有人正在摔倒。"},
            ],
        },
    ],
}

EXPECTED_KEYS = {
    "option_id",
    "letter",
    "probabilities",
    "option_logits",
    "has_image",
    "image_tokens",
    "input_tokens",
    "prompt_version",
    "probability_status",
}


def post(base: str, path: str, body: dict) -> dict:
    request = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        assert response.status == 200, response.status
        return json.loads(response.read())


def main() -> None:
    service = DecisionService(fake_scorer(), "telejev-fake", fake_batch_scorer(), fake_generate())
    httpd = serve(service, "127.0.0.1", 0)
    port = httpd.server_address[1]
    base = f"http://127.0.0.1:{port}"
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        # 1. Single text-only decision returns the frozen decision object.
        decision = post(base, "/decide", ROW)
        assert set(decision) == EXPECTED_KEYS, decision.keys()
        assert abs(sum(decision["probabilities"].values()) - 1.0) < 1e-9
        assert decision["has_image"] is False

        # 2. Single image decision is flagged.
        with_image = post(base, "/decide", {**ROW, "image": TINY_PNG})
        assert with_image["has_image"] is True

        # 3. Batch decision returns one result per criterion plus timings.
        batch = post(base, "/decide-batch", BATCH)
        assert len(batch["results"]) == len(BATCH["criteria"])
        for result in batch["results"]:
            assert EXPECTED_KEYS <= set(result), result.keys()
        assert batch["results"][0]["has_image"] is True
        timing = batch["timing"]
        for key in ("total_seconds", "inference_seconds", "image_seconds", "prefill_seconds", "suffix_seconds", "batch_size", "forward_passes"):
            assert key in timing, key
        assert timing["batch_size"] == len(BATCH["criteria"])
        assert timing["forward_passes"] == 2

        # 4. Full autoregressive generation returns answers and timing.
        gen = post(base, "/generate", {"state": BATCH["state"], "image": TINY_PNG, "criteria": BATCH["criteria"]})
        assert len(gen["answers"]) == len(BATCH["criteria"])
        for key in ("text", "new_tokens", "generate_seconds", "total_seconds", "request_seconds", "forward_passes"):
            assert key in gen, key
        assert gen["forward_passes"] == 1 + gen["new_tokens"]

        # 5. Health probe reports the model name.
        with urllib.request.urlopen(base + "/health") as response:
            health = json.loads(response.read())
        assert health["model"] == "telejev-fake"

        # 6. Unknown paths are rejected.
        try:
            post(base, "/v1/chat/completions", {})
            raise AssertionError("expected /v1/chat/completions to be gone")
        except urllib.error.HTTPError as error:
            assert error.code == 404, error.code

        print("all interface checks passed\n")
        print(json.dumps(batch, ensure_ascii=False, indent=2))
    finally:
        httpd.shutdown()


if __name__ == "__main__":
    main()