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

from telejev.api import DecisionService, fake_scorer, serve  # noqa: E402

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
    httpd = serve(DecisionService(fake_scorer(), "telejev-fake"), "127.0.0.1", 0)
    port = httpd.server_address[1]
    base = f"http://127.0.0.1:{port}"
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        # 1. Text-only decision returns the frozen decision object.
        decision = post(base, "/decide", ROW)
        assert set(decision) == EXPECTED_KEYS, decision.keys()
        assert decision["option_id"] in {option["id"] for option in ROW["options"]}
        assert abs(sum(decision["probabilities"].values()) - 1.0) < 1e-9
        assert decision["has_image"] is False

        # 2. Image input is accepted and flagged.
        with_image = post(base, "/decide", {**ROW, "image": TINY_PNG})
        assert with_image["has_image"] is True
        assert with_image["option_id"] in {option["id"] for option in ROW["options"]}

        # 3. Health probe reports the model name.
        with urllib.request.urlopen(base + "/health") as response:
            health = json.loads(response.read())
        assert health["model"] == "telejev-fake"

        # 4. The OpenAI-compatible surface has been removed.
        try:
            post(base, "/v1/chat/completions", {"model": "telejev", "messages": []})
            raise AssertionError("expected /v1/chat/completions to be gone")
        except urllib.error.HTTPError as error:
            assert error.code == 404, error.code

        print("all interface checks passed\n")
        print(json.dumps(with_image, ensure_ascii=False, indent=2))
    finally:
        httpd.shutdown()


if __name__ == "__main__":
    main()