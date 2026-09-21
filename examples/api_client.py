#!/usr/bin/env python3
"""Example client for a running TeleJev server.

Start the server first (real model on one CUDA GPU)::

    python serve.py --model Qwen/Qwen3.5-4B

or without a GPU::

    python serve.py --fake

Then run this client::

    python examples/api_client.py                  # text-only decision
    python examples/api_client.py path/to/shot.png # decision with an image
"""

import base64
import json
import sys
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8000"

ROW = {
    "id": "route-1",
    "state": "Customer cannot access an account after a password reset.",
    "question": "Which queue should handle this request?",
    "options": [
        {"id": "access", "description": "Account access support."},
        {"id": "billing", "description": "Billing support."},
    ],
}


def post(path: str, body: dict) -> dict:
    request = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read())


def with_image(row: dict, path: str) -> dict:
    payload = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    suffix = (Path(path).suffix.lstrip(".") or "png").lower()
    return {**row, "image": f"data:image/{suffix};base64,{payload}"}


def main() -> None:
    print("decision:", json.dumps(post("/decide", ROW), ensure_ascii=False))
    if len(sys.argv) > 1:
        result = post("/decide", with_image(ROW, sys.argv[1]))
        print("decision+image:", json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()