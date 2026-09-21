#!/usr/bin/env python3
"""Example client for a running TeleJev server.

Start the server first (real model on one CUDA GPU)::

    python serve.py --model Qwen/Qwen3.5-4B

or without a GPU::

    python serve.py --fake

Then run this client::

    python examples/api_client.py
"""

import json
import urllib.request

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


def main() -> None:
    # A. Native call: send the decision row directly.
    decision = post("/decide", ROW)
    print("decision:", json.dumps(decision, ensure_ascii=False))

    # B. OpenAI-compatible call: same payload as the last user message.
    payload = {key: ROW[key] for key in ("state", "question", "options")}
    completion = post(
        "/v1/chat/completions",
        {"model": "telejev", "messages": [{"role": "user", "content": json.dumps(payload)}]},
    )
    content = completion["choices"][0]["message"]["content"]
    print("openai content:", content)


if __name__ == "__main__":
    main()