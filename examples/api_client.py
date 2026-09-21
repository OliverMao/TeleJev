#!/usr/bin/env python3
"""Example client for a running TeleJev server.

Start the server first (real model on one CUDA GPU)::

    python serve.py --model Qwen/Qwen3.5-4B

or without a GPU::

    python serve.py --fake

Then run this client::

    python examples/api_client.py                  # single text decision
    python examples/api_client.py shot.png         # single decision with an image
    python examples/api_client.py shot.png --batch # one image, four criteria
"""

import base64
import json
import sys
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8000"

TASKS = [
    ("fight", "打架", "正在打架"),
    ("fall", "摔倒", "正在摔倒或已经摔倒"),
    ("wave", "挥手", "正在挥手"),
    ("chest", "捂胸口", "正在捂胸口"),
]

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


def data_uri(path: str) -> str:
    payload = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    suffix = (Path(path).suffix.lstrip(".") or "png").lower()
    return f"data:image/{suffix};base64,{payload}"


def batch_body(image: str) -> dict:
    return {
        "state": "监控画面截图。",
        "image": image,
        "criteria": [
            {
                "id": key,
                "question": f"画面中是否有人{verb}？",
                "options": [
                    {"id": "yes", "description": f"画面中有人{verb}。"},
                    {"id": "no", "description": f"画面中没有人{verb}。"},
                ],
            }
            for key, _, verb in TASKS
        ],
    }


def main() -> None:
    image = sys.argv[1] if len(sys.argv) > 1 else None
    if image and "--batch" in sys.argv:
        result = post("/decide-batch", batch_body(data_uri(image)))
        for item in result["results"]:
            print(f"{item['id']}: {item['option_id']} ({item['total_seconds']:.4f}s)")
        print("timing:", json.dumps(result["timing"], ensure_ascii=False))
        return

    row = dict(ROW)
    if image:
        row["image"] = data_uri(image)
    print("decision:", json.dumps(post("/decide", row), ensure_ascii=False))


if __name__ == "__main__":
    main()