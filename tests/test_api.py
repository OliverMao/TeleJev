#!/usr/bin/env python3
"""Offline test for the TeleJev HTTP interface.

Uses the built-in stub scorer, so no GPU, model weights, or network are needed::

    python tests/test_api.py
"""

import base64
import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from telejev.api import (  # noqa: E402
    DecisionService,
    FrameStore,
    fake_batch_scorer,
    fake_generate,
    fake_score_labels,
    fake_score_labels_batch,
    fake_scorer,
    serve,
)

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


def check_backend_batch() -> None:
    """One upstream POST for N conversations, and a clear 404 fallback signal."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from telejev.vllm_backend import BatchNotSupportedError, VLLMBackend

    class Upstream(BaseHTTPRequestHandler):
        batch_ok = True

        def log_message(self, *args) -> None:
            pass

        def do_POST(self) -> None:
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            if self.path == "/v1/chat/completions/batch" and not type(self).batch_ok:
                payload, status = b'{"error": {"message": "Not Found"}}', 404
            else:
                count = 2 if self.path == "/v1/chat/completions/batch" else 1
                payload = json.dumps({
                    "choices": [
                        {"index": index, "logprobs": {"content": [{
                            "token": "A", "logprob": -0.1,
                            "top_logprobs": [{"token": "A", "logprob": -0.1}, {"token": "B", "logprob": -2.3}],
                        }]}}
                        for index in range(count)
                    ],
                    "usage": {"prompt_tokens": 100},
                }).encode()
                status = 200
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    try:
        backend = VLLMBackend(f"http://127.0.0.1:{upstream.server_address[1]}", "telejev-fake")
        readouts = backend.score_labels_batch(
            [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]], ["A", "B"]
        )
        assert backend.requests == 1, backend.requests
        assert len(readouts) == 2 and readouts[0]["probabilities"]["A"] > 0.9, readouts
        assert readouts[0]["batch_usage"]["prompt_tokens"] == 100

        Upstream.batch_ok = False
        try:
            backend.score_labels_batch([[{"role": "user", "content": "a"}]], ["A", "B"])
            raise AssertionError("expected BatchNotSupportedError")
        except BatchNotSupportedError:
            pass

        # /decide-batch style criteria use the same batched endpoint, with a
        # per-criterion fan-out when the server lacks it.
        criteria = [
            {"id": f"c{index}", "question": "q", "options": [
                {"id": "yes", "description": "y"}, {"id": "no", "description": "n"},
            ]}
            for index in range(2)
        ]
        results, timing = backend.score_batch("画面", None, criteria)
        assert timing["mode"] == "fanout" and timing["requests"] == 2, timing
        assert len(results) == 2 and results[0]["probabilities"][0] > 0.9, results
        Upstream.batch_ok = True
        results, timing = backend.score_batch("画面", None, criteria)
        assert timing["mode"] == "batch" and timing["requests"] == 1, timing
        assert len(results) == 2 and results[0]["probabilities"][0] > 0.9, results
    finally:
        upstream.shutdown()


def main() -> None:
    seen_conversations: list = []

    def recording_batch(conversations, labels):
        seen_conversations.append(conversations)
        return fake_score_labels_batch()(conversations, labels)

    service = DecisionService(
        fake_scorer(), "telejev-fake", fake_batch_scorer(), fake_generate(), fake_score_labels(),
        recording_batch, FrameStore(),
    )
    httpd = serve(service, "127.0.0.1", 0)
    port = httpd.server_address[1]
    base = f"http://127.0.0.1:{port}"
    service._frame_base = base  # frames are fetched from this same test server
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
        for key in ("text", "output", "new_tokens", "generate_seconds", "total_seconds", "request_seconds", "forward_passes"):
            assert key in gen, key
        assert set(gen["output"]) == {"has_person", "violations"}
        assert gen["forward_passes"] == 1 + gen["new_tokens"]

        # 5. OpenAI-compatible Jev: one normal chat request -> {has_person, violations}.
        completion = post(base, "/v1/chat/completions", {
            "model": "telejev",
            "messages": [
                {"role": "system", "content": "你是监控助手。"},
                {"role": "user", "content": "任务清单：\n任务名称：摔倒\n任务名称：挥手\n请判定。"},
            ],
        })
        assert completion["object"] == "chat.completion"
        content = json.loads(completion["choices"][0]["message"]["content"])
        assert set(content) == {"has_person", "violations"}
        assert content["has_person"] in (0, 1)
        assert all(name in {"摔倒", "挥手"} for name in content["violations"])
        assert completion["telejev"]["tasks"] == ["摔倒", "挥手"]
        assert completion["telejev"]["requests"] == 3  # person + 2 tasks
        assert completion["telejev"]["http_requests"] == 1  # one batched request for everything
        assert completion["telejev"]["mode"] == "batch"
        assert completion["usage"]["completion_tokens"] == 0

        # 5d. Data-URI images are spilled to /frames/<id>, served back once.
        spill = post(base, "/v1/chat/completions", {
            "model": "telejev",
            "messages": [
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": TINY_PNG}},
                    {"type": "text", "text": "任务名称：摔倒"},
                ]},
            ],
        })
        assert spill["telejev"]["mode"] == "batch"
        parts = seen_conversations[-1][0][-1]["content"]
        frame_url = parts[0]["image_url"]["url"]
        assert frame_url.startswith(base + "/frames/"), frame_url
        with urllib.request.urlopen(frame_url) as response:
            assert response.read() == base64.b64decode(TINY_PNG.split(",", 1)[1])

        # 5e. Servers without the batched endpoint fall back to concurrent reads.
        service._score_labels_batch = None
        fallback = post(base, "/v1/chat/completions", {
            "model": "telejev",
            "messages": [
                {"role": "user", "content": "任务名称：摔倒\n任务名称：挥手"},
            ],
        })
        assert fallback["telejev"]["mode"] == "fanout"
        assert fallback["telejev"]["http_requests"] == 3

        # 5f. A 404 on the batched endpoint disables batch and falls back once.
        class _BatchUnsupported(Exception):
            batch_not_supported = True

        def unsupported_batch(conversations, labels):
            raise _BatchUnsupported("missing endpoint")

        failing_service = DecisionService(
            fake_scorer(), "telejev-fake", fake_batch_scorer(), fake_generate(), fake_score_labels(),
            unsupported_batch,
        )
        fallback_direct = failing_service.openai_chat({
            "messages": [{"role": "user", "content": "任务名称：摔倒"}],
        })
        assert fallback_direct["telejev"]["mode"] == "fanout"
        assert fallback_direct["telejev"]["http_requests"] == 2
        assert failing_service._score_labels_batch is None

        # 5b. OpenAI-compatible model list.
        with urllib.request.urlopen(base + "/v1/models") as response:
            models = json.loads(response.read())
        assert models["data"][0]["id"] == "telejev-fake"

        # 5c. /help documents every endpoint.
        with urllib.request.urlopen(base + "/help") as response:
            help_doc = json.loads(response.read())
        paths = {item["path"] for item in help_doc["endpoints"]}
        assert {"/decide", "/decide-batch", "/generate", "/v1/chat/completions", "/help"} <= paths
        assert "response" in help_doc["decide_batch"] and "output" in help_doc["generate"]["response"]

        # 6. Health probe reports the model name.
        with urllib.request.urlopen(base + "/health") as response:
            health = json.loads(response.read())
        assert health["model"] == "telejev-fake"

        # 7. Unknown paths are rejected.
        try:
            post(base, "/v1/embeddings", {})
            raise AssertionError("expected /v1/embeddings to be missing")
        except urllib.error.HTTPError as error:
            assert error.code == 404, error.code

        print("all interface checks passed\n")
        check_backend_batch()
        print("backend batch checks passed\n")
        print(json.dumps(batch, ensure_ascii=False, indent=2))
    finally:
        httpd.shutdown()


if __name__ == "__main__":
    main()