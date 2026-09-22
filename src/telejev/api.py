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
GET  /v1/models     OpenAI-compatible model list
POST /decide        one decision row -> fixed decision object
POST /decide-batch  one state/image + many criteria, sharing one image prefill
POST /generate      same criteria via full greedy autoregressive generation
POST /v1/chat/completions  OpenAI-compatible Jev readout over client labels

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
from .prompt import task_standard

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

    def __init__(self, score_fn, model_name: str, batch_fn=None, generate_fn=None, score_labels_fn=None):
        self._score = score_fn
        self._batch = batch_fn
        self._generate = generate_fn
        self._score_labels = score_labels_fn
        self.model_name = model_name
        self._lock = threading.Lock()

    def decide(self, body: dict) -> dict:
        row = {key: body[key] for key in ROW_KEYS if key in body}
        row.setdefault("id", "request")
        if body.get("label") or body.get("description_opt") or body.get("description"):
            row["standard"] = task_standard(body)
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

    def generate(self, body: dict) -> dict:
        if self._generate is None:
            raise ValueError("Generation is not available on this server")
        started = time.perf_counter()
        state = body.get("state")
        image = body.get("image")
        criteria = body.get("criteria")
        max_new_tokens = body.get("max_new_tokens")
        with self._lock:
            result = self._generate(state, image, criteria, max_new_tokens)
        result["request_seconds"] = time.perf_counter() - started
        return result

    def openai_chat(self, body: dict) -> dict:
        """OpenAI-compatible Jev: one normal chat request -> {has_person, violations}."""
        if self._score_labels is None:
            raise ValueError("The OpenAI-compatible Jev endpoint requires --backend sglang or vllm")
        from concurrent.futures import ThreadPoolExecutor

        from .openai_compat import (
            PERSON_INSTRUCTION, build_completion, extract_tasks, task_instruction, with_instruction,
        )

        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a nonempty list")
        tasks = extract_tasks(body)

        def judge(instruction: str) -> dict:
            return self._score_labels(with_instruction(messages, instruction), ["A", "B"])

        def hit(result: dict) -> bool:
            probabilities = result.get("probabilities", {})
            return probabilities.get("A", 0.0) >= probabilities.get("B", 0.0)

        started = time.perf_counter()
        # Warm the shared prefix (system + images + user text) once, then fan the
        # task reads out concurrently so the server batches them (and reuses the
        # prefix cache instead of re-encoding every image per task).
        outcomes = [judge(PERSON_INSTRUCTION)]
        task_instructions = [task_instruction(name) for name in tasks]
        workers = min(len(task_instructions), 32) if task_instructions else 1
        if task_instructions:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                outcomes += list(pool.map(judge, task_instructions))
        total_seconds = time.perf_counter() - started

        output = {
            "has_person": 1 if hit(outcomes[0]) else 0,
            "violations": [name for name, result in zip(tasks, outcomes[1:]) if hit(result)],
        }
        detail = {
            "tasks": tasks,
            "has_person_probabilities": outcomes[0].get("probabilities", {}),
            "violation_probabilities": {
                name: outcomes[index + 1].get("probabilities", {}) for index, name in enumerate(tasks)
            },
            "requests": len(outcomes),
            "concurrency": workers,
            "prefix_warmup": True,
            "prompt_tokens": sum(int(result.get("prompt_tokens", 0) or 0) for result in outcomes),
            "cached_tokens": sum(int(result.get("cached_tokens", 0) or 0) for result in outcomes),
            "requests_detail": [
                {
                    "prompt_tokens": int(result.get("prompt_tokens", 0) or 0),
                    "cached_tokens": int(result.get("cached_tokens", 0) or 0),
                    "seconds": float(result.get("seconds", 0.0) or 0.0),
                }
                for result in outcomes
            ],
            "total_seconds": total_seconds,
            "sum_request_seconds": sum(float(result.get("seconds", 0.0) or 0.0) for result in outcomes),
        }
        return build_completion(self.model_name, output, detail)


def load_model_scorers(model: str, max_tokens: int = 4096):
    """Load one CUDA model and return (single scorer, batch scorer, generate scorer, metadata)."""
    from .autoregressive import generate_answers
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

    def generate_fn(state: str, image, criteria, max_new_tokens):
        return generate_answers(
            loaded_model, tokenizer, processor, state, criteria, image, max_new_tokens
        )

    return score_fn, batch_fn, generate_fn, metadata


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
            "prefill_passes": 1,
            "suffix_passes": 1,
            "forward_passes": 2,
        }
        return results, timing

    return batch_fn


def fake_generate():
    """Deterministic generation stub for interface tests."""

    def generate_fn(state: str, image, criteria, max_new_tokens):
        started = time.perf_counter()
        if not isinstance(criteria, list) or not criteria:
            raise ValueError("criteria must be a nonempty list")
        answers = []
        output = {"has_person": 0, "violations": []}
        for criterion in criteria:
            logits = [len(option["description"]) % 7 + index for index, option in enumerate(criterion["options"])]
            best = max(range(len(logits)), key=logits.__getitem__)
            answer = criterion["options"][best]["id"]
            answers.append(answer)
            if criterion.get("id") == "person":
                output["has_person"] = 1 if answer == "yes" else 0
            elif answer == "yes":
                output["violations"].append(criterion.get("label") or criterion.get("id"))
        text = json.dumps(output, ensure_ascii=False)
        generate_seconds = 0.01 * len(criteria)
        new_tokens = 2 * len(criteria)
        return {
            "text": text,
            "output": output,
            "answers": answers,
            "prompt_tokens": 42,
            "new_tokens": new_tokens,
            "encode_seconds": 0.0005,
            "generate_seconds": generate_seconds,
            "total_seconds": time.perf_counter() - started,
            "tokens_per_second": new_tokens / generate_seconds if generate_seconds else None,
            "prefill_passes": 1,
            "decode_passes": new_tokens,
            "forward_passes": 1 + new_tokens,
            "has_image": bool(image),
        }

    return generate_fn


def fake_score_labels():
    """Deterministic label readout stub for interface tests."""

    def score_labels_fn(messages: list, labels: list) -> dict:
        import hashlib
        import math

        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a nonempty list")
        if not isinstance(labels, list) or not labels:
            raise ValueError("labels must be a nonempty list")
        seed = int(hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest(), 16)
        weights = {label: math.exp(((seed + index) % 7) / 5.0) for index, label in enumerate(labels)}
        total = sum(weights.values())
        probabilities = {label: weight / total for label, weight in weights.items()}
        return {
            "probabilities": probabilities,
            "logprobs": {label: math.log(value) for label, value in probabilities.items()},
            "prompt_tokens": 42,
            "cached_tokens": 32 if len(messages) > 1 else 0,
            "seconds": 0.002,
            "raw_token": labels[0],
        }

    return score_labels_fn


def help_document(model_name: str) -> dict:
    """Machine-readable usage guide served by GET /help."""
    return {
        "service": "telejev",
        "model": model_name,
        "summary": "Runtime-defined semantic decisions: Jev-style direct logit readout and full autoregressive generation.",
        "endpoints": [
            {"method": "GET", "path": "/help", "description": "本说明。"},
            {"method": "GET", "path": "/health", "description": "存活探针，返回服务与模型名。"},
            {"method": "GET", "path": "/v1/models", "description": "OpenAI 兼容的模型列表。"},
            {"method": "POST", "path": "/decide", "description": "单条决策行 -> 固定决策对象。"},
            {"method": "POST", "path": "/decide-batch", "description": "一份 state/image + 多个 criteria，图像只 prefill 一次。"},
            {"method": "POST", "path": "/generate", "description": "同一组 criteria 走完整自回归，直接生成 {has_person, violations}。"},
            {"method": "POST", "path": "/v1/chat/completions", "description": "OpenAI 兼容：像普通模型一样发一次请求，返回 {has_person, violations}。"},
        ],
        "decide_batch": {
            "request": {
                "state": "监控画面截图。",
                "image": "https://ossv2.yoobit.cn/nife/fall.png 或 data:image/...;base64,... 或本地路径",
                "criteria": [
                    {
                        "id": "fall",
                        "label": "摔倒",
                        "question": "画面中是否有人正在摔倒？",
                        "options": [
                            {"id": "yes", "description": "画面中有人正在摔倒。"},
                            {"id": "no", "description": "画面中没有人正在摔倒。"},
                        ],
                    }
                ],
            },
            "response": {
                "results": [
                    {
                        "option_id": "no",
                        "letter": "B",
                        "probabilities": {"yes": 0.12, "no": 0.88},
                        "option_logits": {"yes": 3.1, "no": 5.0},
                        "has_image": True,
                        "image_tokens": 1024,
                        "input_tokens": 312,
                    }
                ],
                "timing": {
                    "total_seconds": 0.09,
                    "requests": 1,
                    "concurrency": 1,
                    "forward_passes": 1,
                },
            },
            "curl": "curl -X POST $BASE/decide-batch -H 'Content-Type: application/json' -d @batch.json",
        },
        "generate": {
            "request": {"state": "监控画面截图。", "image": "data:image/...;base64,...", "criteria": "同 /decide-batch"},
            "response": {
                "text": "{\"has_person\": 0, \"violations\": [\"摔倒\"]}",
                "output": {"has_person": 0, "violations": ["摔倒"]},
                "new_tokens": 12,
                "generate_seconds": 1.23,
                "total_seconds": 1.24,
            },
            "curl": "curl -X POST $BASE/generate -H 'Content-Type: application/json' -d '{...}'  # 与 /decide-batch 同体，无 options 也可",
        },
        "openai_chat": {
            "request": {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "你是实时视频流助手，逐帧观察摄像头画面并判定监测行为。"},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
                        {"type": "text", "text": "任务名称：摔倒\n任务名称：挥手\n请只判定这些行为。"},
                    ]},
                ],
                "tasks": ["摔倒", "挥手"],
            },
            "internal": "对每个任务（含“是否有人”）并发发一次 prefill-only logprob 读取给 SGLang/vLLM，由服务端批处理，再拼装结果；用户无需关心。",
            "response": {
                "object": "chat.completion",
                "choices": [{"message": {"role": "assistant", "content": "{\"has_person\": 1, \"violations\": [\"挥手\"]}"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 300, "completion_tokens": 0, "total_tokens": 300, "prompt_tokens_details": {"cached_tokens": 300}},
                "telejev": {
                    "tasks": ["摔倒", "挥手"],
                    "has_person_probabilities": {"A": 0.9, "B": 0.1},
                    "violation_probabilities": {"摔倒": {"A": 0.2, "B": 0.8}, "挥手": {"A": 0.7, "B": 0.3}},
                    "requests": 3,
                    "concurrency": 3,
                    "cached_tokens": 300,
                },
            },
            "task_source": "任务名优先取顶层 tasks，其次从 messages 里的“任务名称：X”解析，都没有则用内置默认集合。",
            "prefix_cache": "system + 图像 + 任务清单在前，逐任务指令拼在最后；服务端 APC/Radix Cache 只编码一次共享前缀。SGLang 默认开，vLLM 需 --enable-prefix-caching（多图还需 --limit-mm-per-prompt image=20）；内部先用“是否有人”预热后缀，再并发其余任务。追加式帧历史（旧帧不变、新帧后）前缀稳定，命中最佳。响应 usage.prompt_tokens_details.cached_tokens / telejev.cached_tokens 可用于验证是否命中。",
        },
        "notes": [
            "除 /v1/chat/completions 外，其余 POST 端点需要 body 中包含 state 与 criteria（或单条决策行）。",
            "image 支持 data URI、http(s) URL、本地路径；仅当模型暴露多模态 processor 时会真正送入。",
            "Jev 式（/decide, /decide-batch, /v1/chat/completions）不生成 token；/generate 为完整自回归基线。",
            "所有响应为 UTF-8 JSON；输出格式固定，无需 JSON 修复。",
            "当前后端：" + model_name,
        ],
    }


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
        elif self.path == "/help":
            self._send(200, help_document(self.service.model_name))
        elif self.path == "/v1/models":
            self._send(200, {
                "object": "list",
                "data": [{"id": self.service.model_name, "object": "model", "owned_by": "telejev"}],
            })
        else:
            self._send(404, {"error": {"message": f"unknown path {self.path}", "type": "invalid_request_error"}})

    def do_POST(self) -> None:
        if self.path not in {"/decide", "/decide-batch", "/generate", "/v1/chat/completions"}:
            self._send(404, {"error": {"message": f"unknown path {self.path}", "type": "invalid_request_error"}})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/decide":
                self._send(200, self.service.decide(body))
            elif self.path == "/decide-batch":
                self._send(200, self.service.decide_batch(body))
            elif self.path == "/generate":
                self._send(200, self.service.generate(body))
            else:
                self._send(200, self.service.openai_chat(body))
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
    parser.add_argument("--backend", choices=("local", "sglang", "vllm"), default="local",
                        help="local = load the model in-process; sglang/vllm = use a running OpenAI-compatible server")
    parser.add_argument("--server-url", dest="server_url", default=None,
                        help="OpenAI-compatible server base URL (default: http://127.0.0.1:30000)")
    parser.add_argument("--sglang-url", dest="server_url", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--served-model", dest="served_model", default=None,
                        help="Model name served by the server (default: --model)")
    parser.add_argument("--sglang-model", dest="served_model", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--fake", action="store_true", help="Serve a stub scorer; no GPU or model required")
    args = parser.parse_args()

    if args.fake:
        service = DecisionService(
            fake_scorer(), "telejev-fake", fake_batch_scorer(), fake_generate(), fake_score_labels()
        )
    elif args.backend in ("sglang", "vllm"):
        from .sglang_backend import SGLangBackend

        url = args.server_url or "http://127.0.0.1:30000"
        backend = SGLangBackend(url, args.served_model or args.model, backend_name=args.backend)
        print(f"{args.backend} backend: {url} model={backend.model}", flush=True)
        service = DecisionService(
            backend.score_row, backend.model, backend.score_batch, backend.generate, backend.score_labels
        )
    else:
        score_fn, batch_fn, generate_fn, metadata = load_model_scorers(args.model, args.max_tokens)
        print(f"Loaded model: multimodal={metadata['multimodal']}", flush=True)
        service = DecisionService(score_fn, args.model, batch_fn, generate_fn)

    httpd = serve(service, args.host, args.port)
    print(f"Serving '{service.model_name}' on http://{args.host}:{args.port}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()