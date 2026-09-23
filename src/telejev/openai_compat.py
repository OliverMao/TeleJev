"""OpenAI-compatible Jev endpoint.

From the client's point of view this is a normal autoregressive model: send one
chat request (text and/or an image) and read one assistant message. Internally we
never generate the answer token by token -- we read the monitoring tasks with
prefill-only logprob passes: the person read and every task read go out in a
single batched upstream request when the server supports
``/v1/chat/completions/batch`` (falling back to concurrent reads). The fixed
structured answer is then assembled:

    {"has_person": 0 或 1, "violations": ["行为名称", ...]}

Task names come from the request in this order:

1. a top-level ``tasks`` list (strings or ``{"name"|"label"|"taskName": ...}``),
2. ``任务名称：<name>`` lines found in the message text,
3. a built-in default set.

The ``violations`` always use the exact task name so the caller can light up the
matching cards.
"""

from __future__ import annotations

import hashlib
import json
import re
import time

DEFAULT_TASKS = ["打架", "摔倒", "挥手", "捂胸口"]

_TASK_RE = re.compile(r"任务名称[：:]\s*([^\n（(]+)")
_YES, _NO = "A", "B"

PERSON_INSTRUCTION = (
    "只判断【最后一帧】画面中是否有人：有人只输出 A，没有人只输出 B。"
    "不要输出其它任何内容。"
)


def task_instruction(name: str) -> str:
    return (
        f"只判断【最后一帧】是否存在「{name}」这一行为：存在只输出 A，不存在只输出 B。"
        "不要输出其它任何内容。"
    )


def _message_texts(messages: list) -> list[str]:
    texts: list[str] = []
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(str(part.get("text", "")))
    return texts


def extract_tasks(body: dict) -> list[str]:
    """Task names to judge, from the body or parsed out of the prompt."""
    declared = body.get("tasks")
    if isinstance(declared, list) and declared:
        names = []
        for item in declared:
            if isinstance(item, str):
                name = item.strip()
            elif isinstance(item, dict):
                name = str(item.get("name") or item.get("label") or item.get("taskName") or "").strip()
            else:
                name = ""
            if name:
                names.append(name)
        if names:
            return list(dict.fromkeys(names))

    blob = "\n".join(_message_texts(body.get("messages") or []))
    found = [match.group(1).strip() for match in _TASK_RE.finditer(blob)]
    found = [name for name in found if name]
    return list(dict.fromkeys(found)) or list(DEFAULT_TASKS)


def with_instruction(messages: list, instruction: str) -> list:
    """Append an instruction to the last user turn (keeps a single user message).

    Appending to the same turn keeps ``system + images + user text`` as a stable
    prefix so the server's prefix cache is reused across the per-task requests.
    """
    if not messages:
        return [{"role": "user", "content": instruction}]
    out = [dict(message) for message in messages]
    last = out[-1]
    content = last.get("content")
    if isinstance(content, str):
        last["content"] = content + "\n\n" + instruction
    elif isinstance(content, list):
        last["content"] = content + [{"type": "text", "text": "\n\n" + instruction}]
    else:
        out.append({"role": "user", "content": instruction})
    return out


def build_completion(model: str, output: dict, detail: dict) -> dict:
    """Wrap the assembled decision in an OpenAI chat.completion envelope."""
    content = json.dumps(output, ensure_ascii=False)
    prompt_tokens = int(detail.get("prompt_tokens", 0) or 0)
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": 0,
        "total_tokens": prompt_tokens,
        "prompt_tokens_details": {"cached_tokens": int(detail.get("cached_tokens", 0) or 0)},
    }
    return {
        "id": "chatcmpl-" + hashlib.sha256(content.encode()).hexdigest()[:24],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
        "telejev": detail,
    }