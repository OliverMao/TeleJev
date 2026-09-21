# TeleJev (formerly OpenJev)

**Semantic ifs from open models, on a 3090 at home.**

*Independent project; not affiliated with Jev or TypeSafe.*

Most agent decisions are small: *route this*, *retry that*, *does the evidence support X?* A chat model can answer them, but it spends time generating text that software immediately parses back into an `if` statement.

This baseline reads typed option probabilities directly from a model on a single CUDA GPU. No answer sentence, JSON repair, or decoding loop.

## Quick start

Python 3.10+, CUDA, and a GPU that can hold a 4B BF16 model. Install the
runtime dependencies only; the package does not need to be installed:

```bash
python -m venv .venv
. .venv/bin/activate
export HF_HOME=/path/to/large-drive/huggingface
pip install -r requirements.txt
```

Run the scorer directly from the checkout:

```bash
CUDA_VISIBLE_DEVICES=0 python run.py \
  --mode direct \
  --model Qwen/Qwen3.5-4B \
  --input decisions.jsonl \
  --output results.jsonl
```

Each result contains typed option scores, timing, and a prompt hash.

Modes:

- `--mode direct` — one forward pass per row, reading declared option logits.
- `--mode serial` — reuse a shared prefix across rows.
- `--mode shared` — prefill an identical state once, then branch across criteria in parallel.

Only one CUDA GPU may be visible to the process; use `CUDA_VISIBLE_DEVICES` to select it.

## Input

```json
{
  "id": "route-1",
  "state": "Customer cannot access an account after a password reset.",
  "question": "Which queue should handle this request?",
  "options": [
    {"id": "access", "description": "Account access support."},
    {"id": "billing", "description": "Billing support."}
  ]
}
```

Returned probabilities are conditional on the supplied options. Calibrate and validate them on the workload where they will make decisions. `state` may also be a nonempty JSON object or array.

## HTTP 接口

该接口零依赖（仅用 Python 标准库），返回固定格式，无需 JSON 修复。在仓库根目录启动：

```bash
# 真实模型，占用唯一可见的 CUDA GPU
CUDA_VISIBLE_DEVICES=0 python serve.py --model Qwen/Qwen3.5-4B

# 无 GPU 时，使用内置的确定性桩打分器
python serve.py --fake
```

默认监听 `http://127.0.0.1:8000`，可用 `--host` / `--port` 修改。

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/decide` | 原生调用：请求体就是一条决策行。 |
| `POST` | `/v1/chat/completions` | OpenAI 兼容；最后一条 user 消息的 content 放同一份决策行 JSON。 |
| `GET` | `/v1/models` | OpenAI 兼容的模型列表。 |
| `GET` | `/health` | 存活探针。 |

调用原生接口：

```bash
curl -X POST http://127.0.0.1:8000/decide \
  -H "Content-Type: application/json" \
  -d '{
    "id": "route-1",
    "state": "Customer cannot access an account after a password reset.",
    "question": "Which queue should handle this request?",
    "options": [
      {"id": "access", "description": "Account access support."},
      {"id": "billing", "description": "Billing support."}
    ]
  }'
```

用任意 OpenAI 客户端调用 OpenAI 兼容接口：

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "telejev",
    "messages": [{"role": "user", "content": "{\"state\":\"Customer cannot access an account after a password reset.\",\"question\":\"Which queue should handle this request?\",\"options\":[{\"id\":\"access\",\"description\":\"Account access support.\"},{\"id\":\"billing\",\"description\":\"Billing support.\"}]}"}]
  }'
```

两个接口返回同一个决策对象（OpenAI 响应把它放在 `choices[0].message.content`）。key 顺序固定：

```json
{
  "option_id": "access",
  "letter": "A",
  "probabilities": {"access": 0.87, "billing": 0.13},
  "option_logits": {"access": 12.3, "billing": 10.1},
  "input_tokens": 128,
  "prompt_version": "direct-options-v1",
  "probability_status": "conditional option score; uncalibrated as decision confidence"
}
```

该结果由计算得出而非生成，所以结构不会漂移、也不需要 JSON 修复。`usage.completion_tokens` 为 `0`，因为没有做任何解码。

### 在应用中调用

```python
import json
import urllib.request

row = {
    "id": "route-1",
    "state": "Customer cannot access an account after a password reset.",
    "question": "Which queue should handle this request?",
    "options": [
        {"id": "access", "description": "Account access support."},
        {"id": "billing", "description": "Billing support."},
    ],
}
request = urllib.request.Request(
    "http://127.0.0.1:8000/decide",
    data=json.dumps(row).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request) as response:
    decision = json.load(response)
print(decision["option_id"], decision["probabilities"])
```

运行离线接口测试（不需要 GPU 或模型权重）：

```bash
python examples/test_api.py
```

`examples/api_client.py` 是连接真实服务的可运行客户端示例。

## License

Project code is released under the [MIT License](LICENSE). Model weights and third-party source records are not included; upstream models retain their licenses. See [THIRD_PARTY.md](THIRD_PARTY.md).