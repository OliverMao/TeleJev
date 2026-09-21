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

- `--mode direct` — one forward pass per row, reading declared option logits. Supports optional image input.
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

A row may also carry an optional `image` field (base64 data URI, http(s) URL, or local path) as extra evidence; see [图像输入](#图像输入).

## HTTP 接口

该接口零额外依赖（仅用 Python 标准库），返回固定格式，无需 JSON 修复；图像输入需要 `pillow`（已列入 `requirements.txt`）。在仓库根目录启动：

```bash
# 真实模型，占用唯一可见的 CUDA GPU
CUDA_VISIBLE_DEVICES=0 python serve.py --model Qwen/Qwen3.5-4B

# 无 GPU 时，使用内置的确定性桩打分器
python serve.py --fake
```

默认监听 `http://127.0.0.1:8000`，可用 `--host` / `--port` 修改。

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/decide` | 请求体就是一条决策行（可含可选 `image`），返回固定决策对象。 |
| `GET` | `/health` | 存活探针，返回服务与模型名。 |

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

返回的决策对象 key 顺序固定：

```json
{
  "option_id": "access",
  "letter": "A",
  "probabilities": {"access": 0.87, "billing": 0.13},
  "option_logits": {"access": 12.3, "billing": 10.1},
  "has_image": false,
  "image_tokens": 0,
  "input_tokens": 128,
  "prompt_version": "direct-options-v1",
  "probability_status": "conditional option score; uncalibrated as decision confidence"
}
```

该结果由计算得出而非生成，所以结构不会漂移、也不需要 JSON 修复。

### 图像输入

决策行可带一个可选的 `image` 字段，支持三种形式：

- base64 data URI：`data:image/png;base64,<...>`
- http(s) URL
- 本地文件路径（服务与文件在同一台机器时）

```bash
curl -X POST http://127.0.0.1:8000/decide \
  -H "Content-Type: application/json" \
  -d '{
    "state": "Customer cannot access an account after a password reset.",
    "question": "Which queue should handle this request?",
    "options": [
      {"id": "access", "description": "Account access support."},
      {"id": "billing", "description": "Billing support."}
    ],
    "image": "/path/to/screenshot.png"
  }'
```

图像只有在 `--model` 暴露多模态 processor 时才会真正送入模型（启动时会打印 `multimodal=True/False`）；纯文本模型会直接报错。`--mode direct` 支持图像，`serial` / `shared` 不支持。返回的 `has_image` 标明本次是否使用了图像，`image_tokens` 是实际喂给视觉塔的图像 token 数（为 0 说明视觉没有生效）。

### 前端测试页

`examples/index.html` 是单文件测试页：可编辑 state / question / options，上传图像，直接调用 `/decide` 并用条形图展示概率。

```bash
python serve.py --fake          # 启动服务
# 浏览器打开 examples/index.html
```

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