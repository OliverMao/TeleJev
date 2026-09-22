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

### SGLang / vLLM 后端

把推理放到运行中的 OpenAI 兼容服务上（SGLang 或 vLLM），Jev 式读 logits 与生成都走该服务，享受融合 kernel / CUDA Graph / 前缀缓存：

```bash
# 1) 起服务（二选一）
python -m sglang.launch_server --model-path /path/to/model --port 30000
# 或
vllm serve /path/to/model --port 30000 --enable-prefix-caching

# 2) 用该后端提供 TeleJev 接口
python serve.py --backend sglang \     # 或 --backend vllm
  --server-url http://127.0.0.1:30000 \
  --served-model Qwen/Qwen3.5-4B --port 22001
```

- **Jev 式读取**：`max_tokens=1` + `logprobs`，只对选项字母归一化（prefill-only，不生成）
- **生成**：同一服务的普通 `max_tokens` 路径
- **多判据**：并发发出各判据请求，由服务端**连续批处理**在同一 step 内完成，并用前缀缓存复用图像/state 共享前缀（SGLang Radix Cache 默认开；vLLM 需 `--enable-prefix-caching`）

直接用 vLLM：`python serve.py --backend vllm --server-url http://127.0.0.1:30000 --served-model <model>`。

直接对比两者（对 SGLang / vLLM 都适用）：

```bash
python examples/sglang_compare.py \
  --base-url http://127.0.0.1:30000 --model Qwen/Qwen3.5-4B --image examples/fall.png
```

默认监听 `http://127.0.0.1:8000`，可用 `--host` / `--port` 修改。

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/decide` | 请求体就是一条决策行（可含可选 `image`），返回固定决策对象。 |
| `POST` | `/decide-batch` | 一份 state/image + 多个 criteria，共享一次图像 prefill；返回每个任务结果与耗时。 |
| `POST` | `/generate` | 同一组 criteria 走完整自回归生成（`model.generate`），返回文本、解析结果与耗时。 |
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

图像只有在 `--model` 暴露多模态 processor 时才会真正送入模型（启动时会打印 `multimodal=True/False`）；纯文本模型会直接报错。`--mode direct` 支持图像，`shared` 不支持。返回的 `has_image` 标明本次是否使用了图像，`image_tokens` 是实际喂给视觉塔的图像 token 数（为 0 说明视觉没有生效）。

### 批量判定（多任务）

同一张画面判定多个行为时，用 `/decide-batch`：图像只 prefill 一次，避免每个任务重复编码。

```bash
curl -X POST http://127.0.0.1:8000/decide-batch \
  -H "Content-Type: application/json" \
  -d '{
    "state": "监控画面截图。",
    "image": "https://ossv2.yoobit.cn/nife/fall.png",
    "criteria": [
      {"id": "fight", "question": "画面中是否有人正在打架？", "options": [{"id": "yes", "description": "画面中有人正在打架。"}, {"id": "no", "description": "画面中没有人正在打架。"}]},
      {"id": "fall", "question": "画面中是否有人正在摔倒？", "options": [{"id": "yes", "description": "画面中有人正在摔倒。"}, {"id": "no", "description": "画面中没有人正在摔倒。"}]}
    ]
  }'
```

图像 prefill 1 次 + 判据后缀批量 1 次，共 **2 次前向**，与判据数量无关。

响应：

```json
{
  "results": [
    {
      "option_id": "no",
      "letter": "B",
      "probabilities": {"yes": 0.12, "no": 0.88},
      "has_image": true,
      "image_tokens": 1024,
      "input_tokens": 312
    }
  ],
  "timing": {
    "total_seconds": 0.61,
    "inference_seconds": 0.60,
    "image_seconds": 0.02,
    "encode_seconds": 0.16,
    "prefill_seconds": 0.42,
    "suffix_seconds": 0.03,
    "image": true,
    "batch_size": 4,
    "prefix_tokens": 1280
  }
}
```

`timing.total_seconds` 是**从服务端拿到请求数据到推理完成**的时间（不含客户端与服务器之间的网络传输）；细分：`image_seconds`（图像解码）、`encode_seconds`（prompt 编码）、`prefill_seconds`（图像+state 前向）、`suffix_seconds`（判据前向）。每个任务不再单独计时（shared 模式下它们共享同一次判据前向）。

### 前端测试页

`examples/index.html` 是单文件多任务测试页：勾选 打架 / 摔倒 / 挥手 / 捂胸口（“是否有人”始终判定），对同一张画面一次调用 `/decide-batch`，并把结果归一为统一格式：

```json
{"has_person": 0, "violations": ["打架", "挥手"]}
```

勾选“同时跑完整自回归对比”后，还会调 `/generate`，把生成结果也归一成同一格式，并展示两边的逐判据一致率、耗时与 `tok/s`。可上传图像，或点“加载示例图”用页面内嵌的示例（不依赖外网）。

`examples/replay.html` 是速度对比回放页：并行发起 direct 与 generate，direct 结果立即出现；generate 返回后，按它**真实生成耗时**用打字机把生成内容逐字回放（可选 1×/2×/4×/8× 回放速度，耗时数字始终是真实值），直观展示 direct 相对完整自回归的加速倍数。

```bash
python serve.py --fake          # 启动服务
# 浏览器打开 examples/index.html
```

### 自回归对比

`examples/compare_generation.py` 用 `model.generate`（贪心，内部逐 token，**无手写解码循环**）对同一张图、同一组判据做完整自回归生成，并与 direct 的共享前缀批量（图像 prefill 1 次 + 判据前向 1 次）对比耗时与答案一致率。

```bash
CUDA_VISIBLE_DEVICES=0 python examples/compare_generation.py \
  --model Qwen/Qwen3.5-4B \
  --image examples/fall.png
```

输出每个任务的 direct / generate 判定与是否一致，以及两边耗时：

- direct：`prefill_seconds`、`suffix_seconds`、`inference_seconds`（2 次前向）
- generate：`prompt_tokens`、`new_tokens`、`generate_seconds`、tok/s、`total_seconds`

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