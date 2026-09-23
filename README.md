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

### vLLM 后端

把推理放到运行中的 vLLM OpenAI 兼容服务上，Jev 式读 logits 与生成都走该服务，享受融合 kernel / CUDA Graph / 前缀缓存：

```bash
# 1) 起服务（多图还需加 --limit-mm-per-prompt image=20）
vllm serve /path/to/model --port 30000 \
  --enable-prefix-caching \
  --served-model-name Qwen/Qwen3.5-4B

# 2) 用该后端提供 TeleJev 接口
python serve.py --backend vllm \
  --server-url http://127.0.0.1:30000 \
  --served-model Qwen/Qwen3.5-4B --alias telejev --port 22001
```

- **Jev 式读取**：`max_tokens=1` + `logprobs`，只对选项字母归一化（prefill-only，不生成）
- **生成**：同一服务的普通 `max_tokens` 路径
- **多判据**：把“是否有人”和全部任务放进 vLLM 的 `/v1/chat/completions/batch`，**一次 HTTP 请求**判完；无该端点的旧版 vLLM 自动退回并发逐个读取
- **模型名别名**：`--alias <name>` 后，`/v1/models`、`/health`、`/help` 以及 completions 响应里的 `model` 一律返回该别名；发给 vLLM 的 `model` 仍是 `--served-model`（默认 `--model`）
- **日志**：默认 INFO 每 `--log-interval` 秒（默认 5）汇总一行：端点、请求数、avg/max 耗时、`mode`、`http_requests`、`tasks`、缓存命中（有流量才打，单次请求也会在间隔后打出）；`--log-interval 0` 每个请求一行；`--log-level debug` 额外输出每次 HTTP 访问、上游每次 POST、批量对话数与图像转存；`--log-level warning` 只剩错误

直接对比 Jev 与完整自回归：

```bash
python benchmarks/vllm_compare.py \
  --base-url http://127.0.0.1:30000 --model Qwen/Qwen3.5-4B --image demo/fall.png
```

默认监听 `http://127.0.0.1:8000`，可用 `--host` / `--port` 修改。

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/help` | 调用说明：端点列表、请求/响应示例。 |
| `POST` | `/decide` | 请求体就是一条决策行（可含可选 `image`），返回固定决策对象。 |
| `POST` | `/decide-batch` | 一份 state/image + 多个 criteria，共享一次图像 prefill；返回每个任务结果与耗时。 |
| `POST` | `/generate` | 同一组 criteria 走完整自回归生成（`model.generate`），返回文本、解析结果与耗时。 |
| `POST` | `/v1/chat/completions` | OpenAI 兼容：像普通模型一样发一次请求（图像 + 提示词），返回 `{has_person, violations}`。 |
| `GET` | `/v1/models` | OpenAI 兼容的模型列表。 |
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

### OpenAI 兼容的 Jev 接口

`/v1/chat/completions` 对客户端来说就是**普通自回归模型**：发一次 chat 请求（可含图像 + 你自己的提示词），收到一条 assistant 消息。内部怎么打给 vLLM 由服务端决定——它会把“是否有人”和所有任务放进一次批量请求（`/v1/chat/completions/batch`）做 **prefill-only logprob 读取**，再拼装结果。**输出永远是**：

```json
{"has_person": 0 或 1, "violations": ["行为名称", ...]}
```

客户端只需在提示词里列出任务（或显式传 `tasks`）：

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "telejev",
    "messages": [
      {"role": "system", "content": "你是实时视频流助手，逐帧观察摄像头画面并判定监测行为。"},
      {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
        {"type": "text", "text": "任务名称：摔倒\n任务名称：挥手\n请只判定这些行为。"}
      ]}
    ],
    "tasks": ["摔倒", "挥手"]
  }'
```

返回：

```json
{
  "object": "chat.completion",
  "choices": [{"message": {"role": "assistant", "content": "{\"has_person\": 1, \"violations\": [\"挥手\"]}"}, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 300, "completion_tokens": 0, "total_tokens": 300},
  "telejev": {
    "tasks": ["摔倒", "挥手"],
    "has_person_probabilities": {"A": 0.9, "B": 0.1},
    "violation_probabilities": {"摔倒": {"A": 0.2, "B": 0.8}, "挥手": {"A": 0.7, "B": 0.3}},
    "requests": 3, "http_requests": 1, "mode": "batch", "concurrency": 1
  }
}
```

- 任务名来源：优先顶层 `tasks`，否则从 messages 里的 `任务名称：X` 解析，都没有则用内置默认集合。
- `violations` 里的名称与任务名逐字一致，便于前端按名点亮卡片。
- `completion_tokens` 为 0（内部不生成 token）；概率放在额外字段 `telejev`，OpenAI 客户端会自动忽略。
- 需 `--backend vllm`（复用服务端 tokenizer / 前缀缓存）。
- **一次 HTTP 请求判完全部任务**：优先走 vLLM 的 `/v1/chat/completions/batch`（较新 vLLM 自带）：一次请求同时带上“是否有人”和全部任务；服务端没有该端点时自动退回并发逐个读取，结果不变。响应里的 `telejev.requests` 是模型读取次数（1 + 任务数），`telejev.http_requests` 是真实上游 HTTP 请求数（批量模式下为 1），`telejev.mode` 为 `batch` / `fanout`。
- **防止图像重复上传（可选）**：给 `serve.py` 加 `--public-url http://<vLLM 能访问到的本机地址>:<port>` 后，data URI / 本地路径的图像会先存进本服务的 `/frames/<id>`，上游凭 URL 只抓取一次（vLLM 按 URL 缓存），不再逐任务重复上传 base64。默认不转存（直接内联）；若上游抓不到该 URL，服务会自动退回内联并关闭转存。
- **前缀缓存（APC）**：请求里 `system + 图像 + 任务清单` 在前、逐任务指令作为**最后一个 user 轮**在后，同一批量请求里的“是否有人”与全部任务共享同一段前缀；开启 vLLM `--enable-prefix-caching`（多图还需 `--limit-mm-per-prompt image=20`）后，后续请求可命中已缓存前缀（**追加式帧历史**：旧帧不变、新帧往尾部加，前缀最稳定）。用 `usage.prompt_tokens_details.cached_tokens` 或 `telejev.cached_tokens` 可直接验证命中量。

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

`--backend vllm` 时，`/decide-batch` 会把全部 criteria 放进**一次** `/v1/chat/completions/batch` 请求（设置 `--public-url` 时图像还会转存 `/frames/<id>`、只抓一次，否则内联）；服务端无该端点时自动退回并发逐个读取。`timing.mode` 为 `batch` / `fanout`，`timing.requests` 为真实上游请求数。

### 前端测试页

`demo/index.html` 是单文件多任务测试页：勾选 打架 / 摔倒 / 挥手 / 捂胸口（“是否有人”始终判定），对同一张画面一次调用 `/decide-batch`，并把结果归一为统一格式：

```json
{"has_person": 0, "violations": ["打架", "挥手"]}
```

勾选“同时跑完整自回归对比”后，还会调 `/generate`，把生成结果也归一成同一格式，并展示两边的逐判据一致率、耗时与 `tok/s`。可上传图像，或点“加载示例图”用页面内嵌的示例（不依赖外网）。

`demo/replay.html` 是速度对比回放页：并行发起 direct 与 generate，direct 结果立即出现；generate 返回后，按它**真实生成耗时**用打字机把生成内容逐字回放（可选 1×/2×/4×/8× 回放速度，耗时数字始终是真实值），直观展示 direct 相对完整自回归的加速倍数。

`demo/completions.html` 是 `/v1/chat/completions` 测试页：自己填 system/user 提示词（预fil 了一套监控任务清单）、可选上传或内嵌示例图像，调用后展示 `{"has_person", "violations"}`、各任务 A/B 概率、请求数/并发/耗时与原始响应。需以 `--backend vllm` 启动。

```bash
python serve.py --fake          # 启动服务
# 浏览器打开 demo/index.html
```

### 自回归对比

`benchmarks/compare_generation.py` 用 `model.generate`（贪心，内部逐 token，**无手写解码循环**）对同一张图、同一组判据做完整自回归生成，并与 direct 的共享前缀批量（图像 prefill 1 次 + 判据前向 1 次）对比耗时与答案一致率。生成的**直接就是最终结构** `{"has_person": 0|1, "violations": ["摔倒", ...]}`，不是中间 yes/no 标签；`/generate` 响应里的 `output` 就是这个对象（`answers` 只是从它反推出来用于逐判据对比）。提示词集中在 `src/telejev/prompt.py`：system 为全局规则（APC 前缀），user 为图像在前、任务清单在后；判定只依据【最后一帧】，`violations` 必须逐字使用任务名。

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/compare_generation.py \
  --model Qwen/Qwen3.5-4B \
  --image demo/fall.png
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
python tests/test_api.py
```

`examples/api_client.py` 是连接真实服务的可运行客户端示例；`examples/chat_curl.sh` 是同一 `/v1/chat/completions` 的 curl 版本（`BASE=http://127.0.0.1:22001 bash examples/chat_curl.sh`）。

## 项目结构

```
src/telejev/          # 库（可 pip install；import telejev）
  core.py             # 模型/图像加载、输入校验
  prompt.py           # 全部提示词（全局规则 / 任务清单拼装）
  direct.py           # Jev 式：单条读选项 logits
  shared.py           # 共享前缀批量（本地）
  batch.py            # /decide-batch 的本地实现
  autoregressive.py   # 自回归基线（本地 model.generate）
  vllm_backend.py     # vLLM OpenAI 兼容后端
  openai_compat.py    # /v1/chat/completions 封装（Jev 式结构化输出）
  api.py              # HTTP 服务（/help /health /decide /decide-batch /generate ...）
  cli.py              # 命令行 JSONL 打分
run.py                # 入口：JSONL 打分
serve.py              # 入口：HTTP 服务
start.sh              # 本地启动示例
demo/                 # 前端页面与示例图（index.html / replay.html / fall.png）
benchmarks/           # 对比脚本（compare_generation.py / vllm_compare.py）
examples/             # 调用示例（api_client.py / chat_curl.sh）
tests/                # 接口离线测试（test_api.py）
```

## License

Project code is released under the [MIT License](LICENSE). Model weights and third-party source records are not included; upstream models retain their licenses. See [THIRD_PARTY.md](THIRD_PARTY.md).