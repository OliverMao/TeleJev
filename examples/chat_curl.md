#!/usr/bin/env bash
# TeleJev /v1/chat/completions 调用示例。

tasks 可省略：留空时服务端从提示词里的“任务名称：X”解析。

 ```bash
   BASE=http://10.127.48.252:32072

   curl -sS -X POST "$BASE/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "telejev",
       "messages": [
         {
           "role": "system",
           "content": "你是实时视频流助手，逐帧观察连续的摄像头画面，只依据【最后一帧】的状态判定。"
         },
         {
           "role": "user",
           "content": [
             {
               "type": "image_url",
               "image_url": { "url": "data:image/jpeg;base64,<这一帧的 base64，省略…>" }
             },
             {
               "type": "text",
               "text": "【任务清单】\n任务名称：摔倒\n任务名称：挥手\n任务名称：捂胸口"
             }
           ]
         }
       ],
       "tasks": ["摔倒", "挥手", "捂胸口"]
     }'
 ```

 返回示例：

 ```json
{"id": "chatcmpl-225686a467dd1affc56d6dbe", "object": "chat.completion", "created": 1790139931, "model": "/workspace/s/yuanc/jzy/VideoCam/VLLM/Qwen3.5-4B", "choices": [{"index": 0, "message": {"role": "assistant", "content": "{\"has_person\": 1, \"violations\": []}"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 371, "completion_tokens": 0, "total_tokens": 371, "prompt_tokens_details": {"cached_tokens": 0}}, "telejev": {"tasks": ["摔倒", "挥手", "捂胸口"], "has_person_probabilities": {"A": 0.7057850154599563, "B": 0.2942149845400438}, "violation_probabilities": {"摔倒": {"A": 0.37754066879814546, "B": 0.6224593312018546}, "挥手": {"A": 0.320821300824607, "B": 0.679178699175393}, "捂胸口": {"A": 0.4378234991142019, "B": 0.5621765008857981}}, "raw_tokens": ["A", "B", "B", "B"], "requests": 4, "http_requests": 1, "mode": "batch", "concurrency": 1, "prompt_tokens": 371, "cached_tokens": 0, "requests_detail": [{"prompt_tokens": 0, "cached_tokens": 0, "seconds": 0.017114761285483837}, {"prompt_tokens": 0, "cached_tokens": 0, "seconds": 0.017114761285483837}, {"prompt_tokens": 0, "cached_tokens": 0, "seconds": 0.017114761285483837}, {"prompt_tokens": 0, "cached_tokens": 0, "seconds": 0.017114761285483837}], "total_seconds": 0.06872447952628136, "sum_request_seconds": 0.06845904514193535, "batch_seconds": 0.06845904514193535}}
 ```
