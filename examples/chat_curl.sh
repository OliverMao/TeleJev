#!/usr/bin/env bash
# TeleJev /v1/chat/completions 调用示例（一次请求判完所有任务）。
#
# 用法:
#   BASE=http://127.0.0.1:22001 bash examples/chat_curl.sh
#
# image_url.url 支持三种形式，示例里 base64 用省略号占位：
#   data:image/jpeg;base64,<这一帧的 base64>     # 内联图像（示例用的形式）
#   /path/to/frame.jpg                           # 本机路径（serve.py 能读到即可）
#   https://example.com/frame.jpg                # http(s) 链接
#
# tasks 可省略：留空时服务端从提示词里的“任务名称：X”解析。

BASE=${BASE:-http://127.0.0.1:8000}

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