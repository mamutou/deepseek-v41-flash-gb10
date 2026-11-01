#!/usr/bin/env python3
"""并发吞吐基准：N 个请求同时发出，测聚合 tok/s 与单请求延迟。

用法：
    python3 bench_throughput.py                 # 1 路，32 token
    python3 bench_throughput.py 4               # 4 路并发
    python3 bench_throughput.py 4 64 http://127.0.0.1:30000
"""
import json
import sys
import threading
import time
import urllib.request

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1
TOK = int(sys.argv[2]) if len(sys.argv) > 2 else 32
BASE = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:30000"
MODEL = "DeepSeek-V4.1-Flash"

# 用不同提示词，避免所有请求命中同一前缀缓存
PROMPTS = [
    "请用中文介绍一下长江，写大约一百字。",
    "请用中文介绍一下黄河，写大约一百字。",
    "请用中文介绍一下珠江，写大约一百字。",
    "请用中文介绍黑龙江，写大约一百字。",
    "请用中文介绍一下淮河，写大约一百字。",
    "请用中文介绍一下海河，写大约一百字。",
]
res = [None] * N


def one(i):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}],
        "max_tokens": TOK,
        "temperature": 0,
        # 关键：DS4.1 的 tokenizer 少了 added-token 处理，True 会把文本解成
        # `**北京_（北京是…` 这种脏串。必须 False。
        "skip_special_tokens": False,
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    res[i] = (time.time() - t0, d["usage"]["completion_tokens"])


t0 = time.time()
ts = [threading.Thread(target=one, args=(i,)) for i in range(N)]
for t in ts:
    t.start()
for t in ts:
    t.join()
wall = time.time() - t0
total = sum(r[1] for r in res if r)
print(f"并发 {N}，每请求 {TOK} token：wall={wall:.1f}s 总 token={total} 聚合={total / wall:.2f} tok/s")
print(f"各请求耗时: {[f'{r[0]:.1f}s' for r in res if r]}")
