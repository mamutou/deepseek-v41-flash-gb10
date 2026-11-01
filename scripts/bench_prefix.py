#!/usr/bin/env python3
"""前缀缓存（radix cache）验证：同一段长前缀发两次，第二次的耗时就是命中后的成本。

加速比 > 1.5 说明 radix cache 生效；≈1.0 说明被 --disable-radix-cache 关了，
或池子不够放下这段前缀（32K 上下文下要留意 CTXPOOL）。

用法：python3 bench_prefix.py [填充句重复次数] [服务地址]
"""
import json
import sys
import time
import urllib.request

REPEAT = int(sys.argv[1]) if len(sys.argv) > 1 else 120
BASE = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:30000"

PREFIX = "以下是背景资料，请只回答最后的问题。" + "这是一段用于填充上下文的中文材料，内容无实际意义。" * REPEAT
Q = "\n\n问题：以上材料重复的句子是什么？只回答一句。"


def call(max_tokens=8):
    body = {
        "model": "DeepSeek-V4.1-Flash",
        "messages": [{"role": "user", "content": PREFIX + Q}],
        "max_tokens": max_tokens,
        "temperature": 0,
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
    return time.time() - t0, d["usage"]["prompt_tokens"], d["usage"]["completion_tokens"]


t1, pt, ct = call()
print(f"第 1 次（冷前缀）：{t1:6.2f}s  prompt={pt} tok  completion={ct}")
t2, pt, ct = call()
print(f"第 2 次（同前缀）：{t2:6.2f}s  prompt={pt} tok  completion={ct}")
print(f"加速比 = {t1 / t2:.2f}×  " + ("前缀缓存生效" if t1 / t2 > 1.5 else "前缀缓存未生效"))
