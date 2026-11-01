#!/usr/bin/env python3
"""二分测 prefill 能承受多长的提示词。

这个栈上的稀疏 MLA prefill 内核有一组固定形状，长输入会撞上未支持的配置并把
**整个服务打崩**（不是返回 500，是 scheduler 抛异常后 kill 进程树）：

    tvm.error.InternalError: Unsupported sparse-MLA prefill configuration:
    model=DSV4 num_heads=64 topk=128 page_block_size=64 topk_extra=512

同一报错也是 `--max-running-requests 8` 起不来的原因。KV 池能开 32K ≠ 能 prefill 32K 输入，
所以这两个上限要分别测。

用法（命中崩溃点即停，服务已挂，需重启后从更小的值继续细分）：
    python3 bisect_prefill.py                    # 256 起，逐档翻倍到 8192
    python3 bisect_prefill.py 128 2048 http://127.0.0.1:30000
"""
import json
import sys
import time
import urllib.request

START = int(sys.argv[1]) if len(sys.argv) > 1 else 256
END = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
BASE = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:30000"

FILLER = "这是一段用于填充上下文的中文材料内容无实际意义"


def probe(n_tokens):
    text = FILLER * (n_tokens // len(FILLER) + 1)
    body = {
        "model": "DeepSeek-V4.1-Flash",
        "messages": [
            {"role": "user", "content": text[: n_tokens * 3] + "\n\n只回答两个字：收到"}
        ],
        "max_tokens": 4,
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
    return time.time() - t0, d["usage"]["prompt_tokens"]


n = START
while n <= END:
    try:
        dt, pt = probe(n)
        print(f"  请求 {n:5d} token -> OK  {dt:6.2f}s  实际 prompt_tokens={pt}", flush=True)
    except Exception as e:
        print(f"  请求 {n:5d} token -> 崩了: {type(e).__name__}: {str(e)[:80]}", flush=True)
        print(f"\n崩溃阈值：< {n} token。服务已挂，需重启后从 {n} 以下继续细分。")
        sys.exit(1)
    n *= 2
print("\n全部通过，上限 ≥", END)
