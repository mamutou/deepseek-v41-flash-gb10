#!/usr/bin/env python3
"""输出成色自检：复现并验证 DS4.1 的 tokenizer 解码缺陷。

背景：这份检查点的 tokenizer 带 826 个含 `▁` 的 added token。`skip_special_tokens=True`
（多数客户端的默认值）会把 `▁` 一起剥掉，于是 `中国首都为北京。` 被解成
`**北京_（北京是北京市的简称…` 这种脏串，甚至出现 token 级复读。
正确用法是 **skip_special_tokens=False**。

用法：python3 check_quality.py [服务地址]
"""
import json
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:30000"

QUESTIONS = [
    "中国的首都是哪里？",
    "写一句关于秋天的诗。",
    "用一句话说明为什么天空是蓝色",
    "把这句话翻译成英文：今天天气很好",
]

# 泄漏/退化信号：added token 残留、重复标点、替换字符、代码围栏残片
LEAKS = ("▁", "｜", "＜｜", "\ufffd", "````")


def ask(q, skip_special_tokens, max_tokens=96):
    body = {
        "model": "DeepSeek-V4.1-Flash",
        "messages": [{"role": "user", "content": q}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "skip_special_tokens": skip_special_tokens,
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)["choices"][0]["message"]["content"].strip()


def repeats(text):
    """找相邻 token 级重复：同一个 2-4 字片段连续出现 3 次以上。"""
    worst = 0
    for size in (2, 3, 4):
        for i in range(len(text) - size * 3):
            frag = text[i:i + size]
            if text[i:i + size * 3] == frag * 3:
                worst = max(worst, size * 3)
    return worst


print("=== 1. 解码开关对照（同一问题）")
q = QUESTIONS[0]
for flag in (True, False):
    out = ask(q, flag)
    print(f"  skip_special_tokens={str(flag):5} -> {out[:70]!r}")

print("\n=== 2. 四题成色（skip_special_tokens=False）")
bad = 0
for q in QUESTIONS:
    out = ask(q, False)
    leaks = [t for t in LEAKS if t in out]
    rep = repeats(out)
    ok = not leaks and rep == 0
    bad += 0 if ok else 1
    tag = "CLEAN" if ok else f"DIRTY repeats={rep}"
    print(f"  [{tag}] leaks={leaks} {q} -> {out[:80]!r}")

print(f"\n结论：{len(QUESTIONS) - bad}/{len(QUESTIONS)} 干净")
if bad:
    print("  出现脏输出时先确认 skip_special_tokens=False；仍脏则检查 chat_template 是否缺失")
