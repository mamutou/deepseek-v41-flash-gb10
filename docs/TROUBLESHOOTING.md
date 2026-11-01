# 排障手册

按「症状 → 真因 → 处理」组织。每条都在这台机器上真踩过。

## 速度只有 0.3 tok/s，所有东西都空闲

**症状**：decode 0.3–0.35 tok/s；`vmstat` 的 `us` 恒定 4%（= 1/20 核）、`id` 95%；
`nvidia-smi` 利用率 2–8%；盘几乎不读；每层 CPU 调用 72 ms 而 GEMM 只需 1–2 ms。

**真因**：`CPUQuota=70%`。systemd 的 `CPUQuota` 百分比是**相对单核**的，`70%` = 0.7 个核，
不是「机器的 70%」。这个本想防整机 wedge 的护栏，把 40 层引擎锁在不到一核。

**处理**：`CPUQuota=1600%`（16 核）。同配置 **0.34 → 2.9 tok/s（8.5 倍）**，质量无变化。
护栏对防 wedge 本身没有作用（wedge 的真因是内存/页缓存），所以调大不增加风险。

## 启动成功，但一发请求就崩

```
[xtier] CPU forward failed (m=8 k=6): No RAM slot for expert 92; output set to NaN
RuntimeError: Prior CPU callback failed
```

**真因**：RAM 槽位不够。prefill 的 `m=8 k=6` 调用最多命中 48 个不同专家；槽位（≈`384 × (1-FRAC)`）
低于这个数就没有落点。`SLOTS=20 FRAC=0.95`（≈19 个槽位）必然崩。

**处理**：槽位 ≥40。注意 GPU 槽位（`XTIER_CACHE_SLOTS`）和 RAM 槽位（由 `FRAC` 决定）是两个量，
但都要从同一块统一内存里出。

## `Loaded weights leave no GPU memory for the KV cache`

```
ValueError: Loaded weights leave no GPU memory for the KV cache under
--mem-fraction-static=0.92. Raise --mem-fraction-static above 0.968
```

**真因**：权重已经把预算吃光，KV 一块都分不到。`SLOTS=64` 在任何上下文下都会这样。

**处理**：降 `SLOTS`（权重 ≈ `20.0 + 1.36 × SLOTS` GB）。别靠抬 `MEM` 硬塞——抬 mem-fraction
只是把 KV 抢过来、逼死槽位，最后仍是 `No RAM slot`。`MEM` 保持 0.95–0.96。

## `MAXREQ=8` 起不来

```
tvm.error.InternalError: Unsupported sparse-MLA prefill configuration:
model=DSV4 num_heads=64 topk=128 page_block_size=64 topk_extra=512
```

**真因**：`max_running_requests=8` 会让 KV 池换成另一套布局（`swa_size=10496` + `c4`/`c128` 池），
稀疏 MLA prefill 内核没有这条路径。

**处理**：并发 ≤6（5、6、7 都验证可用）。

## 长提示词一发就把服务打挂

```
tvm.error.InternalError: Check failed: (ok) is false: Unsupported sparse-MLA prefill configuration:
model=DSV4 num_heads=64 topk=128 page_block_size=64 topk_extra=512 extra_page_block_size=128
[2026-11-01 20:54:36] SIGQUIT received. signum=None, frame=None. It usually means one child failed.
```

**真因**：`index_topk=512`，超过它的序列切到稀疏 MLA prefill，而 `flashinfer` 的
`_sparse_mla_sm120` 不支持当前池配置（`page_block_size=64` + `topk_extra=512` +
`extra_page_block_size=128`）。**进程会直接死，不是返回错误码**，客户端看到的是
`RemoteDisconnected`。

**实测**：几十 token 正常（dense 路径）、~700 token 崩、~2400 token 崩。阈值很可能就在
`index_topk=512` 附近，用 `scripts/bisect_prefill.py` 二分确认。

**处理**：目前只能避开长输入（短提问 + 长输出），或去改内核路径（`--page-size`、attention backend）。
同一报错也是 `--max-running-requests 8` 起不来的原因。

## 输出里有 `**北京_（北京是北京市的简称…` 这种脏串，或者复读

**真因**：`skip_special_tokens=True`。这份 tokenizer 带 826 个含 `▁` 的 added token，
默认解码会把 `▁` 剥掉、把文本解坏。

**处理**：请求里显式 `"skip_special_tokens": false`。对照：

```
True  -> '**北京_（北京是北京市的简称…'
False -> '中国首都为北京。'
```

顺带：这个检查点（以及同门的 V4-Flash-0731）**没有 chat_template**，要用 `encoding/encoding.py`
的 `encode_messages` 生成的格式补一份 `chat_template.jinja` 并写进 `tokenizer_config.json`，
否则 SGLang 会走默认模板、多轮对话格式错位。

## 投机解码起不来

两道闸都在 `sglang/srt/arg_groups/deepseek_v4_hook.py`：

1. `assert cfg.speculative_algorithm in ("EAGLE", "DSPARK")`——`NEXTN` 被拒
2. `validate_deepseek_v41_features` 的 `unsupported` 列表里
   `"speculative decoding other than DSpark"`——放宽第一道后仍被第二道拒掉

**处理**：用 `DSPARK`，并让 `--speculative-dspark-block-size G` 与
`--speculative-num-draft-tokens G+1` 相等、且 `G` 等于检查点 draft 的 `block_size`（1017 是 5）。
**但实测净零收益，建议保持关闭**，见 `PERFORMANCE.md` §6。

## 读不到接受长度 / `/metrics` 是空的

这套栈默认不打 decode 统计，`/metrics` 也没有 spec 指标。加 `--decode-log-interval 1`，
从 `Decode batch, ... accept len: N, accept rate: R, ... gen throughput (token/s): T` 读。

**别用短提示词的 wall time 反推**：提示词如果提前 EOS（只产 3 个 token），时间几乎全是 prefill，
会得出 1.02 tok/s 这种假数字。

## `pkill` 把自己杀了

`pkill -f sglang.launch_server` 会匹配到执行它的那个 ssh 会话（命令行里就含这个字符串），
会话直接断。写成 `pkill -f '[s]glang.launch_server'`。

## 用 ssh + heredoc 改远端脚本，改动没生效

```
SyntaxError: if NEXTN, in src
```

heredoc 层会把 Python 字面量里的双引号吃掉，脚本根本没跑到替换那步。
**改任何远端文件都走「本地 write_file → scp → 远端执行 → 打印改后的行号上下文」**。
改大于 3 行的 shell 脚本同理：整份覆盖 → `bash -n` → 再启动。

## 变量名既当路径又当数值

`COLD=~/models/ds41-cold` 又写 `XTIER_COLD_FRACTION="${COLD:-0.8}"` →
启动十秒后 `could not convert string to float`（还要等 40 层加载完才炸）。
路径 / 比例 / 脚本 / 输出目录一律分名（`COLDDIR` / `FRAC` / `SCRIPT` / `OUTDIR`）。

## 硬编码覆盖了环境变量

`env XTIER_THREADS=16 ...` 写在 `env` 行里时，外层 `XTIER_THREADS=20 ./start_ds41.sh` **不会生效**。
发布版已改成 `XTIER_THREADS="${THREADS:-16}"`。排查同类问题时先 `grep` 脚本里该变量是不是被写死了。
