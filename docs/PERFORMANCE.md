# 实测性能与调参边界

全部数字在**一台** NVIDIA DGX Spark 上测得（GB10，20 核无 SMT，121.63 GiB 统一内存，
LPDDR5X 共享给 CPU 与 GPU），无第二台机器参与。

## 1. 模型的内存账

检查点 475.2 GiB，单机只有 121.63 GiB，所以必须分流：

| 部分 | 体积 | 去处 |
|---|---|---|
| routed experts（40 层 × 384 专家 × top-6，mxfp4）| 275.7 GiB | SSD 冷池 + 少量常驻槽位 |
| Engram 表（第 1、14 层，3.84 亿行 × 264 B）| 189.1 GiB | NVMe 行表（`XTIER_ENGRAM_ROWS=nvme`）|
| 常驻（attn / embed / lm_head / shared expert）| 10.5 GiB | 显存常驻 |

## 2. 配置 → 速度

`tok/s` 均为 decode 速率（输出 token / 秒），提示词「请用中文介绍一下长江，写大约一百字。」，
`max_tokens=32`，`temperature=0`。

| 配置（上下文 / GPU 槽位 / 并发）| 单请求 | 4 路聚合 | 6 路聚合 |
|---|---|---|---|
| `CPUQuota=70%`（=0.7 核）16K / 12 / 1 | **0.34** | — | — |
| `CPUQuota=1600%` 16K / 12 / 1 | 2.90 | — | — |
| 16K / 40 / 4 | 3.79（热）| 6.24 | — |
| 8K / 56 / 4 | 4.04 | 7.6–9.45 | — |
| 32K / 40 / 4 | 3.69 | 6.32 | — |
| **32K / 40 / 6** | **4.22** | 6.31 | 6.99 |
| 32K / 48 / 6 | 3.08 | 6.12 | **8.27** |

**冷热差异约 25%**：服务刚加载完的首个请求（专家缓存还冷）≈2.7–2.8 tok/s，跑几十个 token 后
升到 4.2。给用户报数时说清是哪一档。

**单请求延迟最优**：`CTX=32768 SLOTS=40 FRAC=0.90 MEM=0.95 MAXREQ=6 CPUQ=1600%`（4.22 tok/s）
**聚合吞吐最优**：`CTX=32768 SLOTS=48 FRAC=0.88 MEM=0.96 MAXREQ=6`（8.27 tok/s @6 路）

## 3. 三个硬边界（不是调参能让开的）

### 3.1 上下文上限 32K

KV 约 **1.15 MB/token**（40 层 MLA + c4/c128 压缩池）：

| 上下文 | KV 需求 | 能否 |
|---|---|---|
| 32K | ~38 GB | ✅（`MEM=0.95` 下 KV 余 37.7 GB）|
| 64K | ~73 GB | ❌ 加上权重的 85 GB 超出 121.63 GiB |
| 128K+ | — | ❌ 需要双 Spark |

### 3.2 GPU 槽位上限 48–56

权重占用 ≈ `20.0 + 1.36 × SLOTS` GB（由 20 槽=47.2 GB、56 槽=96.1 GB 两点拟合）。
槽位每多一个多吃 1.36 GB，直接挤 KV：

- `SLOTS=64` 任何上下文都起不来：`Loaded weights leave no GPU memory for the KV cache under --mem-fraction-static=0.92`（要求 ≥0.968）
- `SLOTS≤20` 能启动但一请求就崩：`[xtier] CPU forward failed (m=8 k=6): No RAM slot for expert 92` → NaN → `Prior CPU callback failed`。prefill 的 `m=8 k=6` 调用最多命中 48 个不同专家，槽位低于这个数就没有落点。

### 3.3 并发上限 6

`MAXREQ=8` 会让 KV 池换成另一套布局（`swa_size=10496` + `c4`/`c128` 池），
走到的稀疏 MLA prefill 内核不支持：

```
tvm.error.InternalError: Unsupported sparse-MLA prefill configuration:
model=DSV4 num_heads=64 topk=128 page_block_size=64 topk_extra=512
```

5–7 都能跑，8 必崩。

## 4. 一个 token 的时间去哪了

`XTIER_ROUTE_TRACE` 直接给出每层 CPU 调用（字段是 **ns**）：

```
每层 wall 3.49 ms = gate 1.48 + down 0.43 + 冷读/搬运 1.58
每 token 跨度 ≈198 ms = 层内 119 ms + 层间空隙 ≈80 ms（GPU 注意力，每层 ~2 ms）
另加 lm_head / 采样 / 调度 ≈65 ms
```

**没有单一大头，四块各占 15–30%**，所以任何单一参数都换不来翻倍。已经证伪的：

- 盘不是瓶颈：`ColdStore` 裸读 batch=1 就 **11 GB/s**（18.8 MB 记录 1.7 ms），盘顺序 4.6 GB/s，随机 18 MiB 4.4–5.1 GB/s
- 算力不是瓶颈：NEON GEMV m=1 跑 32 专家 6 ms（187 µs/专家）；GEMM m=8 244 GFLOP/s、m=512 392 GFLOP/s，占端到端 ~2.4%
- GPU 不是瓶颈：利用率 2–8%
- indexer 不是瓶颈：torch 回退与 deep_gemm 原生实测 2.718/3.819 vs 2.725/3.788，完全一致

## 5. 试过但更慢的（别重复踩）

| 尝试 | 结果 | 原因 |
|---|---|---|
| `XTIER_CACHE_SWAPS=2 EVERY=16`（降换页）| 0.208 tok/s | swap 本身就是按 routes 的软件预取，关掉冷读从 24 涨到 82/步 |
| `XTIER_COLD_READ_AHEAD_COVER=0.2` + `COLD_THREADS=16` | 0.228 / 0.261 | 只多读不重叠，冷读涨到 86–103/步，质量还变差 |
| `SLOTS=40` + `CPUQuota=70%` | 0.242 | 配额没修，白搭 |
| `--disable-radix-cache` | 无提速 | 它既不是复读的原因，也不是速度的杠杆；关了只损失多轮前缀命中 |
| `XTIER_DSV41_TORCH_DECODE_INDEXER=1` | 同上 | indexer 路径可换但同速 |

## 6. 投机解码（DSPARK）：能跑，但不划算

这套 SGLang **自带完整 DSPARK 栈**（`speculative/dspark_components/`），草稿头来自检查点自带的
`mtp.0.*`（2401 个键），启动成功的指纹是：

```
Initialized DSpark draft runner. model=DeepseekV4ForCausalLMDSpark, gamma=G
xtier: DSpark draft LM head as a row-scaled FP8 copy (129280, 5120)
DSpark draft proposal (greedy only) folded into the draft cuda graph
```

| gamma | 单请求 | 4 路聚合 | 每步 accept len |
|---|---|---|---|
| 不投机 | 4.22 | 6.31 | — |
| 4（窗口错配）| 2.18 | 3.79 | ~1.2 |
| 5（检查点原生）| 3.69（打平）| 3.56（更差）| 1.0 / 2.0 / 4.0 抖动 |

**结论：保持关闭。** 每步要把 6 个 token 一起验证，并集专家从 6 涨到最多 36（CPU 成本 2–3×），
而平均只接受 2–3 个，正好抵消。accept len 在 1.0 与 4.0 之间抖动（出现过 accept rate 0.0 的步）
说明这份 MTP 头与主模型对齐不稳定，不是调 gamma 能救的。

`gamma` 必须等于检查点自带的 draft `block_size`（1017 是 **5**），传错值启动仍成功，但日志会留
`DSpark gamma mismatch: using gamma=4 ... but draft config block_size=5`，接受率被压低。

## 7. 还想再快只能换硬件

- **双 Spark（TP2）**：内存 243 GiB，可开 64K+ 且槽位不受 KV 顶，按现有曲线外推约 8–12 tok/s 单请求
- **x86 + AVX-512**：GEMV 比 NEON 快 3–5 倍，CPU MoE 那 76 ms 能压到 20 ms 以内
