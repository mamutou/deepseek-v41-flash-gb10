# DeepSeek-V4.1-Flash-1017 on one DGX Spark

在**单台** NVIDIA DGX Spark（GB10，121.63 GiB 统一内存）上部署并调优
`DeepSeek-V4.1-Flash-1017`（475.2 GiB 检查点，510.3 GB / 83 个分片）的实测记录与可复现脚本。

模型比机器大 3.9 倍，靠 **XTier 的 CPU 专家引擎**（路由专家走 SSD 冷池、Engram 表走 NVMe 行表）跑起来，
再逐项压到这套硬件的上限。

## 实测结果

| 指标 | 数值 |
|---|---|
| 单请求 decode | **4.22 tok/s**（热态）|
| 4 路聚合 | 6.31 tok/s |
| 6 路聚合 | 8.27 tok/s（`SLOTS=48` + `MAXREQ=6`）|
| KV 池 | 可开 **32K**（≈1.15 MB/token）|
| **可用输入长度** | ⚠️ **远小于 32K，实测 ~700 token 就崩**（见下）|
| 并发上限 | 6（8 撞内核断言）|
| 输出成色 | 4/4 题干净（含写诗不复读）|
| 对照起点 | 0.34 tok/s（`CPUQuota=70%` 误配）→ **12.4 倍** |

> 冷热差约 25%：刚加载完的首个请求 ≈2.7 tok/s，跑热后 4.2。

### ⚠️ 已知严重限制：长提示词 prefill 会打挂服务

`index_topk=512` 以上走稀疏 MLA prefill，这条内核路径拒绝当前 KV 池配置，
**scheduler 抛异常后整个服务被 kill**（不是返回 500）：

```
tvm.error.InternalError: Check failed: (ok) is false: Unsupported sparse-MLA prefill
configuration: model=DSV4 num_heads=64 topk=128 page_block_size=64 topk_extra=512
extra_page_block_size=128
  at sglang/kernels/ops/attention/flash_mla_sm120.py::_flash_mla_sm120_prefill
```

- 几十 token 的短请求：正常（走 dense 路径）
- 实测 ~700 token：必崩；2400 token：必崩
- 同一报错也是 `--max-running-requests 8` 起不来的原因

⇒ **"KV 池能开 32K" 不等于 "能吃 32K 输入"**。这一版部署适合短提示词、长输出（每轮几十 token
提问、生成几百 token）的场景。要服务长文档，得先解决这条内核路径（换 `--page-size`、换 attention
backend，或等上游修）。

用 [`scripts/bisect_prefill.py`](scripts/bisect_prefill.py) 在自己的环境上二分出确切阈值：

```bash
python3 scripts/bisect_prefill.py 256 8192     # 命中即停，重启服务后从更小值继续细分
```

## 为什么单机能跑

| 部分 | 体积 | 去处 |
|---|---|---|
| routed experts（40 层 × 384 × top-6，mxfp4）| 275.7 GiB | SSD 冷池 + 少量 GPU/RAM 槽位 |
| Engram 表（第 1、14 层，3.84 亿行 × 264 B）| 189.1 GiB | NVMe 行表 |
| 常驻（attn / embed / lm_head / shared）| 10.5 GiB | 显存 |

## 快速开始

前置：同一台机器上已能跑起 XTier 的 ARM 版运行时
（[xtier](https://github.com/mamutou/xtier)，`arm-neon` 分支），SGLang 锁定 **0.5.21**。

```bash
# 1. 取脚本
git clone https://github.com/mamutou/deepseek-v41-flash-gb10.git && cd deepseek-v41-flash-gb10

# 2. 指向你的检查点与冷池（默认值见脚本头部）
export MODEL_DIR=~/models/ds41-flash-1017
export XTIER_COLD_PATHS=~/models/ds41-cold
export XTIER_DIR=~/xtier4arm
export VENV_PY=~/venvs/sglang-521/bin/python

# 3. 启动（默认即单请求延迟最优档：32K / 40 槽位 / 6 并发）
DRY=1 bash scripts/start_ds41.sh      # 先看展开后的完整命令
bash scripts/start_ds41.sh            # 真启动，约 7 分钟（83 个分片 + 冷池）
```

服务起在 `http://127.0.0.1:30000`，OpenAI 兼容，无 API key：

```bash
curl -s http://127.0.0.1:30000/v1/models
```

**客户端必须显式传 `skip_special_tokens: false`**，否则文本会被解坏（见下方「已知边界」）。

## 验证

```bash
python3 scripts/check_quality.py                    # 成色 + 解码开关对照
python3 scripts/bench_throughput.py 1 32            # 单请求
python3 scripts/bench_throughput.py 4 32            # 4 路聚合
python3 scripts/bench_prefix.py                     # 前缀缓存是否生效
```

## 配置怎么选

| 目标 | 命令 |
|---|---|
| 单请求延迟 | `CTX=32768 SLOTS=40 FRAC=0.90 MEM=0.95 MAXREQ=6` → 4.22 tok/s |
| 聚合吞吐 | `CTX=32768 SLOTS=48 FRAC=0.88 MEM=0.96 MAXREQ=6` → 8.27 tok/s @6 路 |
| 短上下文更快 | `CTX=8192 SLOTS=56 FRAC=0.86 MEM=0.92` → 4.04 tok/s 单请求 |

三个数都不能乱给：`CPUQ` 低于 1600% 会掉回 0.3 tok/s；`SLOTS` >48–56 起不来、≤20 会崩；
`MAXREQ` ≥8 撞内核。

## 已知边界

- **⚠️ 长提示词会打挂服务**：超过 ~512 token（`index_topk`）走稀疏 MLA prefill，该内核路径不支持
  当前池配置，scheduler 抛异常后整个进程被 kill。实测 ~700 token 必崩。**本部署适合短提问 +
  长输出**，不适合长文档。见 [PERFORMANCE.md §3.4](docs/PERFORMANCE.md)。
- **KV 池上限 32K**：64K 需要 ~73 GB KV，加上权重超出 121.63 GiB。但注意池子大小 ≠ 可用输入长度。
- **投机解码（DSPARK）能跑但净零收益**，默认关闭：草稿头确实在检查点里、也能加载，但每步验证
  6 个 token 让并集专家从 6 涨到 36，CPU 成本 2–3 倍，平均只接受 2–3 个。
- **`skip_special_tokens=True` 会解坏文本**：tokenizer 带 826 个含 `▁` 的 added token，
  默认解码把 `▁` 剥掉 → `**北京_（北京是北京市的简称…`，并伴随 token 级复读。
- **检查点缺 chat_template**：需要从 `encoding/encoding.py` 的 `encode_messages` 生成一份并写进
  `tokenizer_config.json`，否则多轮对话格式错位。
- 视觉不可用（tokenizer 无 image 特殊 token）；投机之外没有 EAGLE 路径。

细节：[性能与调参边界](docs/PERFORMANCE.md) · [排障手册](docs/TROUBLESHOOTING.md)

## 目录

```
scripts/start_ds41.sh        # 启动器：参数化 + cgroup 护栏 + DRY=1
scripts/bench_throughput.py  # 单请求 / 并发吞吐
scripts/bench_prefix.py      # 前缀缓存验证（注意：长前缀会触发上面的 prefill 崩溃）
scripts/check_quality.py     # 输出成色 + skip_special_tokens 对照
scripts/bisect_prefill.py    # 二分出 prefill 能承受的输入长度
docs/PERFORMANCE.md          # 完整调参表、时间分解、证伪清单、prefill 边界
docs/TROUBLESHOOTING.md      # 症状 → 真因 → 处理
```

## 相关

- [xtier](https://github.com/mamutou/xtier) — CPU 专家引擎的 AArch64/NEON 移植（本项目依赖它的
  `run_single_spark.sh`）
- [qwen38-flash-gb10](https://github.com/mamutou/qwen38-flash-gb10) — 同一台机器上 Qwen3.8-Flash-Next
  的部署与评测

## License

MIT。所有数字都是在本机实测的，命令、日志指纹和失败案例一并给出，方便你复现或反驳。
