#!/usr/bin/env bash
# DeepSeek-V4.1-Flash-1017 on ONE NVIDIA DGX Spark (GB10, 121.63 GiB unified memory)
# via the XTier CPU expert engine.
#
# 实测账目（475.2 GiB 检查点）：
#   routed experts 275.7 GiB | engram 189.1 GiB | 常驻（attn/embed/lm_head/shared）10.5 GiB
#   → 单机放不下，必须让 275.7 GiB 的专家走 SSD 冷池 + 189.1 GiB 的 Engram 表走 NVMe 行表
#
# 用法：  ./start_ds41.sh [额外的 sglang 参数...]
# 可调：  CTX CTXPOOL MEM SLOTS FRAC MAXREQ BATCH WARM ROWS_CACHE
#         CPUQ MEMMAX MEMHIGH THREADS COLDDIR MODEL_DIR VENV_PY XTIER_DIR
#
# 默认值是单请求延迟最优档（见 docs/PERFORMANCE.md）：
#   32K 上下文、单请求 ~4 tok/s、4 路聚合 ~6.3 tok/s
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-$HOME/models/ds41-flash-1017}
COLDDIR=${XTIER_COLD_PATHS:-$HOME/models/ds41-cold}
VENV_PY=${VENV_PY:-$HOME/venvs/sglang-521/bin/python}
XTIER_DIR=${XTIER_DIR:-$HOME/xtier4arm}
LOG_DIR=${LOG_DIR:-$HOME/logs}

EXTRA=(); [ "${WARM:-0}" = "1" ] || EXTRA+=(--skip-server-warmup)

# --- DRY 模式：只打印展开后的命令，不杀进程、不等内存、不加载权重 ----------
if [ "${DRY:-0}" = "1" ]; then
  echo "systemd-run --user --scope --collect --unit=ds41-<ts> \\"
  echo "  -p MemoryMax=${MEMMAX:-100G} -p MemoryHigh=${MEMHIGH:-88G} -p MemorySwapMax=0 \\"
  echo "  -p CPUQuota=${CPUQ:-1600%} \\"
  echo "  env MODEL_DIR=$MODEL_DIR PY=$VENV_PY XTIER_DIR=$XTIER_DIR \\"
  echo "      CONTEXT=${CTX:-32768} CTXPOOL=${CTXPOOL:-${CTX:-32768}} MAXREQ=${MAXREQ:-6} \\"
  echo "      MEM=${MEM:-0.95} \\"
  echo "      XTIER_CACHE_SLOTS=${SLOTS:-40} XTIER_COLD_FRACTION=${FRAC:-0.90} \\"
  echo "      XTIER_THREADS=${THREADS:-16} XTIER_COLD_PATHS=$COLDDIR \\"
  echo "      bash $XTIER_DIR/run_single_spark.sh ${EXTRA[*]:-} $*"
  exit 0
fi

# --- 清理旧实例并等内存回收 ------------------------------------------------
# 注意匹配式要写成 '[s]glang.launch_server'：直接写 sglang.launch_server 会匹配到
# 执行这条命令的 ssh 会话自己，把当前 shell 杀掉。
pkill -f '[s]glang.launch_server' 2>/dev/null || true
for _ in $(seq 1 90); do pgrep -f '[s]glang.launch_server' >/dev/null || break; sleep 2; done
# GB10 是统一内存：上一个实例退出了，页缓存回收还要时间，早启动会被 OOM 掉
for _ in $(seq 1 60); do
  a=$(LC_ALL=C free -g | awk '/^Mem:/{print $7}')
  [ "${a:-0}" -ge 100 ] && break
  sleep 5
done
echo "[ds41] available ${a:-?}G before start"

mkdir -p "$COLDDIR" "$LOG_DIR"

# --- cgroup 护栏 -----------------------------------------------------------
# MemoryMax 是防整机 wedge（超内存会整机卡死，不是只杀进程）。
# CPUQuota 的单位是「相对单核的百分比」：70% = 0.7 个核，不是机器的 70%。
# 这里必须给足核数，否则 40 层引擎被锁在不到一核 —— 这正是 0.34 tok/s 的真因。
UNIT="ds41-$(date +%s)"

exec systemd-run --user --scope --collect --unit="$UNIT" \
  -p MemoryMax="${MEMMAX:-100G}" -p MemoryHigh="${MEMHIGH:-88G}" -p MemorySwapMax=0 \
  -p CPUQuota="${CPUQ:-1600%}" \
  env MODEL_DIR="$MODEL_DIR" SERVED="${SERVED:-DeepSeek-V4.1-Flash}" \
      PY="$VENV_PY" XTIER_DIR="$XTIER_DIR" \
      CONTEXT="${CTX:-32768}" CTXPOOL="${CTXPOOL:-${CTX:-32768}}" \
      MAXREQ="${MAXREQ:-6}" BATCH="${BATCH:-1024}" MEM="${MEM:-0.95}" WARM="${WARM:-0}" \
      XTIER_DSV41_TORCH_DECODE_INDEXER="${TORCH_INDEXER:-0}" \
      XTIER_ENGRAM_ROWS=nvme XTIER_ENGRAM_CACHE_GIB="${ROWS_CACHE:-4}" XTIER_ENGRAM_PREFETCH=1 \
      XTIER_ENGRAM_MODEL="${XTIER_ENGRAM_MODEL:-$MODEL_DIR}" \
      XTIER_COLD_PATHS="$COLDDIR" \
      XTIER_CACHE_SLOTS="${SLOTS:-40}" XTIER_COLD_FRACTION="${FRAC:-0.90}" \
      XTIER_CACHE_SWAPS="${SWAPS:-16}" XTIER_CACHE_EVERY="${EVERY:-4}" \
      XTIER_CACHE_MIN_GAIN="${GAIN:-1.5}" XTIER_COLD_HALF_LIFE="${HALFLIFE:-50}" \
      XTIER_THREADS="${THREADS:-16}" XTIER_FAST_LOAD=1 \
      bash "$XTIER_DIR/run_single_spark.sh" "${EXTRA[@]}" "$@"
