#!/usr/bin/env bash
# 两台 8 卡机器并发 DMD 训练 (共 16 卡), 从 checkpoint_model_000500 续训。
# 沿用 run_dmd_local_cache.sh 的本地缓存方案: 各节点独立把权重同步到自己的 /dockerdata,
# 之后各自从本地盘读, 避免 16 rank 同时打 cephfs。
#
# 用法(两台机器分别执行, NODE_RANK 必须不同):
#   本机(MASTER, 29.209.106.198):  NODE_RANK=0 bash tools/run_dmd_2node.sh
#   另一台机器(29.209.115.41):     NODE_RANK=1 bash tools/run_dmd_2node.sh
set -u

ROOT=/apdcephfs/private_huitinglu/Self-Forcing
PY=/apdcephfs/private_huitinglu/conda_envs/bernini_distill/bin/python
CONFIG=$ROOT/configs/self_forcing_v2v_dmd.yaml
LOGDIR=$ROOT/logs/bernini_v2v_dmd_routeb
CACHE=/dockerdata/ckpt_cache

NODE_RANK=${NODE_RANK:?"必须指定 NODE_RANK=0(主节点,29.209.106.198) 或 1(从节点,29.209.115.41)"}
NNODES=2
MASTER_ADDR=${MASTER_ADDR:-29.209.106.198}
MASTER_PORT=${MASTER_PORT:-29552}
RDZV_ID=${RDZV_ID:-27520}

cd "$ROOT" || exit 1
export PATH=/apdcephfs/private_huitinglu/conda_envs/bernini_distill/bin:$PATH
mkdir -p "$LOGDIR" "$CACHE"

abspath() { case "$1" in /*) echo "$1";; *) echo "$ROOT/$1";; esac; }

get_cfg() {
  grep -E "^$1:" "$CONFIG" | head -1 \
    | sed -E "s/^$1:[[:space:]]*//; s/[[:space:]]*#.*$//; s/[[:space:]]*$//"
}

GEN_SRC=$(abspath "$(get_cfg generator_ckpt)")
REAL_SRC=$(abspath "$(get_cfg real_score_v2v_ckpt)")

GEN_DST=$CACHE/${GEN_SRC#/}
REAL_DST=$CACHE/$(basename "$REAL_SRC")
mkdir -p "$(dirname "$GEN_DST")"

sync_one() {
  local s=$1 d=$2
  if [ ! -f "$s" ]; then echo "[cache][ERR] 源不存在: $s"; exit 1; fi
  if [ -f "$d" ] && [ "$(stat -c%s "$s")" = "$(stat -c%s "$d")" ]; then
    echo "[cache] 命中(大小一致), 跳过: $d"; return
  fi
  echo "[cache] 同步 $s -> $d"
  local t0=$(date +%s)
  cp -f "$s" "$d.tmp" && mv -f "$d.tmp" "$d"
  echo "[cache] 完成 $d  ($(( $(date +%s) - t0 ))s)"
}

sync_one "$GEN_SRC" "$GEN_DST"
sync_one "$REAL_SRC" "$REAL_DST"

RT=$CACHE/self_forcing_v2v_dmd.runtime.yaml
sed -E \
  -e "s|^generator_ckpt:.*|generator_ckpt: $GEN_DST|" \
  -e "s|^real_score_v2v_ckpt:.*|real_score_v2v_ckpt: $REAL_DST|" \
  "$CONFIG" > "$RT"
echo "[cache] runtime config: $RT"

if [ "${CACHE_ONLY:-0}" = "1" ]; then
  echo "[cache] CACHE_ONLY=1, 仅预热缓存, 不起训。"
  exit 0
fi

echo "[launch] NODE_RANK=$NODE_RANK NNODES=$NNODES MASTER=$MASTER_ADDR:$MASTER_PORT RDZV_ID=$RDZV_ID"

exec "$PY" -m torch.distributed.run --nnodes=$NNODES --node_rank=$NODE_RANK --nproc_per_node=8 \
  --rdzv_id=$RDZV_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  train.py --config_path "$RT" \
  --logdir "$LOGDIR" --disable-wandb \
  >> "$LOGDIR/train_from500_2node_node${NODE_RANK}.log" 2>&1
