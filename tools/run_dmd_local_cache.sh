#!/usr/bin/env bash
# 方案②本地缓存: 消除每次重启对 cephfs 的全量并发读(8 rank × ~51GB)。
# 首次重启把 generator_ckpt + real_score_v2v_ckpt 同步到节点本地盘 /dockerdata,
# 之后每次重启从本地盘读(GB/s 级), cephfs 零读。
# 步数解析依赖目录名 checkpoint_model_XXXXXX(distillation.py:256), 缓存路径已保留该目录名。
#
# 用法:
#   bash tools/run_dmd_local_cache.sh              # 同步(增量) + 8 卡起训
#   CACHE_ONLY=1 bash tools/run_dmd_local_cache.sh # 仅预热本地缓存, 不起训
set -u

ROOT=/apdcephfs/private_huitinglu/Self-Forcing
PY=/apdcephfs/private_huitinglu/conda_envs/bernini_distill/bin/python
CONFIG=$ROOT/configs/self_forcing_v2v_dmd.yaml
LOGDIR=$ROOT/logs/bernini_v2v_dmd_routeb
CACHE=/dockerdata/ckpt_cache

cd "$ROOT" || exit 1
export PATH=/apdcephfs/private_huitinglu/conda_envs/bernini_distill/bin:$PATH
mkdir -p "$LOGDIR" "$CACHE"

abspath() { case "$1" in /*) echo "$1";; *) echo "$ROOT/$1";; esac; }

get_cfg() { # key -> 取值(去前缀/去行尾注释/去首尾空白)
  grep -E "^$1:" "$CONFIG" | head -1 \
    | sed -E "s/^$1:[[:space:]]*//; s/[[:space:]]*#.*$//; s/[[:space:]]*$//"
}

GEN_SRC=$(abspath "$(get_cfg generator_ckpt)")
REAL_SRC=$(abspath "$(get_cfg real_score_v2v_ckpt)")

# 缓存目标: generator_ckpt 保留原绝对路径结构(含 checkpoint_model_XXXXXX)
GEN_DST=$CACHE/${GEN_SRC#/}
REAL_DST=$CACHE/$(basename "$REAL_SRC")
mkdir -p "$(dirname "$GEN_DST")"

sync_one() { # src dst
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

# runtime config: 仅把两处权重路径改指本地缓存, 其余原样
RT=$CACHE/self_forcing_v2v_dmd.runtime.yaml
sed -E \
  -e "s|^generator_ckpt:.*|generator_ckpt: $GEN_DST|" \
  -e "s|^real_score_v2v_ckpt:.*|real_score_v2v_ckpt: $REAL_DST|" \
  "$CONFIG" > "$RT"
echo "[cache] runtime config: $RT"
echo "  generator_ckpt      -> $GEN_DST"
echo "  real_score_v2v_ckpt -> $REAL_DST"

if [ "${CACHE_ONLY:-0}" = "1" ]; then
  echo "[cache] CACHE_ONLY=1, 仅预热缓存, 不起训。"
  exit 0
fi

exec "$PY" -m torch.distributed.run --nnodes=1 --nproc_per_node=8 \
  --rdzv_id=7752 --rdzv_backend=c10d --rdzv_endpoint=localhost:29552 \
  train.py --config_path "$RT" \
  --logdir "$LOGDIR" --disable-wandb \
  >> "$LOGDIR/train_from350.log" 2>&1
