#!/usr/bin/env bash
# 路线B Causal ODE 流水线 (Causal Forcing): 因果 teacher 采样 -> build LMDB -> ODE 回归 -> (手动) DMD
set -u

ROOT=/apdcephfs/private_huitinglu/Self-Forcing
PY=/apdcephfs/private_huitinglu/conda_envs/bernini_distill/bin/python
CAUSAL_TEACHER=$ROOT/checkpoints/bernini_causal_v2v_teacher.pt
SHARD_DIR=$ROOT/ode_data/v2v_routeb_causal_shards
LMDB_PATH=$ROOT/ode_data/bernini_v2v_causal_ode_routeb_lmdb
ODE_LOGDIR=$ROOT/logs/bernini_v2v_causal_ode_routeb
MIN_SHARDS=1900
LOG=$ROOT/logs/routeb_causal_ode_pipeline.log

cd "$ROOT" || exit 1
export PATH=/apdcephfs/private_huitinglu/conda_envs/bernini_distill/bin:$PATH

if [ ! -f "$CAUSAL_TEACHER" ]; then
  echo "[$(date '+%F %T')] 缺少 $CAUSAL_TEACHER, 请先运行 tools/train_causal_v2v_teacher.py" >> "$LOG"
  exit 1
fi

echo "[$(date '+%F %T')] 等待因果 ODE 采样进程..." >> "$LOG"
while pgrep -f "generate_v2v_ode_lmdb.py sample.*causal_teacher" >/dev/null 2>&1; do
  sleep 300
done

CNT=$(ls "$SHARD_DIR"/*.pt 2>/dev/null | wc -l)
echo "[$(date '+%F %T')] 采样结束, 分片数=$CNT" >> "$LOG"
if [ "$CNT" -lt "$MIN_SHARDS" ]; then
  echo "[$(date '+%F %T')] 分片 $CNT < $MIN_SHARDS, 中止" >> "$LOG"
  exit 1
fi

"$PY" tools/generate_v2v_ode_lmdb.py build \
  --shard_dir "$SHARD_DIR" --lmdb_path "$LMDB_PATH" >> "$LOG" 2>&1

mkdir -p "$ODE_LOGDIR"
"$PY" -m torch.distributed.run --nnodes=1 --nproc_per_node=8 \
  --rdzv_id=7743 --rdzv_backend=c10d --rdzv_endpoint=localhost:29543 \
  train.py --config_path configs/self_forcing_v2v_causal_ode_routeb.yaml \
  --logdir "$ODE_LOGDIR" --disable-wandb \
  >> "$ODE_LOGDIR/train.log" 2>&1
echo "[$(date '+%F %T')] Causal ODE 回归完成 exit=$?" >> "$LOG"
