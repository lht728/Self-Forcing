#!/usr/bin/env bash
# 路线B ODE 流水线编排: 等采样结束 -> build LMDB -> 起 ODE 回归(8卡)。
# 由 setsid nohup 后台拉起; 采样进程仍是独立进程, 本脚本只等待它退出。
set -u

ROOT=/apdcephfs/private_huitinglu/Self-Forcing
PY=/apdcephfs/private_huitinglu/conda_envs/bernini_distill/bin/python
SHARD_DIR=$ROOT/ode_data/v2v_routeb_shards
LMDB_PATH=$ROOT/ode_data/bernini_v2v_ode_routeb_lmdb
ODE_LOGDIR=$ROOT/logs/bernini_v2v_ode_routeb
MIN_SHARDS=1900   # 采样健康下限(目标2000), 低于此视为异常退出
LOG=$ROOT/logs/routeb_ode_pipeline.log

cd "$ROOT" || exit 1
export PATH=/apdcephfs/private_huitinglu/conda_envs/bernini_distill/bin:$PATH

echo "[$(date '+%F %T')] 编排启动, 等待采样进程退出..." >> "$LOG"

# 1) 等采样进程退出(轮询)
while pgrep -f "generate_v2v_ode_lmdb.py sample" >/dev/null 2>&1; do
  sleep 300
done

CNT=$(ls "$SHARD_DIR"/*.pt 2>/dev/null | wc -l)
echo "[$(date '+%F %T')] 采样进程已退出, 分片数=$CNT" >> "$LOG"
if [ "$CNT" -lt "$MIN_SHARDS" ]; then
  echo "[$(date '+%F %T')] 分片数 $CNT < $MIN_SHARDS, 疑似异常退出, 中止流水线(不 build/不训练)。" >> "$LOG"
  exit 1
fi

# 2) build LMDB
echo "[$(date '+%F %T')] 开始 build LMDB -> $LMDB_PATH" >> "$LOG"
"$PY" tools/generate_v2v_ode_lmdb.py build \
  --shard_dir "$SHARD_DIR" --lmdb_path "$LMDB_PATH" >> "$LOG" 2>&1
if [ ! -d "$LMDB_PATH" ]; then
  echo "[$(date '+%F %T')] build 失败, LMDB 目录不存在, 中止。" >> "$LOG"
  exit 1
fi
echo "[$(date '+%F %T')] build 完成。" >> "$LOG"

# 3) 起 ODE 回归(8卡)
mkdir -p "$ODE_LOGDIR"
echo "[$(date '+%F %T')] 启动 ODE 回归训练 -> $ODE_LOGDIR" >> "$LOG"
"$PY" -m torch.distributed.run --nnodes=1 --nproc_per_node=8 \
  --rdzv_id=7742 --rdzv_backend=c10d --rdzv_endpoint=localhost:29542 \
  train.py --config_path configs/self_forcing_v2v_ode_routeb.yaml \
  --logdir "$ODE_LOGDIR" --disable-wandb \
  >> "$ODE_LOGDIR/train.log" 2>&1
echo "[$(date '+%F %T')] ODE 回归训练进程结束(退出码=$?)。" >> "$LOG"
