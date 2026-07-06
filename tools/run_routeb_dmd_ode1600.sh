#!/usr/bin/env bash
# Stage-2 DMD 蒸馏: 从 ODE routeb ckpt1600 初始化 generator, 8 卡起训。
# CFG(guidance_scale=3.0) + tensorboard 已在 config 内开启。
# 注意: 本脚本仅供手动执行, 默认不被自动拉起。
set -u

ROOT=/apdcephfs/private_huitinglu/Self-Forcing
PY=/apdcephfs/private_huitinglu/conda_envs/bernini_distill/bin/python
CONFIG=$ROOT/configs/self_forcing_v2v_dmd_routeb_ode1600.yaml
LOGDIR=$ROOT/logs/bernini_v2v_dmd_routeb_ode1600

cd "$ROOT" || exit 1
export PATH=/apdcephfs/private_huitinglu/conda_envs/bernini_distill/bin:$PATH
mkdir -p "$LOGDIR"

"$PY" -m torch.distributed.run --nnodes=1 --nproc_per_node=8 \
  --rdzv_id=7751 --rdzv_backend=c10d --rdzv_endpoint=localhost:29551 \
  train.py --config_path "$CONFIG" \
  --logdir "$LOGDIR" --disable-wandb \
  >> "$LOGDIR/train.log" 2>&1
echo "[$(date '+%F %T')] DMD(ode1600) 训练进程结束(退出码=$?)。"
