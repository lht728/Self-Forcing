#!/usr/bin/env bash
# 等 1400 checkpoint 写完整 -> 停当前 ODE 训练 -> 关 cpu_offload + resume 1400 -> 相同方式重启续训。
# 由 setsid nohup 后台拉起。CFG(guidance_scale=6.0) 与 tensorboard 均沿用 config, 无需额外开关。
set -u

ROOT=/apdcephfs/private_huitinglu/Self-Forcing
PY=/apdcephfs/private_huitinglu/conda_envs/bernini_distill/bin/python
CFG=$ROOT/configs/self_forcing_v2v_ode_routeb.yaml
CKPT_DIR=$ROOT/logs/bernini_v2v_ode_routeb/checkpoint_model_001400
CKPT_PT=$CKPT_DIR/model.pt
ODE_LOGDIR=$ROOT/logs/bernini_v2v_ode_routeb
TRAIN_LOG=$ROOT/logs/bernini_v2v_ode_routeb_train.log
WLOG=$ROOT/logs/routeb_restart_1400.log

cd "$ROOT" || exit 1
export PATH=/apdcephfs/private_huitinglu/conda_envs/bernini_distill/bin:$PATH

log(){ echo "[$(date '+%F %T')] $*" >> "$WLOG"; }

log "watcher 启动, 等待 1400 checkpoint ($CKPT_PT) ..."

# 1) 等 1400 model.pt 出现
while [ ! -f "$CKPT_PT" ]; do sleep 60; done
log "检测到 $CKPT_PT, 等待写入完成(大小稳定)..."

# 2) 等大小连续 2 次稳定, 确保写盘完成
prev=-1
while true; do
  cur=$(stat -c %s "$CKPT_PT" 2>/dev/null || echo 0)
  if [ "$cur" = "$prev" ] && [ "$cur" -gt 0 ]; then break; fi
  prev=$cur
  sleep 30
done
log "1400 checkpoint 写入完成, size=$prev bytes"
sleep 30   # 余量, 避免与训练本步落盘竞争

# 3) 停止当前 ODE 训练 (launcher + 8 workers)
log "停止当前 ODE 训练进程..."
pkill -f "torch.distributed.run.*nproc_per_node=8" 2>/dev/null
pkill -f "train.py --config_path .*self_forcing_v2v_ode_routeb" 2>/dev/null
for i in $(seq 1 36); do
  if pgrep -f "train.py --config_path .*self_forcing_v2v_ode_routeb" >/dev/null 2>&1; then
    sleep 5
  else
    break
  fi
done
pkill -9 -f "train.py --config_path .*self_forcing_v2v_ode_routeb" 2>/dev/null
sleep 15   # 等 NCCL/端口 29542 释放
log "旧训练已停止。"

# 4) 改 config: 关 cpu_offload(提速), resume 1400
sed -i "s|^resume_ckpt:.*|resume_ckpt: logs/bernini_v2v_ode_routeb/checkpoint_model_001400/model.pt|" "$CFG"
sed -i "s|^ckpt_step:.*|ckpt_step: 1400|" "$CFG"
sed -i "s|^generator_cpu_offload:.*|generator_cpu_offload: false  # 95GB 卡余量充足, 关闭 offload 去掉 CPU<->GPU 搬运以提速|" "$CFG"
log "config 已更新: generator_cpu_offload=false, resume_ckpt=1400, ckpt_step=1400"

# 4.5) 方案②本地缓存: resume_ckpt 同步到节点本地盘 /dockerdata, 消除重启时 8 rank 对 cephfs 的并发读。
CACHE=/dockerdata/ckpt_cache
RESUME_DST=$CACHE/${CKPT_PT#/}
mkdir -p "$(dirname "$RESUME_DST")"
if [ -f "$RESUME_DST" ] && [ "$(stat -c%s "$CKPT_PT")" = "$(stat -c%s "$RESUME_DST")" ]; then
  log "本地缓存命中(大小一致), 跳过同步: $RESUME_DST"
else
  log "同步 resume_ckpt 到本地盘: $CKPT_PT -> $RESUME_DST"
  cp -f "$CKPT_PT" "$RESUME_DST.tmp" && mv -f "$RESUME_DST.tmp" "$RESUME_DST"
  log "本地缓存完成: $RESUME_DST"
fi
RT=$CACHE/self_forcing_v2v_ode_routeb.runtime.yaml
sed -E "s|^resume_ckpt:.*|resume_ckpt: $RESUME_DST|" "$CFG" > "$RT"
log "runtime config: $RT (resume_ckpt 指向本地缓存)"

# 5) 相同方式重启 (8 卡 torchrun, 后台; 从本地缓存加载, cephfs 零读)
echo "" >> "$TRAIN_LOG"
echo "==================== RESTART @ $(date '+%F %T') : resume 1400(local-cache), cpu_offload=OFF ====================" >> "$TRAIN_LOG"
log "重启 ODE 训练 (8 卡, 关 offload, 从 1400 本地缓存续训)..."
nohup "$PY" -m torch.distributed.run --nnodes=1 --nproc_per_node=8 \
  --rdzv_id=8844 --rdzv_backend=c10d --rdzv_endpoint=localhost:29542 \
  train.py --config_path "$RT" \
  --logdir "$ODE_LOGDIR" --disable-wandb \
  >> "$TRAIN_LOG" 2>&1 &
NEWPID=$!
log "重启已拉起, launcher pid=$NEWPID"
