#!/usr/bin/env bash
# 补齐 ReCo replace/tar_videos：hf download 分片 + tar 解压，与 src 分片对齐。
# 不占 GPU；与 ODE 采样可并行。
#
# 用法:
#   bash tools/sync_reco_tar_videos.sh
#   ONLY_EXTRACT=1 bash tools/sync_reco_tar_videos.sh
#   MAX_SHARD=37 bash tools/sync_reco_tar_videos.sh   # 手动指定分片上限
set -euo pipefail

RECO_DIR=${RECO_DIR:-/apdcephfs/private_huitinglu/ReCo-Data}
TASK=${TASK:-replace}
LOG=${LOG:-$RECO_DIR/sync_tar_${TASK}.log}
ONLY_EXTRACT=${ONLY_EXTRACT:-0}
HF_BIN=${HF_BIN:-/apdcephfs/private_huitinglu/conda_envs/bernini_distill/bin/hf}
REPO=HiDream-ai/ReCo-Data
TOTAL_SHARDS=${TOTAL_SHARDS:-79}   # replace: 00-78

mkdir -p "$(dirname "$LOG")"
exec >> "$LOG" 2>&1
echo "[$(date '+%F %T')] sync_reco_tar_videos start ONLY_EXTRACT=$ONLY_EXTRACT"

SRC_DIR="$RECO_DIR/$TASK/src_videos"
TAR_DIR="$RECO_DIR/$TASK/tar_videos"
mkdir -p "$TAR_DIR"

# 用逐文件 -f 检测 src 已解压分片(避免 ceph glob 卡住)
if [ -n "${MAX_SHARD:-}" ]; then
  MAX_IDX=$MAX_SHARD
else
  MAX_IDX=-1
  for i in $(seq 0 $((TOTAL_SHARDS - 1))); do
    idx=$(printf '%02d' "$i")
    if [ -f "$SRC_DIR/${TASK}_src_video_${idx}.tar.extracted" ]; then
      MAX_IDX=$i
    fi
  done
  [ "$MAX_IDX" -lt 0 ] && MAX_IDX=$((TOTAL_SHARDS - 1))
fi
echo "tar 补齐分片 0-$MAX_IDX (TOTAL_SHARDS=$TOTAL_SHARDS)"

extract_tar() {
  local tf="$1"
  local marker="${tf}.extracted"
  [ -e "$marker" ] && return 0
  [ -f "$tf" ] || return 1
  echo "[$(date '+%F %T')] 解压 $(basename "$tf") ..."
  tar xf "$tf" -C "$(dirname "$tf")"
  touch "$marker"
}

download_tar_shard() {
  local idx="$1"
  local name="${TASK}_tar_video_$(printf '%02d' "$idx").tar"
  local rel="$TASK/tar_videos/$name"
  local dest="$TAR_DIR/$name"
  if [ -f "$dest" ]; then
    echo "已存在 $name，跳过下载"
    return 0
  fi
  echo "[$(date '+%F %T')] hf download $rel ..."
  "$HF_BIN" download "$REPO" --repo-type dataset \
    --include "$rel" --local-dir "$RECO_DIR" ${HF_TOKEN:+--token "$HF_TOKEN"}
}

if [ "$ONLY_EXTRACT" != "1" ]; then
  for i in $(seq 0 "$MAX_IDX"); do
    download_tar_shard "$i" || echo "warn: 下载分片 $i 失败，继续"
  done
fi

for i in $(seq 0 "$MAX_IDX"); do
  tf="$TAR_DIR/${TASK}_tar_video_$(printf '%02d' "$i").tar"
  extract_tar "$tf" || echo "warn: 解压分片 $i 失败，继续"
done

CACHE_DIR="/apdcephfs/private_huitinglu/Self-Forcing/.cache/v2v_existence"
if [ -d "$CACHE_DIR" ]; then
  echo "[$(date '+%F %T')] 清除 V2V 存在性缓存"
  rm -f "$CACHE_DIR"/*.pkl 2>/dev/null || true
fi

python3 - <<PY
import json, os, random
base = "$RECO_DIR"
with open(f"{base}/replace/replace_data_configs.json") as f:
    items = [d for d in json.load(f) if d.get("src_video") and d.get("tar_video")]
random.seed(0)
sample = random.sample(items, min(500, len(items)))
src_ok = tar_ok = both = 0
for d in sample:
    sp, tp = os.path.join(base, d["src_video"]), os.path.join(base, d["tar_video"])
    se, te = os.path.isfile(sp), os.path.isfile(tp)
    src_ok += se; tar_ok += te; both += se and te
print(f"[统计] 抽样{len(sample)}: src={src_ok} tar={tar_ok} both={both} ({100*both/max(1,len(sample)):.1f}%)")
PY

echo "[$(date '+%F %T')] sync_reco_tar_videos done (0-$MAX_IDX)"
