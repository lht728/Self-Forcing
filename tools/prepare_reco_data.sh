#!/bin/bash
# 补齐 ReCo-Data: 拉取(git-lfs)指定任务的 json + 分片 tar, 并就地解压成 mp4。
# json 的 src_video/tar_video 字段引用 "<task>/src_videos/<hash>.mp4", 解压后即可被 dataloader 读取。
#
# 用法:
#   TASKS="replace" bash tools/prepare_reco_data.sh             # 仅补齐 replace
#   TASKS="add remove replace style" bash tools/prepare_reco_data.sh   # 全量(需 >1.3T 磁盘)
#   ONLY_EXTRACT=1 TASKS="replace" bash tools/prepare_reco_data.sh     # 不联网, 只解压已下载的 tar
set -e

RECO_DIR=${RECO_DIR:-/apdcephfs/private_huitinglu/ReCo-Data}
TASKS=${TASKS:-"replace"}
ONLY_EXTRACT=${ONLY_EXTRACT:-0}
# 默认走 huggingface.co(本机经公司代理可达); hf-mirror 经代理不通, 如需镜像自行 export HF_ENDPOINT
HF_BIN=${HF_BIN:-hf}
REPO=HiDream-ai/ReCo-Data

cd "$RECO_DIR"

if [ "$ONLY_EXTRACT" != "1" ]; then
  for t in $TASKS; do
    echo "== hf 下载 $t (json + src_videos + tar_videos) =="
    $HF_BIN download "$REPO" --repo-type dataset \
      --include "${t}/**" "${t}_data_configs.json" \
      --local-dir "$RECO_DIR" ${HF_TOKEN:+--token "$HF_TOKEN"}
  done
fi

for t in $TASKS; do
  for sub in src_videos tar_videos; do
    d="$RECO_DIR/$t/$sub"
    [ -d "$d" ] || continue
    shopt -s nullglob
    for tf in "$d"/*.tar; do
      marker="$tf.extracted"
      [ -e "$marker" ] && continue
      echo "解压 $tf"
      tar xf "$tf" -C "$d" && touch "$marker"
    done
    shopt -u nullglob
  done
  cfg="$RECO_DIR/$t/${t}_data_configs.json"
  [ -e "$cfg" ] && echo "  配置就绪: $cfg" || echo "  warn: 缺少 $cfg (git-lfs 未拉取?)"
done
echo "ReCo-Data 补齐完成: TASKS=[$TASKS]"
