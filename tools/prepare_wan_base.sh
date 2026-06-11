#!/bin/bash
# 准备 Self-Forcing 全套 wrapper 硬编码依赖的基座: wan_models/Wan2.1-T2V-1.3B
# 含 DiT(config.json + safetensors)、VAE(Wan2.1_VAE.pth)、UMT5 文本编码器与 tokenizer。
# Bernini-R-1.3B 由 Wan2.1-1.3B 微调而来, VAE/UMT5 与之一致, 故直接用官方 Wan2.1-T2V-1.3B。
set -e

# 默认走 huggingface.co(本机经公司代理可达); 如需镜像可自行 export HF_ENDPOINT
HF_BIN=${HF_BIN:-hf}
DEST=${1:-wan_models/Wan2.1-T2V-1.3B}
REPO=Wan-AI/Wan2.1-T2V-1.3B

mkdir -p "$DEST"
echo "下载 $REPO -> $DEST (endpoint=${HF_ENDPOINT:-https://huggingface.co})"
$HF_BIN download "$REPO" --local-dir "$DEST" ${HF_TOKEN:+--token "$HF_TOKEN"}

echo "校验关键文件..."
miss=0
for f in models_t5_umt5-xxl-enc-bf16.pth Wan2.1_VAE.pth config.json diffusion_pytorch_model.safetensors; do
  if [ ! -e "$DEST/$f" ]; then echo "  缺少: $f"; miss=1; fi
done
if [ ! -d "$DEST/google/umt5-xxl" ]; then echo "  缺少 tokenizer 目录: google/umt5-xxl"; miss=1; fi
if [ "$miss" = "1" ]; then echo "准备未完成, 请检查下载"; exit 1; fi
echo "Wan2.1-T2V-1.3B 准备完成: $DEST"
