"""Bernini-R-1.3B-Diffusers(WanTransformer3DModel) -> Self-Forcing WanModel 权重转换。

两件事:
  1) diffusers 命名 -> Self-Forcing WanModel 命名 (二者同源 Wan2.1-1.3B, 仅 key 不同)。
  2) patch_embedding 由 16 通道扩为 16+cond_channels 通道(默认 32), 新增通道零初始化,
     得到“通道拼接 v2v teacher”的初始权重; 之后由短训填充新增通道。

输出 state_dict 以 {"generator": sd} 形式保存, 可被 Self-Forcing 的加载逻辑直接使用
(real_score / fake_score / generator 共用同一套 WanModel 键名)。
"""
import argparse
import glob
import os

import torch
from safetensors.torch import load_file


def _block_subkey_map():
    return {
        "attn1.to_q.weight": "self_attn.q.weight", "attn1.to_q.bias": "self_attn.q.bias",
        "attn1.to_k.weight": "self_attn.k.weight", "attn1.to_k.bias": "self_attn.k.bias",
        "attn1.to_v.weight": "self_attn.v.weight", "attn1.to_v.bias": "self_attn.v.bias",
        "attn1.to_out.0.weight": "self_attn.o.weight", "attn1.to_out.0.bias": "self_attn.o.bias",
        "attn1.norm_q.weight": "self_attn.norm_q.weight", "attn1.norm_k.weight": "self_attn.norm_k.weight",
        "attn2.to_q.weight": "cross_attn.q.weight", "attn2.to_q.bias": "cross_attn.q.bias",
        "attn2.to_k.weight": "cross_attn.k.weight", "attn2.to_k.bias": "cross_attn.k.bias",
        "attn2.to_v.weight": "cross_attn.v.weight", "attn2.to_v.bias": "cross_attn.v.bias",
        "attn2.to_out.0.weight": "cross_attn.o.weight", "attn2.to_out.0.bias": "cross_attn.o.bias",
        "attn2.norm_q.weight": "cross_attn.norm_q.weight", "attn2.norm_k.weight": "cross_attn.norm_k.weight",
        # diffusers 块内唯一带参 norm(norm2, affine) = Self-Forcing 的 cross-attn norm(norm3)
        "norm2.weight": "norm3.weight", "norm2.bias": "norm3.bias",
        "ffn.net.0.proj.weight": "ffn.0.weight", "ffn.net.0.proj.bias": "ffn.0.bias",
        "ffn.net.2.weight": "ffn.2.weight", "ffn.net.2.bias": "ffn.2.bias",
        "scale_shift_table": "modulation",
    }


def _top_map():
    return {
        "patch_embedding.weight": "patch_embedding.weight",
        "patch_embedding.bias": "patch_embedding.bias",
        "proj_out.weight": "head.head.weight",
        "proj_out.bias": "head.head.bias",
        "scale_shift_table": "head.modulation",
        "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
        "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
        "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
        "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
        "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
        "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
        "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
        "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
        "condition_embedder.time_proj.weight": "time_projection.1.weight",
        "condition_embedder.time_proj.bias": "time_projection.1.bias",
    }


def convert_key(k, top_map, sub_map):
    if k in top_map:
        return top_map[k]
    if k.startswith("blocks."):
        parts = k.split(".")
        idx = parts[1]
        rest = ".".join(parts[2:])
        if rest in sub_map:
            return f"blocks.{idx}.{sub_map[rest]}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/apdcephfs/private_huitinglu/models/Bernini-R-1.3B-Diffusers/transformer")
    ap.add_argument("--out", default="checkpoints/bernini_wanmodel_v2v_init.pt")
    ap.add_argument("--cond_channels", type=int, default=16)
    ap.add_argument("--no_expand", action="store_true", help="只转命名, 不扩通道(用于纯 t2v base)")
    ap.add_argument("--validate", action="store_true", help="实例化 WanModel 做 strict 加载校验(需 GPU/flash-attn 环境)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.src, "*.safetensors")))
    assert files, f"未找到 safetensors: {args.src}"
    src_sd = {}
    for f in files:
        src_sd.update(load_file(f))
    print(f"读取源权重: {len(src_sd)} keys")

    top_map, sub_map = _top_map(), _block_subkey_map()
    out_sd = {}
    unmapped = []
    for k, v in src_sd.items():
        nk = convert_key(k, top_map, sub_map)
        if nk is None:
            unmapped.append(k)
            continue
        assert nk not in out_sd, f"目标键冲突: {nk} (来自 {k})"
        out_sd[nk] = v.clone()

    assert not unmapped, f"存在未映射的源键: {unmapped[:10]} ... 共 {len(unmapped)}"
    print(f"映射完成: {len(out_sd)} keys")

    # 扩通道: patch_embedding.weight [dim,16,1,2,2] -> [dim,16+cond,1,2,2], 新增通道零初始化
    if not args.no_expand:
        w = out_sd["patch_embedding.weight"]
        dim, old_in = w.shape[0], w.shape[1]
        total_in = old_in + args.cond_channels
        new_w = torch.zeros(dim, total_in, *w.shape[2:], dtype=w.dtype)
        new_w[:, :old_in] = w
        out_sd["patch_embedding.weight"] = new_w
        print(f"patch_embedding 扩通道: {old_in} -> {total_in} (新增 {args.cond_channels} 零初始化)")

    if args.validate:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from wan.modules.model import WanModel
        in_dim = 16 + (0 if args.no_expand else args.cond_channels)
        model = WanModel(model_type="t2v", in_dim=in_dim, dim=1536, ffn_dim=8960,
                         freq_dim=256, out_dim=16, num_heads=12, num_layers=30,
                         text_len=512, eps=1e-6)
        missing, unexpected = model.load_state_dict(out_sd, strict=False)
        missing = [m for m in missing if not m.endswith(".freqs")]
        assert not missing, f"缺失键: {missing[:20]}"
        assert not unexpected, f"多余键: {unexpected[:20]}"
        print("strict 加载校验通过")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"generator": out_sd}, args.out)
    print(f"已保存: {args.out}")


if __name__ == "__main__":
    main()
