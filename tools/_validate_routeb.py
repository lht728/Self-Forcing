"""方案B 自检: (1) prefix v2v 权重 key 匹配; (2) 双向前缀 forward; (3) 因果源前缀 prefill + 目标 forward。"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wan.modules.model import WanModel
from wan.modules.causal_model import CausalWanModel

CKPT = "checkpoints/bernini_v2v_prefix_init.pt"


def check_weight_keys():
    sd = torch.load(CKPT, map_location="cpu")
    sd = sd.get("generator", sd)
    for cls, name in [(WanModel, "WanModel(双向)"), (CausalWanModel, "CausalWanModel(因果)")]:
        m = cls(model_type="t2v", in_dim=16, dim=1536, ffn_dim=8960, freq_dim=256,
                out_dim=16, num_heads=12, num_layers=30, text_len=512, eps=1e-6)
        missing, unexpected = m.load_state_dict(sd, strict=False)
        missing = [x for x in missing if not x.endswith(".freqs") and not x.endswith(".visual_id_freqs")]
        assert not missing, f"{name} 缺失键: {missing[:8]}"
        assert not unexpected, f"{name} 多余键: {unexpected[:8]}"
        print(f"[OK] {name} prefix v2v 权重严格匹配 (忽略 freqs/visual_id_freqs)")
        del m


def small_cfg(dim=128, heads=2, layers=2):
    return dict(model_type="t2v", in_dim=16, dim=dim, ffn_dim=256, freq_dim=256,
                out_dim=16, num_heads=heads, num_layers=layers, text_len=512, eps=1e-6)


@torch.no_grad()
def check_bidirectional_prefix(device="cuda"):
    m = WanModel(**small_cfg()).to(device, torch.bfloat16).eval()
    B, C, F, H, W = 1, 16, 4, 30, 52
    noisy = torch.randn(B, C, F, H, W, device=device, dtype=torch.bfloat16)
    src = torch.randn(B, C, F, H, W, device=device, dtype=torch.bfloat16)
    t = torch.tensor([500] * B, device=device)
    ctx = [torch.randn(20, 4096, device=device, dtype=torch.bfloat16)]
    out = m(noisy, t=t, context=ctx, seq_len=32760, cond_latents=src)
    assert out.shape == (B, C, F, H, W), out.shape
    print(f"[OK] 双向前缀 forward 输出形状 {tuple(out.shape)} (只取目标, 与输入目标一致)")


@torch.no_grad()
def check_causal_prefill(device="cuda"):
    m = CausalWanModel(**small_cfg()).to(device, torch.bfloat16).eval()
    m.num_frame_per_block = 2
    B, C, Fs, H, W = 1, 16, 4, 30, 52
    frame_seqlen = (H // 2) * (W // 2)
    heads, hd = 2, 64
    cache_frames = Fs + 4
    kv_cache = [{
        "k": torch.zeros(B, cache_frames * frame_seqlen, heads, hd, device=device, dtype=torch.bfloat16),
        "v": torch.zeros(B, cache_frames * frame_seqlen, heads, hd, device=device, dtype=torch.bfloat16),
        "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
        "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
    } for _ in range(2)]
    crossattn_cache = [{
        "k": torch.zeros(B, 512, heads, hd, device=device, dtype=torch.bfloat16),
        "v": torch.zeros(B, 512, heads, hd, device=device, dtype=torch.bfloat16),
        "is_init": False,
    } for _ in range(2)]
    ctx = [torch.randn(20, 4096, device=device, dtype=torch.bfloat16)]

    # 源前缀 prefill: source_id=1, current_start=0
    m.num_source_frames = Fs
    src = torch.randn(B, C, Fs, H, W, device=device, dtype=torch.bfloat16)
    t_src = torch.zeros(B, Fs, device=device, dtype=torch.long)
    m(src, t=t_src, context=ctx, seq_len=32760, kv_cache=kv_cache,
      crossattn_cache=crossattn_cache, current_start=0, cache_start=0, source_id=1)
    ge = kv_cache[0]["global_end_index"].item()
    assert ge == Fs * frame_seqlen, ge
    print(f"[OK] 因果源前缀 prefill: global_end={ge} (= {Fs}帧 * {frame_seqlen})")

    # 目标 block forward: source_id=0, current_start=(Fs+0)*frame_seqlen
    tgt = torch.randn(B, C, 2, H, W, device=device, dtype=torch.bfloat16)
    t_tgt = torch.ones(B, 2, device=device, dtype=torch.long) * 500
    out = m(tgt, t=t_tgt, context=ctx, seq_len=32760, kv_cache=kv_cache,
            crossattn_cache=crossattn_cache, current_start=Fs * frame_seqlen,
            cache_start=Fs * frame_seqlen, source_id=0)
    assert out.shape == (B, C, 2, H, W), out.shape
    ge2 = kv_cache[0]["global_end_index"].item()
    assert ge2 == (Fs + 2) * frame_seqlen, ge2
    print(f"[OK] 因果目标 forward 输出 {tuple(out.shape)}, global_end={ge2} (源+目标连续写入)")


def check_causal_prefix_train(device="cuda"):
    """因果 _forward_train 源前缀注入: 前向输出形状 + 反向梯度可回传。"""
    m = CausalWanModel(**small_cfg()).to(device, torch.bfloat16).train()
    m.num_frame_per_block = 2
    B, C, Fs, Ft, H, W = 1, 16, 4, 4, 30, 52
    src = torch.randn(B, C, Fs, H, W, device=device, dtype=torch.bfloat16)
    tgt = torch.randn(B, C, Ft, H, W, device=device, dtype=torch.bfloat16, requires_grad=True)
    t = torch.full((B, Ft), 500, device=device, dtype=torch.long)
    ctx = [torch.randn(20, 4096, device=device, dtype=torch.bfloat16)]
    out = m(tgt, t=t, context=ctx, seq_len=32760, cond_latents=src)
    assert out.shape == (B, C, Ft, H, W), out.shape
    loss = out.float().pow(2).mean()
    loss.backward()
    g = m.patch_embedding.weight.grad
    assert g is not None and torch.isfinite(g).all(), "patch_embedding 无有效梯度"
    print(f"[OK] 因果训练前缀 forward 输出 {tuple(out.shape)} + 反向梯度有效(loss={loss.item():.4f})")


if __name__ == "__main__":
    check_weight_keys()
    if torch.cuda.is_available():
        check_bidirectional_prefix()
        check_causal_prefill()
        check_causal_prefix_train()
        print("\n方案B 全部自检通过 ✅")
    else:
        print("无 GPU, 跳过 forward 自检")
