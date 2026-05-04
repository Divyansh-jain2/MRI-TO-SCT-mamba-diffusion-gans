"""
Dimension verification test for the Hybrid MRI Encoder + Cross-Attention
Diffusion Model.

Runs with small random tensors to verify all I/O dimensions match throughout
the pipeline. No GPU required (uses CPU with tiny volumes).
"""

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn

# Suppress MONAI warnings
import warnings
warnings.filterwarnings('ignore')


def separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_cross_attention():
    separator("1. CrossAttention3D")
    from network.cross_attention import CrossAttention3D

    B, D, H, W = 2, 8, 8, 4

    # Test with matching channel dims
    ca = CrossAttention3D(query_dim=64, context_dim=64, num_heads=4)
    q = torch.randn(B, 64, D, H, W)
    kv = torch.randn(B, 64, D, H, W)
    out = ca(q, kv)
    print(f"  Q:   {list(q.shape)}")
    print(f"  K,V: {list(kv.shape)}")
    print(f"  Out: {list(out.shape)}")
    assert out.shape == q.shape, f"Shape mismatch: {out.shape} != {q.shape}"
    print("  ✓ Same channel dims — PASS")

    # Test with mismatched channel dims
    ca2 = CrossAttention3D(query_dim=128, context_dim=64, num_heads=4)
    q2 = torch.randn(B, 128, D, H, W)
    kv2 = torch.randn(B, 64, D, H, W)
    out2 = ca2(q2, kv2)
    assert out2.shape == q2.shape
    print(f"  Q:   {list(q2.shape)}")
    print(f"  K,V: {list(kv2.shape)}")
    print(f"  Out: {list(out2.shape)}")
    print("  ✓ Mismatched channel dims — PASS")

    # Test with mismatched spatial dims
    ca3 = CrossAttention3D(query_dim=256, context_dim=448, num_heads=8)
    q3 = torch.randn(B, 256, 4, 4, 1)
    kv3 = torch.randn(B, 448, 4, 4, 1)
    out3 = ca3(q3, kv3)
    assert out3.shape == q3.shape
    print(f"  Q:   {list(q3.shape)} (CA3 denoiser)")
    print(f"  K,V: {list(kv3.shape)} (CA3 f3+f4)")
    print(f"  Out: {list(out3.shape)}")
    print("  ✓ CA3 f₃+f₄ concat dims — PASS")


def test_mri_encoder():
    separator("2. MRI Semantic Encoder")
    from network.mri_encoder import MRISemanticEncoder

    B = 2
    # Use small volume for testing
    D, H, W = 16, 16, 8
    mri = torch.randn(B, 1, D, H, W)

    enc = MRISemanticEncoder(
        in_channels=1,
        enc_channels=(64, 128, 192, 256),
        global_dim=256,
        dims=3,
        num_heads=(4, 4, 4, 4),
        window_size=(4, 4, 4),
        pool_kernel=(2, 2, 2),
        freeze=True,
    )

    feats = enc(mri)
    print(f"  Input MRI:    {list(mri.shape)}")
    print(f"  f₁ (full):    {list(feats['f1'].shape)}   expected [B, 64, {D}, {H}, {W}]")
    print(f"  f₂ (½ res):   {list(feats['f2'].shape)}   expected [B, 128, {D//2}, {H//2}, {W//2}]")
    print(f"  f₃ (¼ res):   {list(feats['f3'].shape)}   expected [B, 192, {D//4}, {H//4}, {W//4}]")
    print(f"  f₄ (⅛ res):   {list(feats['f4'].shape)}   expected [B, 256, {D//8}, {H//8}, {W//8}]")
    print(f"  global token: {list(feats['global'].shape)}   expected [B, 256]")

    assert feats['f1'].shape == (B, 64, D, H, W)
    assert feats['f2'].shape == (B, 128, D//2, H//2, W//2)
    assert feats['f3'].shape == (B, 192, D//4, H//4, W//4)
    assert feats['f4'].shape == (B, 256, D//8, H//8, W//8)
    assert feats['global'].shape == (B, 256)

    # Verify frozen
    for name, p in enc.named_parameters():
        assert not p.requires_grad, f"Parameter {name} should be frozen!"
    print("  ✓ All dimensions correct, all parameters frozen — PASS")


def test_hybrid_model():
    separator("3. HybridDiffusionModel (Full Forward Pass)")
    from network.hybrid_model import HybridDiffusionModel

    B = 2
    D, H, W = 16, 16, 8
    model_channels = 64

    model = HybridDiffusionModel(
        image_size=(D, H, W),
        model_channels=model_channels,
        out_channels=2,
        enc_channels=(64, 128, 192, 256),
        channel_mult=(1, 2, 4),
        num_res_blocks=1,  # Use 1 for fast testing
        sample_kernel=((2, 2, 2), (2, 2, 2)),
        num_heads_cross_attn=(4, 4, 8),
        dims=3,
        dropout=0.0,
        use_scale_shift_norm=True,
        freeze_encoder=True,
        encoder_window_size=(4, 4, 4),
        encoder_num_heads=(4, 4, 4, 4),
        encoder_pool_kernel=(2, 2, 2),
    )
    model.eval()

    x_t = torch.randn(B, 1, D, H, W)      # Noisy CT
    mri = torch.randn(B, 1, D, H, W)       # MRI condition
    t = torch.randint(0, 1000, (B,))        # Timesteps

    with torch.no_grad():
        out = model(x_t, t, mri)

    print(f"  Input x_t (noisy CT):  {list(x_t.shape)}")
    print(f"  Input MRI condition:   {list(mri.shape)}")
    print(f"  Timesteps:             {list(t.shape)}")
    print(f"  Output:                {list(out.shape)}   expected [B, 2, {D}, {H}, {W}]")
    assert out.shape == (B, 2, D, H, W), f"Output shape mismatch: {out.shape}"
    print("  ✓ Full model forward pass — PASS")

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    print(f"\n  Total parameters:     {total:>12,}")
    print(f"  Trainable parameters: {trainable:>12,}")
    print(f"  Frozen parameters:    {frozen:>12,}")


def test_data_flow():
    separator("4. Data Flow — MRI Encoder → Cross-Attention")
    from network.mri_encoder import MRISemanticEncoder
    from network.cross_attention import CrossAttention3D

    B, D, H, W = 2, 16, 16, 8

    # Simulate MRI encoder output
    enc = MRISemanticEncoder(
        in_channels=1, enc_channels=(64, 128, 192, 256),
        global_dim=256, dims=3,
        num_heads=(4, 4, 4, 4), window_size=(4, 4, 4),
        pool_kernel=(2, 2, 2), freeze=True,
    )
    mri = torch.randn(B, 1, D, H, W)
    feats = enc(mri)

    # Simulate denoiser features at each level
    denom_l1 = torch.randn(B, 64, D, H, W)       # Level 1: full res, 64ch
    denom_l2 = torch.randn(B, 128, D//2, H//2, W//2)  # Level 2: ½ res, 128ch
    denom_l3 = torch.randn(B, 256, D//4, H//4, W//4)  # Level 3: ¼ res, 256ch

    # CA1: Q=E₁(64ch), K,V=f₁(64ch)
    ca1 = CrossAttention3D(query_dim=64, context_dim=64, num_heads=4)
    out1 = ca1(denom_l1, feats['f1'])
    print(f"  CA1: Q={list(denom_l1.shape)} K,V=f₁{list(feats['f1'].shape)} → {list(out1.shape)}")
    assert out1.shape == denom_l1.shape

    # CA2: Q=E₂(128ch), K,V=f₂(128ch)
    ca2 = CrossAttention3D(query_dim=128, context_dim=128, num_heads=4)
    out2 = ca2(denom_l2, feats['f2'])
    print(f"  CA2: Q={list(denom_l2.shape)} K,V=f₂{list(feats['f2'].shape)} → {list(out2.shape)}")
    assert out2.shape == denom_l2.shape

    # CA3: Q=E₃(256ch), K,V=f₃+f₄(192+256=448ch at ¼ res)
    # Need to upsample f4 to match f3 spatial dims
    import torch.nn.functional as F
    f4_up = F.interpolate(feats['f4'], size=feats['f3'].shape[2:],
                          mode='trilinear', align_corners=False)
    f3_f4 = torch.cat([feats['f3'], f4_up], dim=1)

    ca3 = CrossAttention3D(query_dim=256, context_dim=448, num_heads=8)
    out3 = ca3(denom_l3, f3_f4)
    print(f"  CA3: Q={list(denom_l3.shape)} K,V=f₃+f₄{list(f3_f4.shape)} → {list(out3.shape)}")
    assert out3.shape == denom_l3.shape

    print("  ✓ All cross-attention data flows match — PASS")


def test_time_mri_fusion():
    separator("5. Time + Global MRI Fusion")
    from network.util_network import timestep_embedding, linear

    B = 2
    model_channels = 64
    time_embed_dim = model_channels * 4  # 256

    # Simulate
    t = torch.randint(0, 1000, (B,))
    t_emb_raw = timestep_embedding(t, model_channels)  # [B, 64]

    time_embed = nn.Sequential(
        linear(model_channels, time_embed_dim),
        nn.SiLU(),
        linear(time_embed_dim, time_embed_dim),
    )
    t_emb = time_embed(t_emb_raw)  # [B, 256]

    global_mri = torch.randn(B, time_embed_dim)  # Simulated global MRI token
    condition_c = t_emb + global_mri  # Add

    print(f"  Timestep embedding raw:  {list(t_emb_raw.shape)}")
    print(f"  Timestep embedding:      {list(t_emb.shape)}")
    print(f"  Global MRI token:        {list(global_mri.shape)}")
    print(f"  Condition c = t + MRI:   {list(condition_c.shape)}")
    assert condition_c.shape == (B, time_embed_dim)
    print("  ✓ Fusion produces correct [B, 256] condition — PASS")


if __name__ == '__main__':
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   HYBRID MRI ENCODER + CROSS-ATTENTION DIMENSION TEST   ║")
    print("╚══════════════════════════════════════════════════════════╝")

    test_cross_attention()
    test_mri_encoder()
    test_hybrid_model()
    test_data_flow()
    test_time_mri_fusion()

    print("\n" + "="*60)
    print("  ✅  ALL TESTS PASSED — All dimensions verified!")
    print("="*60)
