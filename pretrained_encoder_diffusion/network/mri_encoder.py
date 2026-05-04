"""
MRI Semantic Encoder — Task-aware dual-head autoencoder for pretraining.

Architecture:
  Encoder (4 levels, shared):
    Stem: 7×7×3 conv → 3×3×1 conv     → c0 features
    L1:   2× ResBlock(GroupNorm) + SA3D  → f₁ (full res)
    L2:   StridedConv↓2 + 2×ResBlock + SA3D → f₂ (½ res)
    L3:   StridedConv↓2 + 2×ResBlock + SA3D → f₃ (¼ res)
    L4:   StridedConv↓2 + 2×ResBlock + SA3D → f₄ (⅛ res)
    Global: AdaptiveAvgPool + Transformer Bottleneck → [B, global_dim]

  Two Decoder Heads (U-Net style with skip connections):
    Head A — MRI Reconstruction:  predicts input MRI
    Head B — CT Prediction:       predicts paired CT (task-specific!)

Training loss:
    L = 0.5 × L1(CT) + 0.3 × L1(MRI) + 0.2 × SSIM(MRI)
    CT prediction forces encoder to learn CT-discriminative MRI features.

After pretraining, only MRISemanticEncoder is used (frozen) in the hybrid model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from network.util_network import conv_nd, normalization, zero_module


# ─── Utility: GroupNorm wrapper (stable for small batch sizes) ────────────────
def group_norm(channels, num_groups=8):
    """GroupNorm with automatic group adjustment if channels < num_groups."""
    g = min(num_groups, channels)
    while channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, channels)


# ─── Improved Encoder ResBlock with GroupNorm ─────────────────────────────────
class EncoderResBlock(nn.Module):
    """
    Dual-conv residual block with GroupNorm.
    Stronger than original: GroupNorm stable for batch=2, SiLU activation.
    """
    def __init__(self, in_ch, out_ch, dims=3, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            group_norm(in_ch),
            nn.SiLU(),
            conv_nd(dims, in_ch, out_ch, 3, padding=1),
            group_norm(out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            conv_nd(dims, out_ch, out_ch, 3, padding=1),
        )
        self.skip = conv_nd(dims, in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.skip(x) + self.block(x)


# ─── Strided Conv Downsampler (learned, better than AvgPool) ──────────────────
class StridedConvDown(nn.Module):
    """Learned downsampling with strided 3×3 conv. Better edge preservation."""
    def __init__(self, channels, stride=(2, 2, 1), dims=3):
        super().__init__()
        self.conv = conv_nd(dims, channels, channels, kernel_size=3,
                            stride=stride, padding=1)
        self.norm = group_norm(channels)

    def forward(self, x):
        return self.norm(self.conv(x))


# ─── Self-Attention with Relative Positional Bias ─────────────────────────────
class SelfAttention3D(nn.Module):
    """
    Window self-attention with relative positional bias.
    Encodes spatial relationships critical for anatomy landmarks.
    """
    def __init__(self, dim, num_heads=4, window_size=(4, 4, 4)):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        head_dim = max(1, dim // num_heads)
        self.scale = head_dim ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

        # Relative positional bias table
        ws = window_size
        self.rel_pos_bias = nn.Embedding(
            (2 * ws[0] - 1) * (2 * ws[1] - 1) * (2 * ws[2] - 1), num_heads
        )
        coords = self._get_rel_pos_index(ws)
        self.register_buffer('rel_pos_index', coords)

    @staticmethod
    def _get_rel_pos_index(ws):
        d = torch.arange(ws[0])
        h = torch.arange(ws[1])
        w = torch.arange(ws[2])
        grid = torch.stack(torch.meshgrid(d, h, w))  # [3, D, H, W]  (ij indexing is default)
        flat = grid.flatten(1)  # [3, L]
        rel = flat[:, :, None] - flat[:, None, :]  # [3, L, L]
        rel[0] += ws[0] - 1
        rel[1] += ws[1] - 1
        rel[2] += ws[2] - 1
        rel[0] *= (2 * ws[1] - 1) * (2 * ws[2] - 1)
        rel[1] *= (2 * ws[2] - 1)
        return rel.sum(0)  # [L, L]

    def forward(self, x):
        """x: [B, C, D, H, W]"""
        B, C, D, H, W = x.shape
        ws = self.window_size

        # Pad
        pd = (ws[0] - D % ws[0]) % ws[0]
        ph = (ws[1] - H % ws[1]) % ws[1]
        pw = (ws[2] - W % ws[2]) % ws[2]
        if pd or ph or pw:
            x = F.pad(x, (0, pw, 0, ph, 0, pd))
        _, _, Dp, Hp, Wp = x.shape

        nD, nH, nW = Dp // ws[0], Hp // ws[1], Wp // ws[2]
        nW_total = nD * nH * nW
        L = ws[0] * ws[1] * ws[2]

        # Window partition: [B*nW, L, C]
        x = x.reshape(B, C, nD, ws[0], nH, ws[1], nW, ws[2])
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
        x = x.reshape(B * nW_total, L, C)

        shortcut = x
        x = self.norm(x)

        qkv = self.qkv(x).reshape(B * nW_total, L, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Add relative positional bias
        bias = self.rel_pos_bias(self.rel_pos_index.reshape(-1))
        bias = bias.reshape(L, L, self.num_heads).permute(2, 0, 1)
        attn = attn + bias.unsqueeze(0)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B * nW_total, L, C)
        x = self.proj(x) + shortcut

        # Reverse window partition
        x = x.reshape(B, nD, nH, nW, ws[0], ws[1], ws[2], C)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        x = x.reshape(B, C, Dp, Hp, Wp)

        if pd or ph or pw:
            x = x[:, :, :D, :H, :W]
        return x


# ─── Transformer Bottleneck for Global Token ──────────────────────────────────
class TransformerBottleneck(nn.Module):
    """2-layer transformer for enriching the global MRI token."""
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(dim),
                nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True),
                nn.LayerNorm(dim),
                nn.Sequential(
                    nn.Linear(dim, int(dim * mlp_ratio)),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(int(dim * mlp_ratio), dim),
                    nn.Dropout(dropout),
                ),
            ]) for _ in range(2)
        ])

    def forward(self, x):
        """x: [B, dim] — treated as a single-token sequence."""
        x = x.unsqueeze(1)  # [B, 1, dim]
        for norm1, attn, norm2, ff in self.layers:
            x = x + attn(norm1(x), norm1(x), norm1(x))[0]
            x = x + ff(norm2(x))
        return x.squeeze(1)  # [B, dim]


# ─── Decoder Block ────────────────────────────────────────────────────────────
class DecoderBlock(nn.Module):
    """Upsample + skip concat + 2× ResBlock."""
    def __init__(self, in_ch, skip_ch, out_ch, scale=(2, 2, 1), dims=3):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale, mode='trilinear', align_corners=False)
        self.res1 = EncoderResBlock(in_ch + skip_ch, out_ch, dims=dims)
        self.res2 = EncoderResBlock(out_ch, out_ch, dims=dims)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle potential size mismatch at boundary
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.res2(self.res1(x))


# ─── MRI Semantic Encoder ─────────────────────────────────────────────────────
class MRISemanticEncoder(nn.Module):
    """
    Improved MRI encoder with:
    - Input stem for low-level feature capture
    - Strided conv downsampling (learned, better than AvgPool)
    - GroupNorm for small-batch stability
    - Window self-attention with relative positional bias
    - Transformer bottleneck for global token
    - Conditional gradient/freeze support

    Returns dict: {'f1', 'f2', 'f3', 'f4', 'global'}
    """

    def __init__(
        self,
        in_channels=1,
        enc_channels=(64, 128, 192, 256),
        global_dim=256,
        dims=3,
        num_heads=(4, 4, 8, 8),
        window_size=(4, 4, 4),
        pool_kernel=(2, 2, 1),   # Z kept at 1 for thin-slice patches
        freeze=True,
        dropout=0.0,
    ):
        super().__init__()
        self.enc_channels = enc_channels
        self.global_dim = global_dim
        self._freeze_encoder = freeze

        c1, c2, c3, c4 = enc_channels

        # ── Input stem ───────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            conv_nd(dims, in_channels, c1 // 2, kernel_size=3, padding=1),
            group_norm(c1 // 2),
            nn.SiLU(),
            conv_nd(dims, c1 // 2, c1 // 2, kernel_size=3, padding=1),
            group_norm(c1 // 2),
            nn.SiLU(),
        )

        # ── Level 1: 2×ResBlock + SelfAttn, full res ─────────────────────────
        self.level1 = nn.Sequential(
            EncoderResBlock(c1 // 2, c1, dims=dims, dropout=dropout),
            EncoderResBlock(c1, c1, dims=dims, dropout=dropout),
            SelfAttention3D(c1, num_heads=num_heads[0], window_size=window_size),
        )

        # ── Level 2: StridedConv↓ + 2×ResBlock + SelfAttn, ½ res ─────────────
        self.down2 = StridedConvDown(c1, stride=pool_kernel, dims=dims)
        self.level2 = nn.Sequential(
            EncoderResBlock(c1, c2, dims=dims, dropout=dropout),
            EncoderResBlock(c2, c2, dims=dims, dropout=dropout),
            SelfAttention3D(c2, num_heads=num_heads[1], window_size=window_size),
        )

        # ── Level 3: StridedConv↓ + 2×ResBlock + SelfAttn, ¼ res ─────────────
        self.down3 = StridedConvDown(c2, stride=pool_kernel, dims=dims)
        self.level3 = nn.Sequential(
            EncoderResBlock(c2, c3, dims=dims, dropout=dropout),
            EncoderResBlock(c3, c3, dims=dims, dropout=dropout),
            SelfAttention3D(c3, num_heads=num_heads[2], window_size=window_size),
        )

        # ── Level 4: StridedConv↓ + 2×ResBlock + SelfAttn, ⅛ res ─────────────
        self.down4 = StridedConvDown(c3, stride=pool_kernel, dims=dims)
        self.level4 = nn.Sequential(
            EncoderResBlock(c3, c4, dims=dims, dropout=dropout),
            EncoderResBlock(c4, c4, dims=dims, dropout=dropout),
            SelfAttention3D(c4, num_heads=num_heads[3], window_size=window_size),
        )

        # ── Global token: AdaptivePool + Transformer Bottleneck ───────────────
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.global_proj = nn.Linear(c4, global_dim)
        self.global_bottleneck = TransformerBottleneck(
            global_dim, num_heads=min(8, global_dim // 32)
        )

        # ── Freeze if required ────────────────────────────────────────────────
        if freeze:
            self._freeze()

    def _freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def train(self, mode=True):
        """Stay in eval if frozen; otherwise behave normally."""
        if self._freeze_encoder:
            super().train(False)
        else:
            super().train(mode)
        return self

    def forward(self, mri):
        """
        Args:
            mri: [B, 1, D, H, W]
        Returns:
            dict with f1, f2, f3, f4  and  global [B, global_dim]
        """
        x = self.stem(mri)

        f1 = self.level1(x)
        f2 = self.level2(self.down2(f1))
        f3 = self.level3(self.down3(f2))
        f4 = self.level4(self.down4(f3))

        g = self.global_pool(f4).flatten(1)   # [B, c4]
        g = self.global_proj(g)                # [B, global_dim]
        g = self.global_bottleneck(g)          # [B, global_dim]  (Transformer)

        return {'f1': f1, 'f2': f2, 'f3': f3, 'f4': f4, 'global': g}


# ─── Dual-Head Autoencoder for Pretraining ────────────────────────────────────
class MRIAutoencoder(nn.Module):
    """
    Dual-task pretraining autoencoder.

    Shared encoder + two decoder heads:
      Head A → reconstruct MRI (structural constraint)
      Head B → predict CT     (task-specific constraint: learn CT-relevant features)

    Loss:
      L = 0.5 * L1(CT) + 0.3 * L1(MRI_recon) + 0.2 * SSIM(MRI_recon)

    The CT prediction task forces the encoder to attend to tissue boundaries,
    bone density, and HU-discriminative MRI features rather than MRI-specific
    artifacts.
    """

    def __init__(
        self,
        enc_channels=(64, 128, 192, 256),
        global_dim=256,
        window_size=(4, 4, 4),
        num_heads=(4, 4, 8, 8),
        pool_kernel=(2, 2, 1),
        dropout=0.1,
    ):
        super().__init__()
        c1, c2, c3, c4 = enc_channels

        # ── Shared encoder (NOT frozen during pretraining) ────────────────────
        self.encoder = MRISemanticEncoder(
            in_channels=1,
            enc_channels=enc_channels,
            global_dim=global_dim,
            num_heads=num_heads,
            window_size=window_size,
            pool_kernel=pool_kernel,
            freeze=False,   # Trainable during pretraining
            dropout=dropout,
        )

        # ── Shared decoder backbone (4 levels up) ────────────────────────────
        # Uses skip connections from encoder levels (U-Net style)
        up_scale = tuple(pool_kernel)  # e.g. (2, 2, 1)
        self.dec4 = DecoderBlock(c4, c3, c3, scale=up_scale)     # f4 + f3 skip → c3
        self.dec3 = DecoderBlock(c3, c2, c2, scale=up_scale)     # dec4 + f2 skip → c2
        self.dec2 = DecoderBlock(c2, c1, c1, scale=up_scale)     # dec3 + f1 skip → c1

        # Final decoder stage (no more skips — reconstruct at full res)
        self.dec1 = nn.Sequential(
            EncoderResBlock(c1, c1 // 2, dropout=dropout),
            EncoderResBlock(c1 // 2, c1 // 2, dropout=dropout),
        )

        # ── Head A: MRI Reconstruction ───────────────────────────────────────
        self.mri_head = nn.Sequential(
            group_norm(c1 // 2),
            nn.SiLU(),
            conv_nd(3, c1 // 2, 1, kernel_size=1),
            nn.Tanh(),   # MRI normalised to [-1, 1]
        )

        # ── Head B: CT Prediction ─────────────────────────────────────────────
        # Separate lightweight conv tower to project shared decoder → CT
        # This head gets stronger gradient signal for CT-discriminative features
        self.ct_head = nn.Sequential(
            EncoderResBlock(c1 // 2, c1 // 2, dropout=dropout),
            group_norm(c1 // 2),
            nn.SiLU(),
            conv_nd(3, c1 // 2, 1, kernel_size=1),
            nn.Tanh(),   # CT normalised to [-1, 1]
        )

    def forward(self, mri):
        """
        Args:
            mri: [B, 1, D, H, W]  — MRI volume, normalised to [-1, 1]

        Returns:
            mri_recon: [B, 1, D, H, W]  — MRI reconstruction
            ct_pred:   [B, 1, D, H, W]  — predicted CT (task-specific)
            feats:     dict with f1..f4,global  — for inspection/loading
        """
        feats = self.encoder(mri)
        f1, f2, f3, f4 = feats['f1'], feats['f2'], feats['f3'], feats['f4']

        # Decode with skip connections
        d = self.dec4(f4, f3)    # upsample f4, concat f3
        d = self.dec3(d,  f2)    # upsample d, concat f2
        d = self.dec2(d,  f1)    # upsample d, concat f1
        d = self.dec1(d)          # final double ResBlock

        mri_recon = self.mri_head(d)
        ct_pred   = self.ct_head(d)

        return mri_recon, ct_pred, feats
