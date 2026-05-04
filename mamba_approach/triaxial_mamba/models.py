"""
TriMamba-UNet V2: Tri-Axial Bidirectional Mamba for MRI-to-CT Synthesis.

VRAM-optimized for 24GB GPUs (target: ≤22GB with 2GB buffer).

Key design:
  1. TriAxialMambaBlock — bidirectional scans along D, H, W axes
  2. CBAM3D on skip connections — channel + spatial attention gating
  3. Deep supervision — auxiliary outputs at decoder stages 2 & 3
  4. Interpolate+Conv upsampling — no checkerboard artifacts
  5. Gradient checkpointing — saves ~60% activation memory
  6. base_ch=32 — accuracy comes from tri-axial scanning, not channel width

Memory budget (batch=1, patch=32×128×128, fp16):
  Model weights:  ~18M params × 2 bytes  =  ~36 MB
  Optimizer:      ~18M × 8 bytes (AdamW)  = ~144 MB
  EMA copy:       ~18M × 4 bytes          =  ~72 MB
  Activations (with grad ckpt + AMP):     = ~8-12 GB
  Total estimated peak:                   = ~14-18 GB  ✓

Dependencies:
    pip install mamba-ssm causal-conv1d
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("[WARNING] mamba-ssm not installed. Using fallback GRU-based SSM.")


# ─────────────────────────────────────────────
# Fallback SSM block if mamba-ssm not installed
# ─────────────────────────────────────────────
class FallbackSSMBlock(nn.Module):
    """GRU-based sequential block as a Mamba fallback."""
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.gru  = nn.GRU(d_model, d_model, batch_first=True, bidirectional=False)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x, _ = self.gru(x)
        return x + residual


def get_ssm_block(d_model, d_state=16, d_conv=4, expand=2):
    if MAMBA_AVAILABLE:
        return Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    else:
        return FallbackSSMBlock(d_model)


# ─────────────────────────────────────────────
# Shared building blocks
# ─────────────────────────────────────────────
class ConvNormAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel, stride=stride, padding=padding),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ResConvBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(ch, ch),
            ConvNormAct(ch, ch)
        )

    def forward(self, x):
        return x + self.block(x)


# ═════════════════════════════════════════════
# TriAxial Bidirectional Mamba Block
# (VRAM optimized: sequential axis processing)
# ═════════════════════════════════════════════
class TriAxialMambaBlock(nn.Module):
    """
    Scans the 3D volume along D, H, W axes separately (bidirectional).

    VRAM optimizations:
      - Axes processed sequentially (not all 3 kept in memory at once)
      - Bidirectional done via flip (reuses same SSM memory region)

    Input/Output: (B, C, D, H, W)
    """
    def __init__(self, channels, d_state=16):
        super().__init__()
        self.channels = channels

        # Normalization before SSM
        self.norm = nn.LayerNorm(channels)

        # 3 axes × 2 directions = 6 SSM blocks
        self.ssm_d_fwd = get_ssm_block(d_model=channels, d_state=d_state)
        self.ssm_d_bwd = get_ssm_block(d_model=channels, d_state=d_state)
        self.ssm_h_fwd = get_ssm_block(d_model=channels, d_state=d_state)
        self.ssm_h_bwd = get_ssm_block(d_model=channels, d_state=d_state)
        self.ssm_w_fwd = get_ssm_block(d_model=channels, d_state=d_state)
        self.ssm_w_bwd = get_ssm_block(d_model=channels, d_state=d_state)

        # Fusion: merge 3 axis outputs (3C → C)
        self.fusion = nn.Sequential(
            nn.Conv3d(channels * 3, channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def _scan_axis(self, x, ssm_fwd, ssm_bwd, axis_dim):
        """
        Scan along one spatial axis bidirectionally.
        x: (B, C, D, H, W), axis_dim: 2=D, 3=H, 4=W
        Returns: (B, C, D, H, W)
        """
        B, C, D, H, W = x.shape

        if axis_dim == 2:  # scan along D
            x_perm = x.permute(0, 3, 4, 2, 1).contiguous()
            x_seq = x_perm.reshape(B * H * W, D, C)
        elif axis_dim == 3:  # scan along H
            x_perm = x.permute(0, 2, 4, 3, 1).contiguous()
            x_seq = x_perm.reshape(B * D * W, H, C)
        elif axis_dim == 4:  # scan along W
            x_perm = x.permute(0, 2, 3, 4, 1).contiguous()
            x_seq = x_perm.reshape(B * D * H, W, C)
        else:
            raise ValueError(f"Invalid axis_dim: {axis_dim}")

        # Normalize
        x_normed = self.norm(x_seq)

        # Bidirectional scan
        y_fwd = ssm_fwd(x_normed)
        y_bwd = torch.flip(ssm_bwd(torch.flip(x_normed, dims=[1])), dims=[1])
        y = (y_fwd + y_bwd) * 0.5

        # Free intermediates ASAP
        del x_seq, x_normed, y_fwd, y_bwd

        # Reshape back
        if axis_dim == 2:
            y = y.reshape(B, H, W, D, C).permute(0, 4, 3, 1, 2).contiguous()
        elif axis_dim == 3:
            y = y.reshape(B, D, W, H, C).permute(0, 4, 1, 3, 2).contiguous()
        elif axis_dim == 4:
            y = y.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()

        return y

    def forward(self, x):
        residual = x

        # Process axes SEQUENTIALLY to minimize peak VRAM
        # Each axis result is kept, but intermediates within _scan_axis are freed
        y_d = self._scan_axis(x, self.ssm_d_fwd, self.ssm_d_bwd, axis_dim=2)
        y_h = self._scan_axis(x, self.ssm_h_fwd, self.ssm_h_bwd, axis_dim=3)
        y_w = self._scan_axis(x, self.ssm_w_fwd, self.ssm_w_bwd, axis_dim=4)

        # Fuse 3 axes
        y_cat = torch.cat([y_d, y_h, y_w], dim=1)  # (B, 3C, D, H, W)
        del y_d, y_h, y_w  # free immediately
        y_fused = self.fusion(y_cat)
        del y_cat

        return y_fused + residual


# ═════════════════════════════════════════════
# CBAM3D — Channel + Spatial Attention
# ═════════════════════════════════════════════
class ChannelAttention3D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False)
        )

    def forward(self, x):
        B, C, D, H, W = x.shape
        avg = x.mean(dim=(2, 3, 4))
        mx  = x.amax(dim=(2, 3, 4))
        att = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        return x * att.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)


class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.amax(dim=1, keepdim=True)
        att = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * att


class CBAM3D(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.ca = ChannelAttention3D(channels, reduction)
        self.sa = SpatialAttention3D(spatial_kernel)

    def forward(self, x):
        return self.sa(self.ca(x))


# ═════════════════════════════════════════════
# TriMamba Encoder/Decoder Block with Gradient Checkpointing
# ═════════════════════════════════════════════
class TriMambaEncoderBlock(nn.Module):
    """
    ResConv → TriAxialMamba → InstanceNorm
    Wraps TriAxialMambaBlock with gradient checkpointing to save ~60% VRAM.
    """
    def __init__(self, ch, d_state=16, use_checkpoint=True):
        super().__init__()
        self.conv  = ResConvBlock(ch)
        self.mamba = TriAxialMambaBlock(ch, d_state=d_state)
        self.norm  = nn.InstanceNorm3d(ch, affine=True)
        self.use_checkpoint = use_checkpoint

    def _forward_mamba(self, x):
        """Wrapper for gradient checkpointing (must not have inplace ops)."""
        return self.mamba(x)

    def forward(self, x):
        x = self.conv(x)
        if self.use_checkpoint and self.training:
            # Gradient checkpointing: saves activation memory by
            # recomputing Mamba forward during backward pass
            x = grad_checkpoint(self._forward_mamba, x, use_reentrant=False)
        else:
            x = self.mamba(x)
        return self.norm(x)


# ═════════════════════════════════════════════
# TriMamba-UNet V2 (VRAM-Optimized)
# ═════════════════════════════════════════════
class TriMambaUNet(nn.Module):
    """
    TriMamba-UNet V2 for MRI-to-CT synthesis.
    VRAM-optimized for 24GB GPUs.

    Architecture (base_ch=32):
      Encoder: 32 → 64 → 128 → 256 (bottleneck)
      Decoder: 256 → 128 → 64 → 32

    Estimated VRAM: ~14-18 GB (batch=1, patch=32×128×128, AMP+GradCkpt)
    Estimated params: ~18M
    """
    def __init__(self, in_ch=1, out_ch=1, base_ch=32, d_state=16,
                 deep_supervision=True, use_checkpoint=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        c1, c2, c3, c4 = base_ch, base_ch*2, base_ch*4, base_ch*8

        # ── Stem ──
        self.stem = nn.Sequential(
            ConvNormAct(in_ch, c1),
            ConvNormAct(c1, c1),
        )

        # ── Encoder ──
        self.enc1 = TriMambaEncoderBlock(c1, d_state=d_state,
                                          use_checkpoint=use_checkpoint)

        self.down1 = ConvNormAct(c1, c2, stride=2)
        self.enc2  = TriMambaEncoderBlock(c2, d_state=d_state,
                                          use_checkpoint=use_checkpoint)

        self.down2 = ConvNormAct(c2, c3, stride=2)
        self.enc3  = TriMambaEncoderBlock(c3, d_state=d_state,
                                          use_checkpoint=use_checkpoint)

        self.down3 = ConvNormAct(c3, c4, stride=2)
        self.enc4  = TriMambaEncoderBlock(c4, d_state=d_state,
                                          use_checkpoint=use_checkpoint)  # bottleneck

        # ── Decoder ──
        # Decode 3: c4 → c3
        self.up3_conv = nn.Sequential(
            nn.Conv3d(c4, c3, kernel_size=1),
            nn.InstanceNorm3d(c3, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.cbam3 = CBAM3D(c3)
        self.dec3  = TriMambaEncoderBlock(c3 * 2, d_state=d_state,
                                          use_checkpoint=use_checkpoint)
        self.proj3 = ConvNormAct(c3 * 2, c3)

        # Decode 2: c3 → c2
        self.up2_conv = nn.Sequential(
            nn.Conv3d(c3, c2, kernel_size=1),
            nn.InstanceNorm3d(c2, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.cbam2 = CBAM3D(c2)
        self.dec2  = TriMambaEncoderBlock(c2 * 2, d_state=d_state,
                                          use_checkpoint=use_checkpoint)
        self.proj2 = ConvNormAct(c2 * 2, c2)

        # Decode 1: c2 → c1
        self.up1_conv = nn.Sequential(
            nn.Conv3d(c2, c1, kernel_size=1),
            nn.InstanceNorm3d(c1, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.cbam1 = CBAM3D(c1)
        self.dec1  = TriMambaEncoderBlock(c1 * 2, d_state=d_state,
                                          use_checkpoint=use_checkpoint)
        self.proj1 = ConvNormAct(c1 * 2, c1)

        # ── Output Head ──
        self.head = nn.Sequential(
            nn.Conv3d(c1, out_ch, 1),
            nn.Tanh()
        )

        # ── Deep Supervision Heads ──
        if deep_supervision:
            self.aux_head_d2 = nn.Sequential(nn.Conv3d(c2, out_ch, 1), nn.Tanh())
            self.aux_head_d3 = nn.Sequential(nn.Conv3d(c3, out_ch, 1), nn.Tanh())

    def _upsample_like(self, x, target):
        return F.interpolate(x, size=target.shape[2:], mode='trilinear',
                            align_corners=False)

    def forward(self, x):
        # Stem
        x0 = self.stem(x)

        # Encoder
        e1 = self.enc1(x0)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))

        # Decoder 3
        up3 = self.up3_conv(self._upsample_like(e4, e3))
        d3 = self.proj3(self.dec3(torch.cat([up3, self.cbam3(e3)], dim=1)))
        del e4, up3  # free encoder features as soon as used

        # Decoder 2
        up2 = self.up2_conv(self._upsample_like(d3, e2))
        d2 = self.proj2(self.dec2(torch.cat([up2, self.cbam2(e2)], dim=1)))
        del e3, up2

        # Decoder 1
        up1 = self.up1_conv(self._upsample_like(d2, e1))
        d1 = self.proj1(self.dec1(torch.cat([up1, self.cbam1(e1)], dim=1)))
        del e2, up1

        # Output
        out = self.head(d1)

        if self.deep_supervision and self.training:
            aux2 = self.aux_head_d2(d2)
            aux3 = self.aux_head_d3(d3)
            return out, aux2, aux3

        return out


# ─────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────
def get_model(model_name='trimamba', base_ch=32, deep_supervision=True,
              use_checkpoint=True):
    """
    Args:
        model_name: 'trimamba' (only option now)
        base_ch: base channel count (32 recommended for 24GB GPUs)
        deep_supervision: enable auxiliary outputs
        use_checkpoint: enable gradient checkpointing (saves ~60% VRAM)
    """
    if model_name == 'trimamba':
        model = TriMambaUNet(in_ch=1, out_ch=1, base_ch=base_ch,
                             deep_supervision=deep_supervision,
                             use_checkpoint=use_checkpoint)
    else:
        raise ValueError(f"Unknown model: {model_name}. Use 'trimamba'.")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {model_name} | base_ch={base_ch} | "
          f"Parameters: {n_params/1e6:.1f}M | "
          f"deep_supervision={deep_supervision} | "
          f"grad_checkpoint={use_checkpoint}")
    return model