"""
Mamba-based encoder-decoder models for MRI-to-CT synthesis.

SegMamba: hybrid hierarchical SSM + CNN
U-Mamba:  nnU-Net style with Mamba blocks replacing conv blocks

Both output a single-channel CT volume from single-channel MRI input.

Dependencies:
    pip install mamba-ssm causal-conv1d monai
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

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
        # x: (B, L, d_model)
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


class MambaBlock3D(nn.Module):
    """
    Applies an SSM (Mamba) block on flattened spatial tokens.
    Input: (B, C, D, H, W)
    Output: (B, C, D, H, W)
    """
    def __init__(self, channels, d_state=16):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.ssm  = get_ssm_block(d_model=channels, d_state=d_state)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x):
        B, C, D, H, W = x.shape
        # Flatten spatial -> tokens
        x_flat = x.flatten(2).permute(0, 2, 1)  # (B, D*H*W, C)
        residual = x_flat
        x_norm = self.norm(x_flat)
        x_ssm  = self.ssm(x_norm)
        x_out  = self.proj(x_ssm) + residual
        # Reshape back
        x_out = x_out.permute(0, 2, 1).reshape(B, C, D, H, W)
        return x_out


# ─────────────────────────────────────────────
# SegMamba
# ─────────────────────────────────────────────
class SegMambaEncoder(nn.Module):
    def __init__(self, in_ch, base_ch=32):
        super().__init__()
        # Stage 1
        self.enc1_conv  = ConvNormAct(in_ch, base_ch)
        self.enc1_mamba = MambaBlock3D(base_ch)
        # Stage 2
        self.down1      = ConvNormAct(base_ch, base_ch*2, stride=2)
        self.enc2_conv  = ResConvBlock(base_ch*2)
        self.enc2_mamba = MambaBlock3D(base_ch*2)
        # Stage 3
        self.down2      = ConvNormAct(base_ch*2, base_ch*4, stride=2)
        self.enc3_conv  = ResConvBlock(base_ch*4)
        self.enc3_mamba = MambaBlock3D(base_ch*4)
        # Stage 4 (bottleneck)
        self.down3      = ConvNormAct(base_ch*4, base_ch*8, stride=2)
        self.enc4_conv  = ResConvBlock(base_ch*8)
        self.enc4_mamba = MambaBlock3D(base_ch*8)

    def forward(self, x):
        e1 = self.enc1_mamba(self.enc1_conv(x))
        e2 = self.enc2_mamba(self.enc2_conv(self.down1(e1)))
        e3 = self.enc3_mamba(self.enc3_conv(self.down2(e2)))
        e4 = self.enc4_mamba(self.enc4_conv(self.down3(e3)))
        return e1, e2, e3, e4


class SegMambaDecoder(nn.Module):
    def __init__(self, base_ch=32):
        super().__init__()
        self.up3   = nn.ConvTranspose3d(base_ch*8, base_ch*4, 2, stride=2)
        self.dec3  = nn.Sequential(ResConvBlock(base_ch*8), MambaBlock3D(base_ch*8))
        self.proj3 = ConvNormAct(base_ch*8, base_ch*4)

        self.up2   = nn.ConvTranspose3d(base_ch*4, base_ch*2, 2, stride=2)
        self.dec2  = nn.Sequential(ResConvBlock(base_ch*4), MambaBlock3D(base_ch*4))
        self.proj2 = ConvNormAct(base_ch*4, base_ch*2)

        self.up1   = nn.ConvTranspose3d(base_ch*2, base_ch, 2, stride=2)
        self.dec1  = nn.Sequential(ResConvBlock(base_ch*2), MambaBlock3D(base_ch*2))
        self.proj1 = ConvNormAct(base_ch*2, base_ch)

    def forward(self, e1, e2, e3, e4):
        d3 = self.up3(e4)
        d3 = self.proj3(self.dec3(torch.cat([d3, e3], dim=1)))

        d2 = self.up2(d3)
        d2 = self.proj2(self.dec2(torch.cat([d2, e2], dim=1)))

        d1 = self.up1(d2)
        d1 = self.proj1(self.dec1(torch.cat([d1, e1], dim=1)))

        return d1


class SegMamba(nn.Module):
    """
    SegMamba adapted for MRI-to-CT synthesis.
    Input:  (B, 1, D, H, W) MRI
    Output: (B, 1, D, H, W) synthetic CT
    """
    def __init__(self, in_ch=1, out_ch=1, base_ch=32):
        super().__init__()
        self.encoder = SegMambaEncoder(in_ch, base_ch)
        self.decoder = SegMambaDecoder(base_ch)
        self.head    = nn.Sequential(
            nn.Conv3d(base_ch, out_ch, 1),
            nn.Tanh()  # output in [-1, 1] matching normalized CT
        )

    def forward(self, x):
        e1, e2, e3, e4 = self.encoder(x)
        dec = self.decoder(e1, e2, e3, e4)
        return self.head(dec)


# ─────────────────────────────────────────────
# U-Mamba
# ─────────────────────────────────────────────
class UMambaBlock(nn.Module):
    """U-Mamba block: Conv -> Mamba -> residual"""
    def __init__(self, ch, d_state=16, time_embed_dim=None):
        super().__init__()
        self.conv  = ResConvBlock(ch)
        self.mamba = MambaBlock3D(ch, d_state=d_state)
        self.norm  = nn.InstanceNorm3d(ch, affine=True)
        if time_embed_dim is not None:
            self.emb_layer = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_embed_dim, ch)
            )
        else:
            self.emb_layer = None

    def forward(self, x, emb=None):
        x = self.conv(x)
        if self.emb_layer is not None and emb is not None:
            emb_out = self.emb_layer(emb)
            while len(emb_out.shape) < len(x.shape):
                emb_out = emb_out.unsqueeze(-1)
            x = x + emb_out
        x = self.mamba(x)
        return self.norm(x)


class UMamba(nn.Module):
    """
    U-Mamba adapted for MRI-to-CT synthesis.
    Retains nnU-Net-like self-configuring design with Mamba blocks.
    Input:  (B, 1, D, H, W) MRI
    Output: (B, 1, D, H, W) synthetic CT
    """
    def __init__(self, in_ch=1, out_ch=1, base_ch=32, is_diffusion=False, strides=((2,2,2), (2,2,2), (2,2,2))):
        super().__init__()
        self.is_diffusion = is_diffusion
        self.base_ch = base_ch
        
        if is_diffusion:
            time_embed_dim = base_ch * 4
            self.time_embed = nn.Sequential(
                nn.Linear(base_ch, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )
        else:
            time_embed_dim = None

        s1, s2, s3 = strides

        # Encoder
        self.enc1 = UMambaBlock(base_ch, time_embed_dim=time_embed_dim)
        self.stem  = ConvNormAct(in_ch, base_ch)
        self.down1 = ConvNormAct(base_ch,   base_ch*2, stride=s1)
        self.enc2  = UMambaBlock(base_ch*2, time_embed_dim=time_embed_dim)
        self.down2 = ConvNormAct(base_ch*2, base_ch*4, stride=s2)
        self.enc3  = UMambaBlock(base_ch*4, time_embed_dim=time_embed_dim)
        self.down3 = ConvNormAct(base_ch*4, base_ch*8, stride=s3)
        self.enc4  = UMambaBlock(base_ch*8, time_embed_dim=time_embed_dim)

        # Decoder
        self.up3   = nn.ConvTranspose3d(base_ch*8, base_ch*4, s3, stride=s3)
        self.dec3  = UMambaBlock(base_ch*8, time_embed_dim=time_embed_dim)
        self.proj3 = ConvNormAct(base_ch*8, base_ch*4)

        self.up2   = nn.ConvTranspose3d(base_ch*4, base_ch*2, s2, stride=s2)
        self.dec2  = UMambaBlock(base_ch*4, time_embed_dim=time_embed_dim)
        self.proj2 = ConvNormAct(base_ch*4, base_ch*2)

        self.up1   = nn.ConvTranspose3d(base_ch*2, base_ch, s1, stride=s1)
        self.dec1  = UMambaBlock(base_ch*2, time_embed_dim=time_embed_dim)
        self.proj1 = ConvNormAct(base_ch*2, base_ch)

        if is_diffusion:
            self.head = nn.Conv3d(base_ch, out_ch, 1)
        else:
            self.head  = nn.Sequential(
                nn.Conv3d(base_ch, out_ch, 1),
                nn.Tanh()
            )

    def forward(self, x, t=None, **kwargs):
        emb = None
        if self.is_diffusion and t is not None:
            emb = self.time_embed(timestep_embedding(t, self.base_ch))
            
        e1 = self.enc1(self.stem(x), emb=emb)
        e2 = self.enc2(self.down1(e1), emb=emb)
        e3 = self.enc3(self.down2(e2), emb=emb)
        e4 = self.enc4(self.down3(e3), emb=emb)

        d3 = self.proj3(self.dec3(torch.cat([self.up3(e4), e3], dim=1), emb=emb))
        d2 = self.proj2(self.dec2(torch.cat([self.up2(d3), e2], dim=1), emb=emb))
        d1 = self.proj1(self.dec1(torch.cat([self.up1(d2), e1], dim=1), emb=emb))

        return self.head(d1)


# ─────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────
def get_model(model_name='segmamba', base_ch=32):
    """
    Args:
        model_name: 'segmamba' or 'umamba'
        base_ch: base channel count (32 is good for A5000 24GB)
    """
    if model_name == 'segmamba':
        model = SegMamba(in_ch=1, out_ch=1, base_ch=base_ch)
    elif model_name == 'umamba':
        model = UMamba(in_ch=1, out_ch=1, base_ch=base_ch)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'segmamba' or 'umamba'.")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {model_name} | Parameters: {n_params/1e6:.1f}M")
    return model