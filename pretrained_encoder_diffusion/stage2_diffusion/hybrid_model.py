"""
HybridDiffusionModel — Complete model integrating:
  1. Frozen MRI Semantic Encoder  (produces f₁–f₄ + global token)
  2. Time + Global MRI Fusion     (adds global MRI token to timestep emb)
  3. Denoising UNet with CrossAttn (Q from denoiser, K/V from MRI encoder)

Architecture (from hybrid_mri_ct_diffusion_architecture.svg):

  MRI Encoder (frozen):
    L1: ResBlock+Swin   64ch  full res  → f₁
    L2: AvgPool+ResBlock+Swin  128ch  ½ res   → f₂
    L3: AvgPool+ResBlock+Swin  192ch  ¼ res   → f₃
    L4: AvgPool+ResBlock+Swin  256ch  ⅛ res   → f₄
    MLP projection: f₄ → global token [B,256]

  Time + Global MRI Fusion:
    condition c = sinusoidal_embed(t) + global_MRI_token   [B, time_embed_dim]

  Denoising UNet (trainable, 3 levels):
    Init Conv:     1ch → 64ch, full res
    Down L1:       ResBlock+TimeCond(c) 64ch   + CrossAttn(Q=E₁, K,V=f₁)
    Downsample×2
    Down L2:       ResBlock+TimeCond(c) 128ch  + CrossAttn(Q=E₂, K,V=f₂)
    Downsample×2
    Down L3:       ResBlock+TimeCond(c) 256ch  + CrossAttn(Q=E₃, K,V=f₃+f₄)
    Middle Block:  2×ResBlock + global c
    Up L3 + skip E₃
    Up L2 + skip E₂
    Up L1 + skip E₁
    Output Conv:   64ch → 2ch (predict ε + v)
"""

import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from network.util_network import (
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from network.cross_attention import CrossAttention3D
from network.mri_encoder import MRISemanticEncoder


# ─── Timestep-conditioned ResBlock ───────────────────────────────────────────
class TimestepBlock(nn.Module):
    """Module whose forward() takes (x, emb) where emb is a timestep embedding."""
    def forward(self, x, emb):
        raise NotImplementedError


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """Sequential that passes timestep embeddings to children that support it."""
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class ResBlock(TimestepBlock):
    """
    Residual block with timestep conditioning via FiLM (scale+shift).
    Optionally includes up/downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout=0.0,
        out_channels=None,
        dims=3,
        use_scale_shift_norm=True,
        up=False,
        down=False,
        sample_kernel=None,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_scale_shift_norm = use_scale_shift_norm
        self.updown = up or down

        # Up/Down sampling
        if up and sample_kernel is not None:
            if dims == 3:
                sk = (sample_kernel[0], sample_kernel[1], sample_kernel[2])
            else:
                sk = (sample_kernel[0], sample_kernel[1])
            self.h_upd = nn.Upsample(scale_factor=sk, mode='nearest')
            self.x_upd = nn.Upsample(scale_factor=sk, mode='nearest')
        elif down and sample_kernel is not None:
            if dims == 3:
                sk = (1/sample_kernel[0], 1/sample_kernel[1], 1/sample_kernel[2])
            else:
                sk = (1/sample_kernel[0], 1/sample_kernel[1])
            self.h_upd = nn.Upsample(scale_factor=sk, mode='nearest')
            self.x_upd = nn.Upsample(scale_factor=sk, mode='nearest')
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )

        if channels != self.out_channels:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)
        else:
            self.skip_connection = nn.Identity()

    def forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)

        return self.skip_connection(x) + h


class Downsample3D(nn.Module):
    """Downsample using nearest-neighbour resize + conv."""
    def __init__(self, channels, sample_kernel=(2, 2, 2), dims=3):
        super().__init__()
        self.channels = channels
        sk = tuple(1.0 / s for s in sample_kernel)
        self.down = nn.Upsample(scale_factor=sk, mode='nearest')
        self.conv = conv_nd(dims, channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv(self.down(x))


class Upsample3D(nn.Module):
    """Upsample using nearest-neighbour resize + conv."""
    def __init__(self, channels, sample_kernel=(2, 2, 2), dims=3):
        super().__init__()
        self.channels = channels
        self.up = nn.Upsample(scale_factor=tuple(sample_kernel), mode='nearest')
        self.conv = conv_nd(dims, channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


# ─── Main Hybrid Model ──────────────────────────────────────────────────────
class HybridDiffusionModel(nn.Module):
    """
    Complete hybrid model: Frozen MRI encoder + cross-attention denoiser.

    Args:
        image_size:   spatial dims of input patches, e.g. (64, 64, 4)
        model_channels: base channel count (default 64)
        out_channels: output channels (default 2 for learned variance)
        enc_channels: MRI encoder channel progression
        channel_mult: denoiser channel multipliers (default (1, 2, 4) for 3 levels)
        num_res_blocks: ResBlocks per level
        sample_kernel: downsampling kernel per level
        num_heads_cross_attn: attention heads for cross-attention per level
        dims: spatial dimensions (3 for 3D)
        dropout: dropout rate
        freeze_encoder: whether to freeze MRI encoder
    """

    def __init__(
        self,
        image_size=(64, 64, 4),
        model_channels=64,
        out_channels=2,
        enc_channels=(64, 128, 192, 256),
        channel_mult=(1, 2, 4),
        num_res_blocks=2,
        sample_kernel=((2, 2, 2), (2, 2, 1)),
        num_heads_cross_attn=(4, 4, 8),
        dims=3,
        dropout=0.0,
        use_scale_shift_norm=True,
        freeze_encoder=True,
        encoder_window_size=(4, 4, 4),
        encoder_num_heads=(4, 4, 8, 8),
        encoder_pool_kernel=(2, 2, 2),
    ):
        super().__init__()
        self.image_size = image_size
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.dims = dims

        time_embed_dim = model_channels * 4  # 256

        # ═══════════════════════════════════════════════════════════════════
        # 1. MRI SEMANTIC ENCODER (frozen)
        # ═══════════════════════════════════════════════════════════════════
        self.mri_encoder = MRISemanticEncoder(
            in_channels=1,
            enc_channels=enc_channels,
            global_dim=time_embed_dim,  # match time_embed_dim for addition
            dims=dims,
            num_heads=encoder_num_heads,
            window_size=encoder_window_size,
            pool_kernel=encoder_pool_kernel,
            freeze=freeze_encoder,
        )

        # ═══════════════════════════════════════════════════════════════════
        # 2. TIME + GLOBAL MRI FUSION
        # ═══════════════════════════════════════════════════════════════════
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        # Global MRI token projection to time_embed_dim (in case dims differ)
        self.global_mri_proj = nn.Sequential(
            linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # ═══════════════════════════════════════════════════════════════════
        # 3. DENOISING UNET (trainable) — 3 levels
        # ═══════════════════════════════════════════════════════════════════

        # ── Init Conv ────────────────────────────────────────────────────
        ch = int(channel_mult[0] * model_channels)  # 64
        self.init_conv = conv_nd(dims, 1, ch, 3, padding=1)

        # ── Encoder (Down) Path ──────────────────────────────────────────
        self.down_blocks = nn.ModuleList()
        self.cross_attn_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        encoder_skip_channels = [ch]  # track channels for skip connections
        num_levels = len(channel_mult)

        # MRI encoder feature channels for cross-attention K,V
        # CA1 uses f1 (64ch), CA2 uses f2 (128ch), CA3 uses f3+f4 (192+256=448ch)
        ca_context_dims = [
            enc_channels[0],                          # f₁: 64
            enc_channels[1],                          # f₂: 128
            enc_channels[2] + enc_channels[3],        # f₃+f₄: 192+256 = 448
        ]

        for level in range(num_levels):
            out_ch = int(channel_mult[level] * model_channels)

            # ResBlocks at this level
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(
                    ResBlock(
                        ch, time_embed_dim, dropout,
                        out_channels=out_ch, dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                )
                ch = out_ch

            self.down_blocks.append(blocks)
            encoder_skip_channels.append(ch)

            # Cross-attention at this level (skip level 0 to save memory)
            if level > 0:
                self.cross_attn_blocks.append(
                    CrossAttention3D(
                        query_dim=ch,
                        context_dim=ca_context_dims[level],
                        num_heads=num_heads_cross_attn[level],
                    )
                )
            else:
                self.cross_attn_blocks.append(None)

            # Downsampler (not on last level)
            if level < num_levels - 1:
                sk = sample_kernel[level]
                self.downsamplers.append(Downsample3D(ch, sk, dims=dims))
                encoder_skip_channels.append(ch)
            else:
                self.downsamplers.append(None)

        # ── Middle Block ─────────────────────────────────────────────────
        self.middle_block = nn.ModuleList([
            ResBlock(ch, time_embed_dim, dropout, out_channels=ch, dims=dims,
                     use_scale_shift_norm=use_scale_shift_norm),
            ResBlock(ch, time_embed_dim, dropout, out_channels=ch, dims=dims,
                     use_scale_shift_norm=use_scale_shift_norm),
        ])

        # ── Decoder (Up) Path ────────────────────────────────────────────
        self.up_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        for level in reversed(range(num_levels)):
            out_ch = int(channel_mult[level] * model_channels)

            # ResBlocks with skip concat
            blocks = nn.ModuleList()
            for i in range(num_res_blocks):
                skip_ch = encoder_skip_channels.pop()
                blocks.append(
                    ResBlock(
                        ch + skip_ch, time_embed_dim, dropout,
                        out_channels=out_ch, dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                )
                ch = out_ch

            self.up_blocks.append(blocks)

            # Upsampler (not on first/outermost level)
            if level > 0:
                sk = sample_kernel[level - 1]
                self.upsamplers.append(Upsample3D(ch, sk, dims=dims))
            else:
                self.upsamplers.append(None)

        # ── Output Conv ──────────────────────────────────────────────────
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, ch, out_channels, 3, padding=1)),
        )

    def forward(self, x_t, timesteps, mri_condition, y=None):
        """
        Full forward pass.

        Args:
            x_t:            noisy CT  [B, 1, D, H, W]
            timesteps:      diffusion timestep  [B]
            mri_condition:  MRI volume [B, 1, D, H, W]
            y:              unused (for API compatibility)

        Returns:
            predicted output [B, out_channels, D, H, W]
        """
        # ═══════════════════════════════════════════════════════════════════
        # Step 1: Extract MRI features (frozen, no grad)
        # ═══════════════════════════════════════════════════════════════════
        with th.no_grad():
            mri_feats = self.mri_encoder(mri_condition)
        # mri_feats: {'f1': ..., 'f2': ..., 'f3': ..., 'f4': ..., 'global': ...}

        # ═══════════════════════════════════════════════════════════════════
        # Step 2: Time + Global MRI Fusion → condition c
        # ═══════════════════════════════════════════════════════════════════
        t_emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        g_emb = self.global_mri_proj(mri_feats['global'].detach())
        emb = t_emb + g_emb  # [B, time_embed_dim]

        # ═══════════════════════════════════════════════════════════════════
        # Step 3: Denoising UNet forward
        # ═══════════════════════════════════════════════════════════════════

        # Prepare cross-attention contexts
        f3_f4 = self._concat_f3_f4(mri_feats['f3'].detach(), mri_feats['f4'].detach())
        ca_contexts = [
            mri_feats['f1'].detach(),   # CA1: f₁
            mri_feats['f2'].detach(),   # CA2: f₂
            f3_f4,                       # CA3: f₃ + f₄ concatenated along channel
        ]

        # ── Init Conv ────────────────────────────────────────────────────
        h = self.init_conv(x_t)
        hs = [h]  # skip connections

        # ── Encoder (Down) ───────────────────────────────────────────────
        for level in range(len(self.down_blocks)):
            # ResBlocks
            for block in self.down_blocks[level]:
                h = block(h, emb)

            # Cross-attention (skip if None)
            if self.cross_attn_blocks[level] is not None:
                h = self.cross_attn_blocks[level](h, ca_contexts[level])

            hs.append(h)

            # Downsample
            if self.downsamplers[level] is not None:
                h = self.downsamplers[level](h)
                hs.append(h)

        # ── Middle Block ─────────────────────────────────────────────────
        for block in self.middle_block:
            h = block(h, emb)

        # ── Decoder (Up) ────────────────────────────────────────────────
        for level_idx, blocks in enumerate(self.up_blocks):
            for block in blocks:
                skip = hs.pop()
                h = th.cat([h, skip], dim=1)
                h = block(h, emb)

            if self.upsamplers[level_idx] is not None:
                h = self.upsamplers[level_idx](h)

        # ── Output ───────────────────────────────────────────────────────
        return self.out(h)

    def _concat_f3_f4(self, f3, f4):
        """
        Concatenate f₃ and f₄ along channel dim for CA3.
        f₃ is at ¼ res, f₄ is at ⅛ res → upsample f₄ to match f₃ spatial dims.
        """
        if f3.shape[2:] != f4.shape[2:]:
            f4_up = F.interpolate(
                f4, size=f3.shape[2:], mode='trilinear', align_corners=False
            )
        else:
            f4_up = f4
        return th.cat([f3, f4_up], dim=1)  # [B, 192+256, D/4, H/4, W/4]


# ─── Wrapper that matches the original SwinVITModel API ─────────────────────
class HybridSwinVITModel(HybridDiffusionModel):
    """
    Drop-in wrapper matching the original SwinVITModel's __init__ signature
    as closely as possible, for use with existing GaussianDiffusion code.

    The key change: forward() now takes `mri_condition` instead of
    concatenated [noisy_ct, mri].
    """

    def __init__(
        self,
        image_size,
        in_channels=1,  # Now 1 (noisy CT only)
        model_channels=64,
        out_channels=2,
        num_res_blocks=2,
        attention_resolutions=(32, 16, 8),
        dropout=0,
        channel_mult=(1, 2, 4),
        conv_resample=False,
        dims=3,
        sample_kernel=None,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=(4, 4, 8),
        window_size=None,
        num_head_channels=64,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_new_attention_order=False,
        # New params for hybrid architecture
        enc_channels=(64, 128, 192, 256),
        freeze_encoder=True,
        encoder_window_size=(4, 4, 4),
        encoder_num_heads=(4, 4, 8, 8),
        encoder_pool_kernel=(2, 2, 2),
    ):
        # Extract sample_kernel list
        if sample_kernel is not None:
            sk_list = sample_kernel[0] if isinstance(sample_kernel, tuple) and \
                isinstance(sample_kernel[0], (list, tuple)) and \
                isinstance(sample_kernel[0][0], (list, tuple)) else sample_kernel
            if isinstance(sk_list, tuple) and len(sk_list) == 1:
                sk_list = sk_list[0]
        else:
            sk_list = ((2, 2, 2), (2, 2, 1))

        # For 3 levels we need 2 downsampling kernels
        if isinstance(sk_list[0], (list, tuple)):
            # Already a list of kernels
            sk = tuple(tuple(k) for k in sk_list[:len(channel_mult)-1])
        else:
            sk = (tuple(sk_list),) * (len(channel_mult) - 1)

        super().__init__(
            image_size=image_size,
            model_channels=model_channels,
            out_channels=out_channels,
            enc_channels=enc_channels,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks if isinstance(num_res_blocks, int) else num_res_blocks[0],
            sample_kernel=sk,
            num_heads_cross_attn=num_heads if isinstance(num_heads, (list, tuple)) else [num_heads]*len(channel_mult),
            dims=dims,
            dropout=dropout,
            use_scale_shift_norm=use_scale_shift_norm,
            freeze_encoder=freeze_encoder,
            encoder_window_size=encoder_window_size,
            encoder_num_heads=encoder_num_heads,
            encoder_pool_kernel=encoder_pool_kernel,
        )

        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_classes = num_classes
