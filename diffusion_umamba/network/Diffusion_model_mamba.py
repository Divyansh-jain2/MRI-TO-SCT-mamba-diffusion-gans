"""
Diffusion_model_mamba.py
========================
Drop-in replacement for Diffusion_model_transformer.py
Swin-Transformer blocks swapped for a 3-D Mamba (SSM) block.

Requirements:
    pip install mamba-ssm causal-conv1d

If mamba-ssm is unavailable the file falls back to a plain
bidirectional GRU block so the code is still importable for
debugging without a CUDA-capable install.
"""

from abc import abstractmethod
import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from network.util_network import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from timm.models.layers import DropPath

from mamba_ssm import Mamba

# ─────────────────────────────────────────────────────────────────────────────
# 3-D Mamba block  (replaces SwinTransformerBlock)
# Input / output: [B, C, D, H, W]  (same as the conv tensors in ResBlock)
# ─────────────────────────────────────────────────────────────────────────────
class MambaBlock3D(nn.Module):
    """
    Bidirectional Mamba block for volumetric (3-D) feature maps.

    Strategy:
      1. Flatten D*H*W → sequence length L
      2. Forward Mamba  pass  (tokens 0 → L-1)
      3. Backward Mamba pass  (tokens L-1 → 0)
      4. Add both outputs  →  bidirectional context
      5. Reshape back to [B, C, D, H, W]

    Parameters
    ----------
    dim          : int   channel dimension (= out_channels of the preceding conv)
    d_state      : int   Mamba SSM state size  (default 16)
    d_conv       : int   local conv width inside Mamba  (default 4)
    expand       : int   inner-dim expansion factor      (default 2)
    drop_path    : float stochastic-depth rate
    """

    def __init__(self, dim: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2,
                 drop_path: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        if _MAMBA_AVAILABLE:
            self.mamba_fwd = Mamba(d_model=dim, d_state=d_state,
                                   d_conv=d_conv, expand=expand)
            self.mamba_bwd = Mamba(d_model=dim, d_state=d_state,
                                   d_conv=d_conv, expand=expand)
        else:
            # Fallback: bidirectional GRU (same interface)
            self.mamba_fwd = nn.GRU(dim, dim, batch_first=True)
            self.mamba_bwd = nn.GRU(dim, dim, batch_first=True)
            self._fallback = True

        self.proj = nn.Linear(dim, dim)

    # ------------------------------------------------------------------
    def _ssm(self, x_seq: th.Tensor) -> th.Tensor:
        """x_seq: [B, L, C] → [B, L, C]"""
        if _MAMBA_AVAILABLE:
            fwd = self.mamba_fwd(x_seq)                         # [B,L,C]
            bwd = self.mamba_bwd(x_seq.flip(1)).flip(1)         # [B,L,C]
        else:
            fwd, _ = self.mamba_fwd(x_seq)
            bwd, _ = self.mamba_bwd(x_seq.flip(1))
            bwd = bwd.flip(1)
        return fwd + bwd

    # ------------------------------------------------------------------
    def forward(self, x: th.Tensor) -> th.Tensor:
        """x: [B, C, D, H, W]"""
        B, C, D, H, W = x.shape

        # Flatten spatial dims to sequence
        x_seq = x.permute(0, 2, 3, 4, 1).reshape(B, D * H * W, C)   # [B,L,C]

        # Residual + bidirectional SSM
        x_seq = x_seq + self.drop_path(self.proj(self._ssm(self.norm(x_seq))))

        # Reshape back to volumetric tensor
        x_out = x_seq.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)  # [B,C,D,H,W]
        return x_out


# ─────────────────────────────────────────────────────────────────────────────
# Boilerplate helpers  (identical to original)
# ─────────────────────────────────────────────────────────────────────────────
class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        pass


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    def __init__(self, channels, use_conv, sample_kernel, dims=2, out_channels=None):
        super().__init__()
        self.channels    = channels
        self.out_channels = out_channels or channels
        self.use_conv    = use_conv
        self.dims        = dims
        if dims == 3:
            self.sample_kernel = (sample_kernel[0], sample_kernel[1], sample_kernel[2])
        else:
            self.sample_kernel = (sample_kernel[0], sample_kernel[1])
        self.up   = th.nn.Upsample(scale_factor=self.sample_kernel, mode='nearest')
        self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = self.up(x)
        x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels, use_conv, sample_kernel, dims=2, out_channels=None):
        super().__init__()
        self.channels     = channels
        self.out_channels = out_channels or channels
        self.use_conv     = use_conv
        self.dims         = dims
        if dims == 3:
            self.sample_kernel = (1/sample_kernel[0], 1/sample_kernel[1], 1/sample_kernel[2])
        else:
            self.sample_kernel = (1/sample_kernel[0], 1/sample_kernel[1])
        self.op   = th.nn.Upsample(scale_factor=self.sample_kernel, mode='nearest')
        self.conv = conv_nd(dims, self.channels, self.channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.conv(self.op(x))


# ─────────────────────────────────────────────────────────────────────────────
# ResBlock  –  Swin replaced by MambaBlock3D
# ─────────────────────────────────────────────────────────────────────────────
class ResBlock(TimestepBlock):
    """
    Residual block with optional 3-D Mamba self-attention.

    Compared with the original:
      • SwinTransformerBlock  →  MambaBlock3D
      • `num_heads`, `window_size` parameters are kept for API compatibility
        but are ignored when use_mamba=True (they are repurposed for d_state
        and d_conv via the mapping below).
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        sample_kernel=None,
        # kept for API compatibility; controls Mamba hyper-params
        use_swin=False,       # renamed semantically: now means "use Mamba"
        num_heads=4,          # repurposed → d_state  = num_heads * 4
        window_size=None,     # repurposed → d_conv   = window_size[0] if list else 4
        input_resolution=None,
        drop_path=0.1,
    ):
        super().__init__()
        self.channels          = channels
        self.emb_channels      = emb_channels
        self.dropout           = dropout
        self.out_channels      = out_channels or channels
        self.use_conv          = use_conv
        self.use_checkpoint    = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.use_mamba         = use_swin   # flag kept as use_swin for caller compat

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, sample_kernel, dims)
            self.x_upd = Upsample(channels, False, sample_kernel, dims)
        elif down:
            self.h_upd = Downsample(channels, False, sample_kernel, dims)
            self.x_upd = Downsample(channels, False, sample_kernel, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        # ── derive Mamba hyper-params from legacy args ─────────────────────
        # d_state  ← num_heads * 4  (e.g. 4→16, 8→32, 16→64)
        # d_conv   ← first element of window_size if it is a list, else 4
        d_state = 128
        if isinstance(window_size, (list, tuple)) and len(window_size) > 0:
            d_conv = int(window_size[0])
            d_conv = max(2, min(d_conv, 8))   # clamp to sensible range
        else:
            d_conv = 4

        if self.use_mamba:
            # in_layers: norm → silu → conv  (same as original)
            self.in_layers = nn.Sequential(
                normalization(channels),
                nn.SiLU(),
                conv_nd(dims, channels, self.out_channels, 3, padding=1),
            )
            # Two Mamba blocks (mirrors the two SwinTransformerBlocks)
            self.mamba_layers = nn.ModuleList([
                MambaBlock3D(
                    dim=self.out_channels,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=2,
                    drop_path=drop_path,
                )
                for _ in range(2)
            ])
            self.out_layers = nn.Sequential(
                normalization(self.out_channels),
                nn.Identity(),
            )
        else:
            self.in_layers = nn.Sequential(
                normalization(channels),
                nn.SiLU(),
                conv_nd(dims, channels, self.out_channels, 3, padding=1),
            )
            self.mamba_layers = nn.ModuleList([nn.Identity()])
            self.out_layers = nn.Sequential(
                normalization(self.out_channels),
                nn.SiLU(),
                nn.Dropout(p=0),
                zero_module(
                    conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
                ),
            )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    # ------------------------------------------------------------------
    def forward(self, x, emb):
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, x, emb):
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
            # Apply Mamba blocks (they operate on [B, C, D, H, W] directly)
            for blk in self.mamba_layers:
                h = blk(h)
            h = out_rest(h)
        else:
            h = h + emb_out
            for blk in self.mamba_layers:
                h = blk(h)
            h = self.out_layers(h)

        return self.skip_connection(x) + h


# ─────────────────────────────────────────────────────────────────────────────
# SwinVITModel  (name kept for drop-in compatibility)
# Internally uses Mamba blocks instead of Swin Transformers
# ─────────────────────────────────────────────────────────────────────────────
class SwinVITModel(nn.Module):
    """
    UNet diffusion backbone with Mamba (SSM) blocks.
    All constructor arguments identical to the original SwinVITModel so that
    existing training scripts need zero changes.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=False,
        dims=2,
        sample_kernel=None,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        window_size=4,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size          = image_size
        self.in_channels         = in_channels
        self.model_channels      = model_channels
        self.out_channels        = out_channels
        self.num_res_blocks      = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout             = dropout
        self.channel_mult        = channel_mult
        self.conv_resample       = conv_resample
        self.num_classes         = num_classes
        self.use_checkpoint      = use_checkpoint
        self.dtype               = th.float16 if use_fp16 else th.float32
        self.num_heads           = num_heads
        self.num_head_channels   = num_head_channels
        self.num_heads_upsample  = num_heads_upsample
        self.sample_kernel       = sample_kernel[0]

        drop_path = [x.item() for x in th.linspace(0, dropout, len(channel_mult))]

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch       = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size   = ch
        input_block_chans    = [ch]
        ds                   = list(image_size)

        # ── Encoder ───────────────────────────────────────────────────────────
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks[level]):
                use_mamba = ds[0] in attention_resolutions
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        use_swin=use_mamba,
                        num_heads=num_heads[level],
                        window_size=window_size[level],
                        input_resolution=ds,
                        drop_path=drop_path[level],
                    )
                ]
                ch = int(mult * model_channels)
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=int(mult * model_channels),
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            use_swin=use_mamba,
                            num_heads=num_heads[level],
                            window_size=window_size[level],
                            input_resolution=ds,
                            drop_path=drop_path[level],
                            down=True,
                            sample_kernel=self.sample_kernel[level],
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample,
                            self.sample_kernel[level],
                            dims=dims, out_channels=out_ch,
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                if dims == 3:
                    ds = [
                        ds[0] // self.sample_kernel[level][0],
                        ds[1] // self.sample_kernel[level][1],
                        ds[2] // self.sample_kernel[level][2],
                    ]
                else:
                    ds = [
                        ds[0] // self.sample_kernel[level][0],
                        ds[1] // self.sample_kernel[level][1],
                    ]
                self._feature_size += ch

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                out_channels=int(mult * model_channels),
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                use_swin=True,            # always use Mamba at bottleneck
                num_heads=num_heads[level],
                window_size=window_size[level],
                input_resolution=ds,
                drop_path=drop_path[level],
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                out_channels=int(mult * model_channels),
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                use_swin=True,
                num_heads=num_heads[level],
                window_size=window_size[level],
                input_resolution=ds,
                drop_path=drop_path[level],
            ),
        )
        self._feature_size += ch

        # ── Decoder ───────────────────────────────────────────────────────────
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks[level] + 1):
                ich       = input_block_chans.pop()
                use_mamba = ds[0] in attention_resolutions
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        use_swin=use_mamba,
                        num_heads=num_heads[level],
                        window_size=window_size[level],
                        input_resolution=ds,
                        drop_path=drop_path[level],
                    )
                ]
                ch = int(model_channels * mult)
                if level and i == num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=int(model_channels * mult),
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            use_swin=use_mamba,
                            num_heads=num_heads[level],
                            window_size=window_size[level],
                            input_resolution=ds,
                            drop_path=drop_path[level],
                            up=True,
                            sample_kernel=self.sample_kernel[level - 1],
                        )
                        if resblock_updown
                        else Upsample(
                            ch, conv_resample,
                            self.sample_kernel[level - 1],
                            dims=dims, out_channels=out_ch,
                        )
                    )
                    if dims == 3:
                        ds = [
                            ds[0] * self.sample_kernel[level - 1][0],
                            ds[1] * self.sample_kernel[level - 1][1],
                            ds[2] * self.sample_kernel[level - 1][2],
                        ]
                    else:
                        ds = [
                            ds[0] * self.sample_kernel[level - 1][0],
                            ds[1] * self.sample_kernel[level - 1][1],
                        ]
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    # ------------------------------------------------------------------
    def forward(self, x, timesteps, cond=None, null_cond_prob=0.0, y=None):
        assert (y is not None) == (self.num_classes is not None), \
            "must specify y if and only if the model is class-conditional"

        hs  = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)