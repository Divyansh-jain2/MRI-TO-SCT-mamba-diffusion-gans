"""
ControlNet-style Hybrid Diffusion Model.
Fixed Skip-Connection logic and Spatial Alignment.
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from network.util_network import (
    conv_nd,
    linear,
    zero_module,
    normalization,
    timestep_embedding,
)
from network.mri_encoder import MRISemanticEncoder

class TimestepBlock(nn.Module):
    def forward(self, x, emb):
        raise NotImplementedError

class ResBlock(TimestepBlock):
    def __init__(self, channels, emb_channels, dropout=0.0, out_channels=None, dims=3, use_scale_shift_norm=True):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )
        self.skip_connection = conv_nd(dims, channels, self.out_channels, 1) if channels != self.out_channels else nn.Identity()

    def forward(self, x, emb):
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
    def __init__(self, channels, sample_kernel=(2, 2, 1), dims=3):
        super().__init__()
        sk = tuple(1.0 / s for s in sample_kernel)
        self.down = nn.Upsample(scale_factor=sk, mode='nearest')
        self.conv = conv_nd(dims, channels, channels, 3, padding=1)
    def forward(self, x): return self.conv(self.down(x))

class Upsample3D(nn.Module):
    def __init__(self, channels, sample_kernel=(2, 2, 1), dims=3):
        super().__init__()
        self.up = nn.Upsample(scale_factor=tuple(sample_kernel), mode='nearest')
        self.conv = conv_nd(dims, channels, channels, 3, padding=1)
    def forward(self, x): return self.conv(self.up(x))

class HybridSwinVITModel(nn.Module):
    def __init__(
        self,
        image_size=(64, 64, 4),
        model_channels=64,
        out_channels=2,
        enc_channels=(64, 128, 192, 256),
        channel_mult=(1, 2, 4),
        num_res_blocks=2,
        sample_kernel=((2, 2, 1), (2, 2, 1)), # Forced sync with MRI Encoder
        dims=3,
        dropout=0.0,
        freeze_encoder=True,
        **kwargs
    ):
        super().__init__()
        self.model_channels = model_channels
        time_embed_dim = model_channels * 4

        # 1. MRI Control Encoder
        self.mri_encoder = MRISemanticEncoder(
            in_channels=1, enc_channels=enc_channels, global_dim=time_embed_dim,
            dims=dims, freeze=freeze_encoder, pool_kernel=(2,2,1)
        )

        # 2. ControlNet Zero-Convolutions
        self.control_projections = nn.ModuleList()
        for i, mult in enumerate(channel_mult):
            self.control_projections.append(
                zero_module(conv_nd(dims, enc_channels[i], model_channels * mult, 1))
            )

        # 3. Time Embedding
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # 4. Denoising UNet
        self.init_conv = conv_nd(dims, 1, model_channels, 3, padding=1)
        
        self.down_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        ch = model_channels
        self.skip_channels = [ch]

        for i, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            level_blocks = nn.ModuleList()
            n_blocks = num_res_blocks[i] if isinstance(num_res_blocks, (list, tuple)) else num_res_blocks
            for _ in range(n_blocks):
                level_blocks.append(ResBlock(ch, time_embed_dim, dropout, out_channels=out_ch))
                ch = out_ch
                self.skip_channels.append(ch)
            self.down_blocks.append(level_blocks)
            
            if i < len(channel_mult) - 1:
                sk = sample_kernel[i] if isinstance(sample_kernel[0], (list, tuple)) else sample_kernel
                self.downsamplers.append(Downsample3D(ch, sk))
                self.skip_channels.append(ch)
            else:
                self.downsamplers.append(None)

        self.middle_block = nn.Sequential(
            ResBlock(ch, time_embed_dim, dropout),
            ResBlock(ch, time_embed_dim, dropout),
        )

        self.up_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        
        for i, mult in reversed(list(enumerate(channel_mult))):
            out_ch = model_channels * mult
            level_blocks = nn.ModuleList()
            n_blocks = num_res_blocks[i] if isinstance(num_res_blocks, (list, tuple)) else num_res_blocks
            for _ in range(n_blocks):
                skip_ch = self.skip_channels.pop()
                # Triple-Concat: Current + UNet_Skip + MRI_Control
                # Decoder input channel = ch + skip_ch + out_ch (from control projection)
                level_blocks.append(ResBlock(ch + skip_ch + out_ch, time_embed_dim, dropout, out_channels=out_ch))
                ch = out_ch
            self.up_blocks.append(level_blocks)
            
            if i > 0:
                sk = sample_kernel[i-1] if isinstance(sample_kernel[0], (list, tuple)) else sample_kernel
                self.upsamplers.append(Upsample3D(ch, sk))
                # Pop the downsampler's skip channel (not used in decoder blocks)
                self.skip_channels.pop() 
            else:
                self.upsamplers.append(None)

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, ch, out_channels, 3, padding=1)),
        )

    def forward(self, x_t, timesteps, mri_condition, **kwargs):
        with th.no_grad() if self.mri_encoder._freeze_encoder else th.enable_grad():
            mri_feats = self.mri_encoder(mri_condition)
        
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        
        h = self.init_conv(x_t)
        hs = [h]
        for i, blocks in enumerate(self.down_blocks):
            for block in blocks:
                h = block(h, emb)
                hs.append(h)
            if self.downsamplers[i]:
                h = self.downsamplers[i](h)
                hs.append(h)
        
        for block in self.middle_block:
            h = block(h, emb)

        mri_skips = [mri_feats['f1'], mri_feats['f2'], mri_feats['f3']]

        for i, blocks in enumerate(self.up_blocks):
            level_idx = len(self.up_blocks) - 1 - i
            control_signal = self.control_projections[level_idx](mri_skips[level_idx])
            
            if h.shape[2:] != control_signal.shape[2:]:
                control_signal = F.interpolate(control_signal, size=h.shape[2:], mode='trilinear', align_corners=False)

            for block in blocks:
                skip = hs.pop()
                h = th.cat([h, skip, control_signal], dim=1)
                h = block(h, emb)
            
            if self.upsamplers[i]:
                h = self.upsamplers[i](h)
                hs.pop() # Remove the downsampler skip

        return self.out(h)
