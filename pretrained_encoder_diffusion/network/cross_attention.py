"""
CrossAttention3D — Multi-head cross-attention for 3D volumetric features.

Used to inject MRI semantic encoder features (K,V) into the denoising UNet
features (Q) at matching spatial resolutions.
"""

import torch
import torch.nn as nn


class CrossAttention3D(nn.Module):
    """
    Multi-head cross-attention between denoiser features (Q) and
    MRI encoder features (K, V).

    Operates on flattened 3D spatial dims:
        Q from denoiser:  [B, C_q, D, H, W]  →  [B, D*H*W, C_q]
        KV from encoder:  [B, C_kv, D', H', W'] → [B, D'*H'*W', C_kv]

    Handles mismatched channels via learned linear projections.
    """

    def __init__(self, query_dim, context_dim, num_heads=4, head_dim=None):
        """
        Args:
            query_dim:   channel dimension of Q (denoiser features)
            context_dim: channel dimension of K,V (MRI encoder features)
            num_heads:   number of attention heads
            head_dim:    per-head dimension (default: query_dim // num_heads)
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim or (query_dim // num_heads)
        inner_dim = self.num_heads * self.head_dim
        self.scale = self.head_dim ** -0.5

        # Pre-norm
        self.norm_q = nn.LayerNorm(query_dim)
        self.norm_kv = nn.LayerNorm(context_dim)

        # Projections
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        # Output projection — small normal init so cross-attn contributes from step 1
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim, bias=True),
            nn.Dropout(0.0),
        )
        # Small normal init (NOT zero) — allows MRI conditioning signal to flow immediately
        nn.init.normal_(self.to_out[0].weight, std=0.02)
        nn.init.zeros_(self.to_out[0].bias)

    def forward(self, x, context, x_spatial_shape=None):
        """
        Args:
            x:       denoiser features  [B, C_q, D, H, W] (5D)
                     or already flattened [B, N, C_q] (3D)
            context: MRI encoder features [B, C_kv, D', H', W'] (5D)
                     or already flattened  [B, M, C_kv] (3D)
            x_spatial_shape: tuple (D, H, W) if x is passed as 3D — needed
                             to reshape back. If x is 5D this is inferred.

        Returns:
            out: same shape as input x (residual added)
        """
        # ── Handle 5D (volumetric) inputs ────────────────────────────────
        reshape_back = False
        if x.dim() == 5:
            B, C, D, H, W = x.shape
            x_spatial_shape = (D, H, W)
            x = x.reshape(B, C, -1).permute(0, 2, 1)       # [B, N, C]
            reshape_back = True

        if context.dim() == 5:
            B_c, C_c, Dc, Hc, Wc = context.shape
            context = context.reshape(B_c, C_c, -1).permute(0, 2, 1)  # [B, M, C_kv]

        # ── Attention ────────────────────────────────────────────────────
        B, N, _ = x.shape
        h = self.num_heads
        d = self.head_dim

        q = self.to_q(self.norm_q(x))                       # [B, N, inner]
        k = self.to_k(self.norm_kv(context))                 # [B, M, inner]
        v = self.to_v(self.norm_kv(context))                 # [B, M, inner]

        M = k.shape[1]

        # Reshape to multi-head: [B, seq, h*d] → [B, h, seq, d]
        q = q.view(B, N, h, d).permute(0, 2, 1, 3)          # [B, h, N, d]
        k = k.view(B, M, h, d).permute(0, 2, 1, 3)          # [B, h, M, d]
        v = v.view(B, M, h, d).permute(0, 2, 1, 3)          # [B, h, M, d]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale       # [B, h, N, M]
        attn = attn.softmax(dim=-1)

        out = attn @ v                                       # [B, h, N, d]
        # Reshape back: [B, h, N, d] → [B, N, h*d]
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, h * d)
        out = self.to_out(out)                               # [B, N, C_q]

        # Residual connection
        out = out + x

        # ── Reshape back to 5D if needed ─────────────────────────────────
        if reshape_back:
            D, H, W = x_spatial_shape
            out = out.permute(0, 2, 1).reshape(B, -1, D, H, W)

        return out
