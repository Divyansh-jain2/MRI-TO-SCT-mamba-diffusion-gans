"""
diffusion_mamba_models.py
=========================
Architecture-only file. Contains every class and loss utility needed to
build and run DiffusionUMamba. No training loop, no data loading.

Exported symbols
----------------
  Building blocks:
    ConvNormAct                  3-D Conv -> GroupNorm -> SiLU
    ResConvBlock                 two-layer residual conv block
    MambaBlock3D                 bidirectional Mamba over 3-D spatial volume
    AdaLN                        adaptive LayerNorm conditioned on t_emb
    FiLMProjection               Feature-wise Linear Modulation from t_emb
    UMambaBlockTime              full upgraded U-Mamba block (core unit)
    AttentionGate                attention-gated skip connection
    SinusoidalPositionEmbeddings

  Main model:
    DiffusionUMamba              noise-prediction U-Net for MRI to sCT DDPM

  Loss utilities (imported by main_diffusionUmamba.py):
    tissue_weighted_l1           L1 upweighted at bone / air voxels
    freq_loss                    L1 in 3-D FFT magnitude spectrum

  Inference helper:
    guided_sample                classifier-free guidance sampling

Improvements over the Swin-UNet / baseline forward-only Mamba
--------------------------------------------------------------
  1.  FiLM conditioning       each block predicts (gamma, beta) from t_emb
  2.  Bidirectional Mamba     forward + backward scan summed
  3.  Parallel depthwise conv runs alongside Mamba for local texture
  4.  Pre-LayerNorm           applied before Mamba for training stability
  5.  AdaLN output norm       replaces InstanceNorm; timestep-conditioned
  6.  Attention-gated skips   all three decoder skip connections are gated
  7.  Deep supervision        aux output heads at dec3 + dec2
  8.  Tissue-weighted L1      bone/air voxels upweighted in the loss
  9.  Frequency-domain loss   L1 in FFT magnitude spectrum
  10. CFG support             condition dropout + guided_sample() helper
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Mamba import — GRU fallback keeps the file runnable without mamba_ssm
# ---------------------------------------------------------------------------
try:
    from mamba_ssm import Mamba
    _MAMBA_AVAILABLE = True
except ImportError:
    _MAMBA_AVAILABLE = False
    print(
        "[diffusion_mamba_models] WARNING: mamba_ssm not found — "
        "using bidirectional GRU fallback."
    )


# ===========================================================================
# Section 1 — Loss utilities
# ===========================================================================

def tissue_weighted_l1(
    pred:      torch.Tensor,
    target:    torch.Tensor,
    condition: torch.Tensor,
    bone_w:    float = 5.0,
    air_w:     float = 2.0,
) -> torch.Tensor:
    """
    Voxel-weighted L1 loss with bone and air regions upweighted.

    Bone and air account for the vast majority of clinical error in
    brain sCT (dosimetry, attenuation correction) but occupy only ~5 %
    and ~10 % of the volume respectively. Upweighting them forces the
    diffusion model to prioritise accuracy where it matters most.

    Proxy masks are derived from the MRI condition (no extra labels):
        bone proxy : condition >  +0.60   (bright in T1 -> skull)
        air  proxy : condition <  -0.80   (dark  in T1 -> air cavities)

    Args:
        pred, target : (B, 1, D, H, W)  in [-1, 1]
        condition    : (B, 1, D, H, W)  MRI input in [-1, 1]
        bone_w       : loss weight for bone voxels  (default 5.0)
        air_w        : loss weight for air  voxels  (default 2.0)

    Returns:
        Scalar weighted L1 loss.
    """
    bone_mask = (condition >  0.60).float()
    air_mask  = (condition < -0.80).float()
    weight    = 1.0 + (bone_w - 1.0) * bone_mask + (air_w - 1.0) * air_mask
    return (F.l1_loss(pred, target, reduction='none') * weight).mean()


def freq_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    L1 loss in the 3-D FFT magnitude spectrum.

    Plain L1 in pixel-space penalises low-frequency errors (global
    contrast) more than high-frequency ones (bone edges, cortical
    boundaries). Adding an FFT-magnitude L1 term balances this and
    consistently improves bone sharpness with negligible overhead.

    Args:
        pred, target : (B, 1, D, H, W)

    Returns:
        Scalar frequency-domain L1 loss.
    """
    p_fft = torch.fft.fftn(pred,   dim=(-3, -2, -1)).abs()
    t_fft = torch.fft.fftn(target, dim=(-3, -2, -1)).abs()
    return F.l1_loss(p_fft, t_fft)


# ===========================================================================
# Section 2 — Primitive building blocks
# ===========================================================================

class ConvNormAct(nn.Module):
    """
    3-D Conv -> GroupNorm -> SiLU.

    Used for the stem, downsampling transitions, and decoder projections.
    GroupNorm with groups = max(1, out_ch // 8) is well-suited to the
    small batch sizes typical in 3-D medical imaging.

    Args:
        in_ch  : input channels
        out_ch : output channels
        stride : 1 (default) or 2 for spatial downsampling
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(
                in_ch, out_ch, kernel_size=3,
                stride=stride, padding=1, bias=False,
            ),
            nn.GroupNorm(max(1, out_ch // 8), out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResConvBlock(nn.Module):
    """
    Two-layer residual convolution block.

    Conv3d -> GroupNorm -> SiLU -> Conv3d -> GroupNorm -> residual + SiLU.
    Used inside UMambaBlockTime for spatial pre-processing before Mamba.
    """

    def __init__(self, ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(ch, ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(max(1, ch // 8), ch),
            nn.SiLU(inplace=True),
            nn.Conv3d(ch, ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(max(1, ch // 8), ch),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


# ===========================================================================
# Section 3 — Bidirectional Mamba block (with GRU fallback)
# ===========================================================================

class _GRUMamba(nn.Module):
    """
    Bidirectional GRU used as a drop-in fallback when mamba_ssm is not
    installed. Matches the (B, L, C) -> (B, L, C) interface of Mamba.
    """

    def __init__(self, d_model: int, **_):
        super().__init__()
        self.gru  = nn.GRU(d_model, d_model, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.proj(out)


class MambaBlock3D(nn.Module):
    """
    Bidirectional Mamba scan over a flattened 3-D spatial volume.

    The D x H x W spatial dimensions are flattened into a single sequence
    of length L = D*H*W.  Two separate Mamba modules scan the sequence
    forward and backward; their outputs are summed.

    This recovers long-range context at both ends of the sequence that is
    lost in the forward-only baseline — especially important for the first
    and last ~20 axial slices of a 96-slice brain volume.

    Shape contract:
        input  : (B, C, D, H, W)
        output : (B, C, D, H, W)   same spatial size

    Args:
        ch      : number of channels (= Mamba d_model)
        d_state : SSM state dimension (Mamba-specific, default 16)
    """

    def __init__(self, ch: int, d_state: int = 16):
        super().__init__()
        if _MAMBA_AVAILABLE:
            self.fwd = Mamba(d_model=ch, d_state=d_state, d_conv=4, expand=2)
            self.bwd = Mamba(d_model=ch, d_state=d_state, d_conv=4, expand=2)
        else:
            self.fwd = _GRUMamba(ch)
            self.bwd = _GRUMamba(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        seq     = x.flatten(2).permute(0, 2, 1)         # (B, L, C)
        fwd_out = self.fwd(seq)                           # (B, L, C) forward
        bwd_out = self.bwd(seq.flip(1)).flip(1)           # (B, L, C) backward
        out     = fwd_out + bwd_out                       # element-wise sum
        return out.permute(0, 2, 1).view(B, C, D, H, W)


# ===========================================================================
# Section 4 — Adaptive Layer Normalisation (AdaLN)
# ===========================================================================

class AdaLN(nn.Module):
    """
    Adaptive LayerNorm conditioned on an external embedding vector.

    Replaces InstanceNorm + additive time injection from the baseline.
    Predicts per-channel (gamma, beta) from t_emb so that normalisation
    parameters vary with the diffusion timestep.

    Zero-initialised projection so the block starts as an identity
    transform at the beginning of training (stable early dynamics).

    Used as the output normalisation layer in every UMambaBlockTime.

    Args:
        ch      : feature map channel dimension
        emb_dim : size of the conditioning embedding vector
    """

    def __init__(self, ch: int, emb_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(ch, elementwise_affine=False)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, 2 * ch),
        )
        # Zero-init -> identity transform at epoch 0
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        # x   : (B, C, D, H, W)
        # emb : (B, emb_dim)
        gamma, beta = self.proj(emb).chunk(2, dim=-1)    # each (B, C)
        gamma = gamma[:, :, None, None, None]             # (B, C, 1, 1, 1)
        beta  = beta[:,  :, None, None, None]

        # LayerNorm requires channel dim last; permute, norm, permute back
        B, C, D, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).contiguous()        # (B, D, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()        # (B, C, D, H, W)
        return x * (1.0 + gamma) + beta


# ===========================================================================
# Section 5 — FiLM conditioning
# ===========================================================================

class FiLMProjection(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM).

    Maps t_emb -> (gamma, beta) and applies:
        x' = x * (1 + gamma) + beta

    Strictly more expressive than plain additive injection (x = x + bias)
    because it provides per-channel multiplicative scaling. Zero-init
    keeps it at identity at the start of training.

    Args:
        emb_dim : size of the incoming time embedding
        ch      : channel dimension of the feature map to modulate
    """

    def __init__(self, emb_dim: int, ch: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, ch * 2),
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        # x   : (B, C, D, H, W)
        # emb : (B, emb_dim)
        gamma, beta = self.proj(emb).chunk(2, dim=-1)    # each (B, C)
        gamma = gamma[:, :, None, None, None]
        beta  = beta[:,  :, None, None, None]
        return x * (1.0 + gamma) + beta


# ===========================================================================
# Section 6 — Core upgraded block: UMambaBlockTime
# ===========================================================================

class UMambaBlockTime(nn.Module):
    """
    Upgraded U-Mamba block conditioned on the diffusion timestep.

    Processing order inside each block
    -----------------------------------
    1. FiLM      : modulate incoming feature map with (gamma, beta) from t_emb
    2. ResConv   : two-layer residual spatial convolution
    3a. Mamba branch  : pre-LN -> bidirectional MambaBlock3D  (global context)
    3b. DW-conv branch: 3x3x3 depthwise + 1x1 pointwise conv  (local texture)
       Both branches output (B, C, D, H, W) and are summed with the residual:
           x = x + mamba_out + local_out
    4. AdaLN     : adaptive LayerNorm normalisation conditioned on t_emb

    Why parallel branches (not series)?
    ------------------------------------
    Mamba excels at long-range sequential dependencies.
    Depthwise conv excels at local 3-D texture.
    Running them in parallel lets each specialise independently, while the
    shared residual connection prevents either branch from dominating.

    Args:
        ch           : number of channels
        time_emb_dim : dimension of the shared diffusion time embedding
        d_state      : Mamba SSM state dimension (default 16)
    """

    def __init__(self, ch: int, time_emb_dim: int, d_state: int = 16):
        super().__init__()

        # 1. FiLM timestep injection (applied first, before any conv)
        self.film      = FiLMProjection(time_emb_dim, ch)

        # 2. Residual spatial convolution
        self.res_conv  = ResConvBlock(ch)

        # 3a. Mamba branch — pre-LayerNorm stabilises the sequence scan
        self.pre_norm  = nn.LayerNorm(ch)
        self.mamba     = MambaBlock3D(ch, d_state=d_state)

        # 3b. Parallel local depthwise-pointwise conv branch
        self.local_dw  = nn.Conv3d(ch, ch, kernel_size=3, padding=1,
                                   groups=ch, bias=False)
        self.local_pw  = nn.Conv3d(ch, ch, kernel_size=1, bias=False)
        self.local_gn  = nn.GroupNorm(max(1, ch // 8), ch)
        self.local_act = nn.SiLU(inplace=True)

        # 4. AdaLN output normalisation
        self.ada_ln    = AdaLN(ch, time_emb_dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # x     : (B, C, D, H, W)
        # t_emb : (B, time_emb_dim)

        # 1. FiLM: timestep-conditioned feature modulation
        x = self.film(x, t_emb)

        # 2. Residual spatial conv
        x = self.res_conv(x)

        # 3a. Mamba branch (with pre-norm over the channel dimension)
        B, C, D, H, W = x.shape
        seq      = x.flatten(2).permute(0, 2, 1)          # (B, L, C)
        seq_ln   = self.pre_norm(seq)                      # LayerNorm over C
        mamba_in = seq_ln.permute(0, 2, 1).view(B, C, D, H, W)
        mamba_out = self.mamba(mamba_in)                   # (B, C, D, H, W)

        # 3b. Local depthwise-conv branch
        local_out = self.local_act(
            self.local_gn(self.local_pw(self.local_dw(x)))
        )

        # Fuse: residual + both parallel branches
        x = x + mamba_out + local_out

        # 4. AdaLN output normalisation
        return self.ada_ln(x, t_emb)


# ===========================================================================
# Section 7 — Attention gate for decoder skip connections
# ===========================================================================

class AttentionGate(nn.Module):
    """
    Attention U-Net gating mechanism for encoder skip connections.

    Learns a spatial attention map from the decoder query g and the
    encoder skip feature x. Suppresses encoder features that are
    irrelevant to the current decoder prediction — e.g. soft-tissue MRI
    signal that has no correspondence in CT HU space.

    Architecture:
        gate = sigmoid( psi( relu( Wg(g) + Wx(x) ) ) )
        output = x * gate

    Args:
        ch : channel count (same for both g and x)

    Forward:
        g      : (B, ch, D, H, W)  decoder feature (query)
        x      : (B, ch, D, H, W)  encoder skip    (key/value)
        returns: (B, ch, D, H, W)  spatially gated encoder feature
    """

    def __init__(self, ch: int):
        super().__init__()
        mid       = max(1, ch // 2)
        self.Wg   = nn.Conv3d(ch, mid, kernel_size=1, bias=True)
        self.Wx   = nn.Conv3d(ch, mid, kernel_size=1, bias=False)
        self.psi  = nn.Conv3d(mid, 1,  kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        gate = self.relu(self.Wg(g) + self.Wx(x))         # (B, mid, D, H, W)
        gate = torch.sigmoid(self.psi(gate))               # (B,   1, D, H, W)
        return x * gate                                    # broadcast over C


# ===========================================================================
# Section 8 — Sinusoidal time embedding
# ===========================================================================

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal embedding for diffusion timesteps.

    Uses 256 frequency components (vs 64 in the baseline) to maintain
    fine-grained resolution across all 1000 diffusion steps.

    Args:
        dim : embedding dimension (must be even, default 256)
    """

    def __init__(self, dim: int = 256):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device   = time.device
        half_dim = self.dim // 2
        emb      = math.log(10000) / (half_dim - 1)
        emb      = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb      = time[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)   # (B, dim)


# ===========================================================================
# Section 9 — Main model: DiffusionUMamba
# ===========================================================================

class DiffusionUMamba(nn.Module):
    """
    Upgraded noise-prediction U-Net for MRI to synthetic CT DDPM.

    Architecture summary
    --------------------
    Encoder : stem -> enc1 -> down1 -> enc2 -> down2 -> enc3 -> down3 -> enc4
              each encN is a UMambaBlockTime (FiLM + BiMamba || DW-conv + AdaLN)

    Decoder : up3 -> att3(gate) -> dec3 -> proj3
              up2 -> att2(gate) -> dec2 -> proj2
              up1 -> att1(gate) -> dec1 -> proj1
              head (Conv3d 1x1, no activation)

    All skip connections pass through AttentionGate before concatenation.

    Optional features
    -----------------
    cfg_dropout_p    : probability of zeroing the MRI condition during training
                       (Classifier-Free Guidance dropout). Use guided_sample()
                       at inference for guidance-weighted output.
    deep_supervision : attach aux output heads at dec3 (1/4 res) and dec2
                       (1/2 res). Call model.deep_sup_loss(target) in the
                       train loop immediately after forward() to get the
                       auxiliary scalar loss.

    Forward signature — unchanged from baseline
    -------------------------------------------
        pred = model(x, t)
        x : (B, in_ch, D, H, W)  — torch.cat([noisy_CT, MRI_cond], dim=1)
        t : (B,)                  — integer diffusion timestep indices

    Args:
        in_ch            : input channels  (2 = noisy_CT + MRI_cond)
        out_ch           : output channels (2 = mean + variance with
                           learn_sigma=True, else 1)
        base_ch          : base channel width       (default 64)
        time_emb_dim     : projected time emb dim   (default 256)
        d_state          : Mamba SSM state dim       (default 16)
        cfg_dropout_p    : CFG condition dropout     (default 0.10)
        deep_supervision : enable auxiliary heads    (default True)
    """

    def __init__(
        self,
        in_ch:            int   = 2,
        out_ch:           int   = 2,
        base_ch:          int   = 64,
        time_emb_dim:     int   = 256,
        d_state:          int   = 16,
        cfg_dropout_p:    float = 0.10,
        deep_supervision: bool  = True,
    ):
        super().__init__()
        self.cfg_dropout_p    = cfg_dropout_p
        self.deep_supervision = deep_supervision
        ch = base_ch

        # ── Time embedding: sinusoidal -> 2-layer MLP ────────────────────────
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(256),
            nn.Linear(256, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # ── Encoder ─────────────────────────────────────────────────────────
        self.stem  = ConvNormAct(in_ch, ch)                         # full res
        self.enc1  = UMambaBlockTime(ch,   time_emb_dim, d_state)   # full res
        self.down1 = ConvNormAct(ch,   ch*2, stride=2)
        self.enc2  = UMambaBlockTime(ch*2, time_emb_dim, d_state)   # 1/2 res
        self.down2 = ConvNormAct(ch*2, ch*4, stride=2)
        self.enc3  = UMambaBlockTime(ch*4, time_emb_dim, d_state)   # 1/4 res
        self.down3 = ConvNormAct(ch*4, ch*8, stride=2)
        self.enc4  = UMambaBlockTime(ch*8, time_emb_dim, d_state)   # 1/8 bottleneck

        # ── Decoder up-convolutions ──────────────────────────────────────────
        self.up3 = nn.ConvTranspose3d(ch*8, ch*4, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(ch*4, ch*2, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose3d(ch*2, ch,   kernel_size=2, stride=2)

        # ── Attention gates (one per skip connection) ────────────────────────
        self.att3 = AttentionGate(ch*4)
        self.att2 = AttentionGate(ch*2)
        self.att1 = AttentionGate(ch)

        # ── Decoder blocks ───────────────────────────────────────────────────
        # Input channels = upsampled + gated_skip, so channel count doubles
        self.dec3 = UMambaBlockTime(ch*8, time_emb_dim, d_state)    # ch*4 + ch*4
        self.dec2 = UMambaBlockTime(ch*4, time_emb_dim, d_state)    # ch*2 + ch*2
        self.dec1 = UMambaBlockTime(ch*2, time_emb_dim, d_state)    # ch   + ch

        # ── Decoder projections (restore single-scale channels) ──────────────
        self.proj3 = ConvNormAct(ch*8, ch*4)
        self.proj2 = ConvNormAct(ch*4, ch*2)
        self.proj1 = ConvNormAct(ch*2, ch)

        # ── Final output head ────────────────────────────────────────────────
        # No activation: DDPM noise predictors output unconstrained values
        self.head = nn.Conv3d(ch, out_ch, kernel_size=1)

        # ── Deep supervision auxiliary heads ─────────────────────────────────
        if deep_supervision:
            self.aux_head3 = nn.Conv3d(ch*4, out_ch, kernel_size=1)  # 1/4 res
            self.aux_head2 = nn.Conv3d(ch*2, out_ch, kernel_size=1)  # 1/2 res

        # Tensors populated during forward(); consumed by deep_sup_loss()
        self.aux3: torch.Tensor | None = None
        self.aux2: torch.Tensor | None = None

    # -----------------------------------------------------------------------
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, in_ch, D, H, W)
                torch.cat([noisy_CT, MRI_condition], dim=1)
            t : (B,)  integer diffusion timestep indices

        Returns:
            (B, out_ch, D, H, W)  predicted noise (or x0 if predict_xstart)
        """
        # ── Classifier-free guidance condition dropout ────────────────────────
        # x[:,0]  = noisy CT       (never zeroed)
        # x[:,1:] = MRI condition  (zeroed with probability cfg_dropout_p)
        if self.training and self.cfg_dropout_p > 0.0:
            B         = x.shape[0]
            drop_mask = torch.rand(B, device=x.device) < self.cfg_dropout_p
            x         = x.clone()            # avoid in-place on the compute graph
            x[drop_mask, 1:] = 0.0

        # ── Time embedding ───────────────────────────────────────────────────
        t_emb = self.time_mlp(t)             # (B, time_emb_dim)

        # ── Encoder ─────────────────────────────────────────────────────────
        x  = self.stem(x)
        e1 = self.enc1(x,  t_emb)            # full res   skip feature

        e2 = self.down1(e1)
        e2 = self.enc2(e2, t_emb)            # 1/2  res   skip feature

        e3 = self.down2(e2)
        e3 = self.enc3(e3, t_emb)            # 1/4  res   skip feature

        e4 = self.down3(e3)
        e4 = self.enc4(e4, t_emb)            # 1/8  res   bottleneck

        # ── Decoder with attention-gated skip connections ────────────────────
        # Stage 3: 1/8 -> 1/4
        d3_up = self.up3(e4)                 # (B, ch*4, 1/4 res)
        s3    = self.att3(g=d3_up, x=e3)    # attention-gated enc3 skip
        d3    = self.dec3(torch.cat([d3_up, s3], dim=1), t_emb)
        d3    = self.proj3(d3)               # (B, ch*4, 1/4 res)

        # Stage 2: 1/4 -> 1/2
        d2_up = self.up2(d3)                 # (B, ch*2, 1/2 res)
        s2    = self.att2(g=d2_up, x=e2)    # attention-gated enc2 skip
        d2    = self.dec2(torch.cat([d2_up, s2], dim=1), t_emb)
        d2    = self.proj2(d2)               # (B, ch*2, 1/2 res)

        # Stage 1: 1/2 -> full
        d1_up = self.up1(d2)                 # (B, ch,   full res)
        s1    = self.att1(g=d1_up, x=e1)    # attention-gated enc1 skip
        d1    = self.dec1(torch.cat([d1_up, s1], dim=1), t_emb)
        d1    = self.proj1(d1)               # (B, ch,   full res)

        # ── Deep supervision: store aux predictions for deep_sup_loss() ──────
        if self.deep_supervision:
            self.aux3 = self.aux_head3(d3)   # (B, out_ch, 1/4 res)
            self.aux2 = self.aux_head2(d2)   # (B, out_ch, 1/2 res)

        return self.head(d1)                 # (B, out_ch, full res)

    # -----------------------------------------------------------------------
    def deep_sup_loss(
        self,
        target:  torch.Tensor,
        w3:      float = 0.30,
        w2:      float = 0.50,
        loss_fn         = F.l1_loss,
    ) -> torch.Tensor:
        """
        Weighted auxiliary loss from the two deep-supervision heads.

        Must be called immediately after forward() in the same train step.
        Both aux tensors (stored at 1/4 and 1/2 resolution) are trilinearly
        upsampled to the full spatial resolution of target before the loss.

        Args:
            target  : (B, out_ch, D, H, W)  ground-truth CT at full resolution
            w3      : weight for the dec3 aux head   (default 0.30)
            w2      : weight for the dec2 aux head   (default 0.50)
            loss_fn : callable — default F.l1_loss

        Returns:
            Scalar combined auxiliary loss.

        Raises:
            RuntimeError if deep_supervision=False or forward() not called yet.
        """
        if not self.deep_supervision:
            return torch.tensor(0.0, device=target.device)
        if self.aux3 is None or self.aux2 is None:
            raise RuntimeError(
                "deep_sup_loss() called before forward(). "
                "Run a forward pass first to populate aux3 / aux2."
            )
        # Keep only channel 0: mean prediction
        # (channel 1 is the variance head used internally by the diffusion lib)
        tgt = target[:, :1]                              # (B, 1, D, H, W)

        a3 = F.interpolate(
            self.aux3[:, :1], size=tgt.shape[2:],
            mode='trilinear', align_corners=False,
        )
        a2 = F.interpolate(
            self.aux2[:, :1], size=tgt.shape[2:],
            mode='trilinear', align_corners=False,
        )
        return w3 * loss_fn(a3, tgt) + w2 * loss_fn(a2, tgt)


# ===========================================================================
# Section 10 — Classifier-free guidance inference helper
# ===========================================================================

@torch.no_grad()
def guided_sample(
    diffusion,
    model:     DiffusionUMamba,
    condition: torch.Tensor,
    out_shape: tuple,
    w:         float = 3.0,
    clip:      bool  = True,
) -> torch.Tensor:
    """
    Classifier-free guidance (CFG) sampling.

    At each denoising step runs two model forward passes:
        1. Conditional   — real MRI condition concatenated
        2. Unconditional — zeroed condition (null) concatenated

    Then combines the noise predictions:
        noise_guided = noise_uncond + w * (noise_cond - noise_uncond)

    A higher w sharpens bone edges and HU accuracy at the cost of
    some diversity. Typical useful range: w in [2, 7].

    Prerequisite:
        The model must have been trained with cfg_dropout_p > 0 so it has
        seen null-conditioned inputs and learned an unconditional distribution.

    Args:
        diffusion  : GaussianDiffusion object (from diffusion.Create_diffusion)
        model      : DiffusionUMamba in eval mode, on the correct device
        condition  : (B, 1, D, H, W)  MRI input
        out_shape  : desired output shape tuple, typically condition.shape
        w          : guidance weight (0 = pure unconditional)
        clip       : clip output to [-1, 1] (default True)

    Returns:
        Sampled synthetic CT volume  (B, 1, D, H, W)

    Note:
        diffusion.p_sample_loop must accept a plain Python callable as its
        first argument. If your version of the library only accepts nn.Module,
        wrap model_fn in a thin nn.Module subclass.
    """
    null_cond = torch.zeros_like(condition)

    def model_fn(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # --- conditional pass ---
        x_cond   = torch.cat([x_t, condition], dim=1)
        n_cond   = model(x_cond, t)[:, :1]         # mean channel only

        # --- unconditional pass (null MRI condition) ---
        x_uncond = torch.cat([x_t, null_cond], dim=1)
        n_uncond = model(x_uncond, t)[:, :1]

        # --- CFG combination ---
        return n_uncond + w * (n_cond - n_uncond)

    return diffusion.p_sample_loop(
        model_fn,
        out_shape,
        clip_denoised=clip,
    )