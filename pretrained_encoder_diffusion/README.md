# Pretrained MRI Encoder + Diffusion Model — MRI-to-CT Synthesis

A **two-stage pipeline** for MRI → Synthetic CT generation. The core contribution is a **task-aware MRI semantic encoder** pretrained to extract CT-discriminative MRI features, which then conditions a 3D diffusion denoising model via cross-attention.

---

## Motivation

In the original MC-IDDPM pipeline, the MRI is concatenated directly to the noisy CT input as a 2-channel tensor. This requires the denoising network itself to learn MRI-to-CT semantics while also learning the diffusion denoising task — two objectives competing inside a single model.

**This project separates those concerns:**

| | Original MC-IDDPM | This approach |
|---|---|---|
| MRI role | Concatenated to noisy CT input | Encoded into rich semantic features |
| MRI feature learning | Inside the denoiser (implicit) | **Dedicated pretrained encoder (explicit)** |
| Denoiser input | `[noisy_CT ‖ MRI]` 2-channel | `noisy_CT` + cross-attention from encoder |
| Stage count | 1 | 2 |

---

## Pipeline Overview

```mermaid
flowchart TD
    subgraph Stage1["Stage 1 — MRI Encoder Pretraining (frozen after)"]
        MRI1["MRI\n(1, H, W, D)"] --> Enc["MRIAutoencoder\n4-level hierarchical encoder"]
        Enc --> HeadA["Head A · MRI Reconstruction\nL1 + SSIM · weight 0.3 + 0.2"]
        Enc --> HeadB["Head B · CT Prediction\nL1 · weight 0.5"]
        HeadA & HeadB --> S1Loss["Loss = 0.5×L1(CT) + 0.3×L1(MRI) + 0.2×SSIM(MRI)"]
        S1Loss --> S1Out["Save: stage1_encoder/checkpoints/best_mri_encoder.pt\n(encoder weights only · heads discarded)"]
    end

    subgraph Stage2["Stage 2 — Hybrid Diffusion Training (encoder frozen)"]
        MRI2["MRI"] --> FrozenEnc["Frozen MRI Encoder"]
        FrozenEnc --> Feats["f₁ 64ch · f₂ 128ch · f₃ 192ch · f₄ 256ch\nglobal_token 256-dim"]
        NoisyCT["noisy CT\n(timestep t)"] --> UNet["Denoising UNet\ntrainable"]
        Feats -->|"CrossAttention3D\nK, V from encoder · Q from UNet"| UNet
        UNet --> Out2["ε + learned variance prediction → Synthetic CT"]
    end

    Stage1 --> Stage2
```

---

## Folder Structure

```
pretrained_encoder_diffusion/
├── README.md
├── environment.yml / environment_linux.yml
├── LICENSE
│
├── network/                           # Model architectures
│   ├── mri_encoder.py                 # MRIAutoencoder + MRISemanticEncoder
│   ├── hybrid_model.py                # HybridDiffusionModel (encoder + denoiser)
│   ├── cross_attention.py             # CrossAttention3D
│   ├── Diffusion_model_transformer.py # SwinViT-based denoising network
│   ├── Diffusion_model_Unet.py        # UNet-based denoising network
│   └── util_network.py               # Shared utilities
│
├── diffusion/                         # Diffusion process
│   ├── GaussianDiffusion.py
│   ├── HybridGaussianDiffusion.py
│   ├── HybridSpacedDiffusion.py
│   └── respace.py / resampler.py
│
├── stage1_encoder/                    # Stage 1: MRI encoder pretraining
│   ├── pretrain_mri_encoder.py
│   ├── eval_mri_encoder.py
│   ├── run_pretrain.sh
│   ├── checkpoints/best_mri_encoder.pt
│   ├── visualizations/               # Per-epoch: MRI in/recon + CT gt/pred
│   └── eval_results/                 # Encoder evaluation outputs
│
├── stage2_diffusion/                  # Stage 2: Hybrid diffusion
│   ├── main_hybrid.py
│   ├── inference_hybrid.py
│   ├── run_training_hybrid.sh
│   ├── run_inference_hybrid.sh
│   ├── checkpoints/best_model.pt
│   ├── results/brain_hybrid/         # Training-time visualizations
│   └── inference_results/hybrid/
│       ├── metrics.txt
│       └── vis/sample_000.png … sample_036.png
│
├── baseline/                          # Original MC-IDDPM (no encoder)
└── scripts/                           # Preprocessing + evaluation utilities
```

---

## Stage 1 — MRI Semantic Encoder

### Architecture

```mermaid
flowchart TD
    MRI["MRI · (B, 1, H, W, D)"]
    MRI --> Stem["Stem\n7×7×3 Conv → 3×3×1 Conv → 64 ch"]

    Stem --> L1["Level 1 · full res · 64 ch\n2× ResBlock + GroupNorm\n+ WindowSelfAttention3D window=(4,4,4) heads=4"]
    L1 --> P1["StridedConv pool=(2,2,1) · XY↓2 Z unchanged"]

    P1 --> L2["Level 2 · ½ res · 128 ch\n2× ResBlock + GroupNorm\n+ WindowSelfAttention3D heads=4"]
    L2 --> P2["StridedConv pool=(2,2,1)"]

    P2 --> L3["Level 3 · ¼ res · 192 ch\n2× ResBlock + GroupNorm\n+ WindowSelfAttention3D heads=8"]
    L3 --> P3["StridedConv pool=(2,2,1)"]

    P3 --> L4["Level 4 · ⅛ res · 256 ch\n2× ResBlock + GroupNorm\n+ WindowSelfAttention3D heads=8"]
    L4 --> GAP["AdaptiveAvgPool3d\n+ Transformer bottleneck"]
    GAP --> GT["global_token · (B, 256)"]

    L1 --> f1["f₁ · (B, 64, H, W, D)"]
    L2 --> f2["f₂ · (B, 128, H/2, W/2, D)"]
    L3 --> f3["f₃ · (B, 192, H/4, W/4, D)"]
    L4 --> f4["f₄ · (B, 256, H/8, W/8, D)"]
```

### Pretraining dual heads (discarded after Stage 1)

```mermaid
flowchart LR
    Feats["f₁ f₂ f₃ f₄\nglobal_token"] --> DecA["Decoder A · MRI Reconstruction\nUpsampling + Conv → (1, H, W, D)"]
    Feats --> DecB["Decoder B · CT Prediction\nUpsampling + Conv → (1, H, W, D)"]
    DecA --> LA["L1(recon, MRI_gt) + SSIM(recon, MRI_gt)\nweight = 0.3 + 0.2"]
    DecB --> LB["L1(pred, CT_gt)\nweight = 0.5"]
```

### Stage 1 Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | Adam (β₁=0.9, β₂=0.999) |
| Initial LR | 1 × 10⁻⁴ |
| LR schedule | CosineAnnealingLR · T_max=500 · η_min=1×10⁻⁷ |
| Epochs | 500 (early stopping patience=80) |
| Batch size | 4 |
| Patch size | (64, 64, 4) |
| Encoder channels | 64 → 128 → 192 → 256 |
| SA window size | (4, 4, 4) |
| SA heads per level | (4, 4, 8, 8) |
| Pooling kernel | (2, 2, 1) — XY↓2, Z unchanged |
| Mixed precision | AMP (fp16) with SSIM cast to float32 |
| Loss weights | CT: 0.5 · MRI recon L1: 0.3 · MRI recon SSIM: 0.2 |

> **LR = 1e-4, not 3e-4:** 3×10⁻⁴ caused fp16 NaN in SSIM Gaussian convolution. SSIM is explicitly cast to float32 under AMP to prevent silent overflow.

---

## Stage 2 — Hybrid Diffusion Model

### Architecture

```mermaid
flowchart TD
    subgraph Enc["Frozen MRI Encoder"]
        MRI2["MRI"] --> FE["MRISemanticEncoder"]
        FE --> f1e["f₁ · 64 ch"]
        FE --> f2e["f₂ · 128 ch"]
        FE --> f3e["f₃ · 192 ch"]
        FE --> f4e["f₄ · 256 ch"]
        FE --> GT2["global_token · 256"]
    end

    T["Timestep t"] --> SinEmb["Sinusoidal Embed\n+ global_token → condition c"]

    subgraph Denoiser["Trainable Denoising UNet"]
        NoisyCT["noisy CT · (B,1,H,W,D)"] --> IC["Init Conv 1→64 ch"]
        IC --> DL1["Down L1 · ResBlock(c)\n+ CrossAttn3D Q=feat K,V=f₁"]
        DL1 --> DL2["Down L2 · ResBlock(c)\n+ CrossAttn3D Q=feat K,V=f₂"]
        DL2 --> DL3["Down L3 · ResBlock(c)\n+ CrossAttn3D Q=feat K,V=f₃+f₄"]
        DL3 --> Mid["Middle · 2× ResBlock(c)"]
        Mid --> UL3["Up L3 + skip → Dec L3"]
        UL3 --> UL2["Up L2 + skip → Dec L2"]
        UL2 --> UL1["Up L1 + skip → Dec L1"]
        UL1 --> OC["Output Conv 64→2 ch"]
        OC --> Pred["ε prediction + learned variance"]
    end

    GT2 --> SinEmb
    f1e -->|K, V| DL1
    f2e -->|K, V| DL2
    f3e & f4e -->|K, V| DL3

    Pred --> SCT["Synthetic CT\n(DDPM reverse · 1000 train / 50 infer steps)"]
```

### CrossAttention3D

```mermaid
flowchart LR
    Q["Query · Denoiser feature\n(B, C_d, D, H, W)"] --> QP["Linear projection → Q"]
    KV["Key/Value · Encoder fᵢ\n(B, C_e, D', H', W')"] --> KP["Linear projection → K"]
    KV --> VP["Linear projection → V"]
    QP & KP & VP --> Attn["Scaled Dot-Product Attention\nsoftmax(QKᵀ / √d) · V"]
    Attn --> Add["+ Residual"]
    Add --> Out["Output feature"]
```

### Stage 2 Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | Adam (β₁=0.9, β₂=0.999) |
| Initial LR | 1 × 10⁻⁴ |
| LR schedule | CosineAnnealingLR · T_max=500 · η_min=1×10⁻⁷ |
| Diffusion steps (train) | 1000 |
| Diffusion steps (infer) | 50 (spaced / DDIM-style) |
| Loss | Hybrid ELBO (ε-prediction + learned variance) |
| Encoder | Frozen (Stage 1 weights loaded) |
| Patch size | (64, 64, 4) |
| Batch size | 4 |
| Mixed precision | AMP (fp16) |

---

## Running

### Step 1 — Pretrain the MRI Encoder

```bash
cd stage1_encoder
bash run_pretrain.sh
# Or: python pretrain_mri_encoder.py
```

Monitor:
```bash
tensorboard --logdir stage1_encoder/tensorboard_logs --port 6007
```

Output: `stage1_encoder/checkpoints/best_mri_encoder.pt`

### Step 2 — Train Hybrid Diffusion

```bash
cd stage2_diffusion
bash run_training_hybrid.sh
# Or: python main_hybrid.py
```

`main_hybrid.py` auto-loads `../stage1_encoder/checkpoints/best_mri_encoder.pt` and freezes it.

### Step 3 — Run Inference

```bash
cd stage2_diffusion
bash run_inference_hybrid.sh
# Or: python inference_hybrid.py
```

Results → `stage2_diffusion/inference_results/hybrid/`

---

## Results

### Stage 2 Test-Set Performance (37 brain cases)

| Metric | Score | Std Dev |
|---|---|---|
| L1 | 0.0492 | ± 0.0077 |
| PSNR | 24.12 dB | ± 1.14 dB |
| SSIM | 0.8037 | ± 0.0334 |
| MAE (HU) | 65.8 HU | ± 10.3 HU |

Full breakdown: [`stage2_diffusion/inference_results/hybrid/metrics.txt`](stage2_diffusion/inference_results/hybrid/metrics.txt)

### Comparison with Mamba Approaches

| Metric | Hybrid Encoder-Diffusion | Best Mamba (TriPlane) |
|---|---|---|
| PSNR | 24.12 dB | **25.79 dB** |
| SSIM | 0.8037 | **0.8561** |
| MAE | 0.0492 | **0.0445** |

---

## Sample Inference Results

Best inference case (Sample 0 — PSNR 26.12 dB, SSIM 0.8386):

![Sample 000](stage2_diffusion/inference_results/hybrid/vis/sample_000.png)

![Sample 001](stage2_diffusion/inference_results/hybrid/vis/sample_001.png)

![Sample 002](stage2_diffusion/inference_results/hybrid/vis/sample_002.png)

> All 37 samples: [`stage2_diffusion/inference_results/hybrid/vis/`](stage2_diffusion/inference_results/hybrid/vis/)

### Stage 1 Encoder Pretraining Progression

Each image: MRI input · MRI reconstructed · CT ground-truth · CT predicted

![Epoch 0001](stage1_encoder/visualizations/epoch_0001.png)

![Epoch 0080](stage1_encoder/visualizations/epoch_0080.png)

![Epoch 0500](stage1_encoder/visualizations/epoch_0500.png)

> All epoch visualizations: [`stage1_encoder/visualizations/`](stage1_encoder/visualizations/)

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| Dual-task pretraining | CT prediction forces HU-discriminative features; MRI recon forces spatial fidelity |
| CT loss weight = 0.5 (highest) | Primary pretraining goal is CT-relevant MRI representation |
| Encoder frozen in Stage 2 | Prevents catastrophic forgetting of pretrained semantics |
| Cross-attention (not concat) | Denoiser selectively attends to relevant encoder features per scale |
| GroupNorm in encoder | Stable at batch size 2–4; BatchNorm unstable at small batches |
| LR = 1e-4 not 3e-4 | 3e-4 caused fp16 SSIM Gaussian conv overflow → NaN in early runs |
| SSIM in float32 under AMP | fp16 Gaussian conv overflows silently; explicit cast prevents NaN |
| Pooling kernel (2,2,1) | XY downsampling preserves Z-axis resolution for thin-slice brain data |

---

## Reference

Original MC-IDDPM paper: [Synthetic CT generation from MRI using 3D transformer-based denoising diffusion model](https://aapm.onlinelibrary.wiley.com/doi/abs/10.1002/mp.16847) — Shaoyan Pan et al., *Medical Physics* 2023.

Built on: [guided-diffusion](https://github.com/openai/guided-diffusion) · SwinUnet · MONAI
