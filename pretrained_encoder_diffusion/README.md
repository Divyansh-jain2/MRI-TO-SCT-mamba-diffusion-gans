# Pretrained MRI Encoder + Diffusion Model — MRI-to-CT Synthesis

This project implements a **two-stage pipeline** for MRI → Synthetic CT generation. The core contribution is a **task-aware MRI semantic encoder** pretrained to extract CT-discriminative MRI features, which then conditions a 3D diffusion denoising model via cross-attention.

## Motivation

In the original MC-IDDPM pipeline, the MRI is concatenated directly to the noisy CT input as a 2-channel tensor. This requires the denoising network itself to learn MRI-to-CT semantics while also learning the diffusion denoising task — two objectives that compete inside a single model.

**This project separates those concerns:**

| | Original MC-IDDPM | This approach |
|---|---|---|
| MRI role | Concatenated to noisy CT input | Encoded into rich semantic features |
| MRI feature learning | Inside the denoiser (implicit) | **Dedicated pretrained encoder (explicit)** |
| Denoiser input | `[noisy_CT ‖ MRI]` 2-channel | `noisy_CT` + cross-attention from encoder |
| Stage count | 1 | 2 |

---

## Pipeline Overview

```
Stage 1 — MRI Encoder Pretraining
──────────────────────────────────────────────────────────
  MRI ──► MRIAutoencoder ──► Head A: MRI Reconstruction
                         └──► Head B: CT Prediction
  Loss = 0.5 × L1(CT_pred, CT_gt)
       + 0.3 × L1(MRI_recon, MRI_gt)
       + 0.2 × SSIM(MRI_recon, MRI_gt)
  ► Only encoder weights saved: stage1_encoder/checkpoints/best_mri_encoder.pt

Stage 2 — Hybrid Diffusion Training (encoder frozen)
──────────────────────────────────────────────────────────
  MRI ──► Frozen MRI Encoder ──► [f₁, f₂, f₃, f₄, global]
                                           │ Cross-Attention (K, V)
  t   ──► Sinusoidal Embed + global_MRI   │
                                           ▼
  noisy_CT ──► Denoising UNet ◄───────────┘
               (trainable)
  ► stage2_diffusion/checkpoints/best_model.pt
```

---

## Folder Structure

```
pretrained_encoder_diffusion/
│
├── README.md                          ← you are here
├── environment.yml                    # Conda environment (cross-platform)
├── environment_linux.yml              # Linux-specific environment
├── LICENSE
│
├── network/                           ← Model architecture definitions
│   ├── mri_encoder.py                 # MRIAutoencoder + MRISemanticEncoder
│   ├── hybrid_model.py                # HybridDiffusionModel (encoder + denoiser)
│   ├── cross_attention.py             # CrossAttention3D (encoder → denoiser link)
│   ├── Diffusion_model_transformer.py # SwinViT-based denoising network
│   ├── Diffusion_model_transformer_no_conv.py
│   ├── Diffusion_model_transformer_ori.py
│   ├── Diffusion_model_Unet.py        # UNet-based denoising network
│   ├── nnFormer.py
│   ├── SwinUnetr.py
│   └── util_network.py               # Shared utilities (conv_nd, normalization…)
│
├── diffusion/                         ← Diffusion process implementations
│   ├── Create_diffusion.py            # Factory: build GaussianDiffusion
│   ├── GaussianDiffusion.py           # Core DDPM forward/reverse process
│   ├── HybridGaussianDiffusion.py     # Hybrid variant with encoder conditioning
│   ├── HybridSpacedDiffusion.py       # Spaced-timestep hybrid diffusion
│   ├── normal_diffusion.py
│   ├── resampler.py                   # Timestep resampler
│   └── respace.py                     # Timestep respacing (DDIM-style)
│
├── stage1_encoder/                    ← Stage 1: MRI encoder pretraining
│   ├── pretrain_mri_encoder.py        # Dual-task training script
│   ├── eval_mri_encoder.py            # Evaluation of encoder quality
│   ├── run_pretrain.sh                # bash run_pretrain.sh
│   ├── checkpoints/
│   │   └── best_mri_encoder.pt        # Best encoder weights (by combined SSIM)
│   ├── visualizations/                # Per-epoch PNG: MRI in/recon + CT gt/pred
│   │   └── epoch_NNNN.png
│   ├── eval_results/                  # Evaluation outputs
│   └── tensorboard_logs/             # TensorBoard event files
│       └── mri_encoder_YYYYMMDD_HHMMSS/
│
├── stage2_diffusion/                  ← Stage 2: Hybrid diffusion
│   ├── main_hybrid.py                 # Training entry point
│   ├── hybrid_model.py                # Local model reference copy
│   ├── inference_hybrid.py            # Sliding-window + MC-sample inference
│   ├── inference_hybrid2.py           # Inference v2
│   ├── test_hybrid_model.py           # Sanity checks
│   ├── run_training_hybrid.sh         # bash run_training_hybrid.sh
│   ├── run_inference_hybrid.sh        # bash run_inference_hybrid.sh
│   ├── checkpoints/
│   │   └── best_model.pt
│   ├── results/
│   │   ├── brain_hybrid/             # Training-time visualizations
│   │   └── final/
│   └── inference_results/
│       └── hybrid/
│           ├── metrics.txt            # Aggregate + per-sample test metrics
│           └── vis/                   # sample_000.png … sample_036.png
│
├── baseline/                          ← Original MC-IDDPM (comparison)
│   ├── main.py                        # Original training (MRI concat to noisy CT)
│   ├── main_no_tensorboard.py
│   └── run_training.sh
│
├── scripts/                           ← Utility and analysis scripts
│   ├── run_preprocessing.py           # Data format conversion
│   ├── analyze_task1_data.py          # Dataset statistics
│   ├── dataInfo.py                    # Data inspection
│   ├── restructure.py                 # Dataset restructuring helper
│   ├── quick_eval.py                  # Quick metric calculation
│   ├── evaluate_final.py              # Full evaluation pipeline
│   └── generate_presentation.py      # Result presentation PNGs
│
├── notebooks/
│   └── MC-IDDPM main.ipynb            # Interactive diffusion walkthrough
│
└── data/
    └── MRI_to_CT_brain_for_dosimetric/
        └── imagesTr/                  # Sample .mat files for dosimetric analysis
```

---

## Architecture Details

### Stage 1 — MRI Semantic Encoder (`network/mri_encoder.py`)

A 4-level hierarchical feature extractor with window self-attention at each scale:

```
MRI (1, H, W, D)
    └─ Stem: 7×7×3 conv → 3×3×1 conv
        ├─ L1: 2×ResBlock(GroupNorm) + SA3D  →  f₁  [full res,  64 ch]
        ├─ L2: StridedConv↓2 + 2×ResBlock + SA3D →  f₂  [½ res, 128 ch]
        ├─ L3: StridedConv↓2 + 2×ResBlock + SA3D →  f₃  [¼ res, 192 ch]
        └─ L4: StridedConv↓2 + 2×ResBlock + SA3D →  f₄  [⅛ res, 256 ch]
                                                           │
                                                AdaptiveAvgPool
                                                + Transformer bottleneck
                                                           │
                                              global_token [B, 256]
```

**Pretraining dual heads (discarded after Stage 1):**
- **Head A** — MRI reconstruction: forces spatial precision in encoder features
- **Head B** — CT prediction: forces CT-discriminative (HU-relevant) feature learning

**Loss:** `0.5 × L1(CT) + 0.3 × L1(MRI recon) + 0.2 × SSIM(MRI recon)`

CT prediction carries the highest weight because capturing HU-relevant MRI structure is the primary pretraining objective.

### Stage 2 — Hybrid Diffusion Model (`network/hybrid_model.py`)

```
MRI ──► Frozen Encoder  →  f₁ (64ch), f₂ (128ch), f₃ (192ch), f₄ (256ch), global (256)
                                                                         │
timestep t ──► sinusoidal_embed ──+─ global_token  →  condition c       │
                                                                         │ K, V (Cross-Attn)
noisy_CT ──► Init Conv (1→64ch)                                          │
              Down L1: ResBlock(c) + CrossAttn(Q=feat, K,V=f₁) ◄────────┘
              Down L2: ResBlock(c) + CrossAttn(Q=feat, K,V=f₂)
              Down L3: ResBlock(c) + CrossAttn(Q=feat, K,V=f₃+f₄)
              Middle:  2× ResBlock(c)
              Up L3 + skip  →  Dec L3
              Up L2 + skip  →  Dec L2
              Up L1 + skip  →  Dec L1
              Output Conv (64→2ch)  →  ε + variance prediction
                    │
               Predicted CT
```

| Hyperparameter | Value |
|---|---|
| Encoder channels | 64 → 128 → 192 → 256 |
| Global token dim | 256 |
| SA window size | (4, 4, 4) |
| SA heads per level | (4, 4, 8, 8) |
| Encoder pooling kernel | (2, 2, 1) — XY↓2, Z unchanged |
| Diffusion steps | 1000 train / 50 inference |
| Loss | Hybrid ELBO (ε + learned variance) |

---

## Environment Setup

```bash
conda env create -f environment.yml
conda activate mri_ct_diffusion
```

---

## Running the Pipeline

### Step 1 — Pretrain the MRI Encoder

```bash
cd stage1_encoder
bash run_pretrain.sh
# Or: python pretrain_mri_encoder.py
```

Key config inside `pretrain_mri_encoder.py`:

| Variable | Default | Notes |
|---|---|---|
| `DATA_ROOT` | `/DATA/divyansh/mc_ddpm_data/brain_npy` | NPY dataset root |
| `EPOCHS` | 500 | Early stopping: patience = 80 |
| `BATCH_SIZE` | 4 | |
| `LR` | 1e-4 | Cosine annealing → 1e-7. **Do not use 3e-4** (causes fp16 NaN) |
| `PATCH_SIZE` | (64, 64, 4) | Must match `main_hybrid.py` |

Monitor:
```bash
tensorboard --logdir stage1_encoder/tensorboard_logs --port 6007
```

Output: `stage1_encoder/checkpoints/best_mri_encoder.pt`

### Step 2 — Train Hybrid Diffusion (encoder frozen)

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

### Stage 2 — Test-Set Performance (37 brain test cases)

| Metric | Score | Std Dev |
|---|---|---|
| L1 | 0.0492 | ± 0.0077 |
| PSNR | 24.12 dB | ± 1.14 dB |
| SSIM | 0.8037 | ± 0.0334 |
| MAE (HU) | 65.8 HU | ± 10.3 HU |

Full breakdown: [`stage2_diffusion/inference_results/hybrid/metrics.txt`](stage2_diffusion/inference_results/hybrid/metrics.txt)

### Sample Inference Outputs

![Sample 000](stage2_diffusion/inference_results/hybrid/vis/sample_000.png)
![Sample 001](stage2_diffusion/inference_results/hybrid/vis/sample_001.png)
![Sample 002](stage2_diffusion/inference_results/hybrid/vis/sample_002.png)

> All 37 samples: [`stage2_diffusion/inference_results/hybrid/vis/`](stage2_diffusion/inference_results/hybrid/vis/)

### Encoder Pretraining Visualizations

Each image shows: MRI input · MRI reconstructed · CT ground-truth · CT predicted

![Encoder epoch 0001](stage1_encoder/visualizations/epoch_0001.png)
![Encoder epoch 0080](stage1_encoder/visualizations/epoch_0080.png)

> All epoch visualizations: [`stage1_encoder/visualizations/`](stage1_encoder/visualizations/)

---

## Comparison with Mamba Approaches

| Metric | Hybrid Encoder-Diffusion | Best Mamba (TriPlane) |
|---|---|---|
| PSNR | 24.12 dB | **25.79 dB** |
| SSIM | 0.8037 | **0.8561** |
| MAE | 0.0492 | **0.0445** |

The Mamba approaches currently outperform the hybrid encoder-diffusion model. The encoder pretraining idea is sound — the gap may narrow with more training epochs, larger encoder capacity, or fine-tuning the cross-attention conditioning strategy.

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| Dual-task pretraining | CT prediction forces HU-discriminative features; MRI recon forces spatial fidelity |
| CT weight = 0.5 (highest) | Primary pretraining goal is CT-relevant MRI representation |
| Encoder frozen in Stage 2 | Prevents catastrophic forgetting of pretrained semantics |
| Cross-attention (not concat) | Denoiser selectively attends to relevant encoder features per scale |
| GroupNorm in encoder | Stable for batch size 2–4; BatchNorm unstable at small batches |
| LR = 1e-4 not 3e-4 | 3e-4 caused fp16 SSIM Gaussian conv overflow → NaN in early runs |
| SSIM computed in float32 under AMP | fp16 Gaussian conv overflows silently; explicit cast prevents NaN |

---

## Reference

Original MC-IDDPM paper: [Synthetic CT generation from MRI using 3D transformer-based denoising diffusion model](https://aapm.onlinelibrary.wiley.com/doi/abs/10.1002/mp.16847) — Shaoyan Pan et al., *Medical Physics* 2023.

Built on: [guided-diffusion](https://github.com/openai/guided-diffusion) · SwinUnet · MONAI
