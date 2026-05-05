# Diffusion UMamba — MRI-to-CT Synthesis

A conditional **Denoising Diffusion Probabilistic Model (DDPM)** for MRI → Synthetic CT generation, using a **3D UMamba backbone** as the noise-prediction network. This replaces the standard Swin Transformer denoiser with Mamba State Space Models, achieving linear O(N) complexity for 3D volumetric data.

---

## Folder Structure

```
diffusion_umamba/
├── README.md
├── Diffusion_umamba_report.md     # Full technical report
├── models.py                      # UMamba model definition
├── main_umamba_diffusion.py       # Training entry point
├── test_umamba_diffusion.py       # Inference + test evaluation
├── evaluate_dosimetry.py          # Dosimetric analysis (RED, Gamma)
├── run_umamba_diffusion.sh        # Training launch script
├── run_test_umamba_diffusion.sh   # Test / inference script
├── run_manual_pipeline.sh         # Step-by-step manual pipeline
├── training_log.txt               # Full training log (500 epochs)
├── test_umamba.log                # Inference / test log
│
├── network/                       # Denoising network architectures
│   ├── Diffusion_model_mamba.py   # Mamba-based diffusion denoiser
│   ├── Diffusion_model_transformer.py
│   ├── Diffusion_model_Unet.py
│   ├── SwinUnetr.py
│   ├── nnFormer.py
│   └── util_network.py
│
├── diffusion/                     # DDPM process
│   ├── Create_diffusion.py        # Factory: build GaussianDiffusion
│   ├── GaussianDiffusion.py       # Core DDPM forward/reverse process
│   ├── normal_diffusion.py
│   ├── resampler.py
│   └── respace.py
│
├── checkpoints/                   # Model weights
│   ├── best_model.pt              # Best val-loss checkpoint
│   └── latest_model.pt            # Most recent epoch
│
├── visualizations/                # Training-time comparison PNGs
│   ├── epoch_10_comparison.png … epoch_490_comparison.png
│   └── training_metrics.png       # Loss + PSNR curves
│
├── inference_results/             # Test-set outputs
│   ├── dosimetric_metrics_all.csv # Per-case dosimetric metrics (38 cases)
│   ├── dosimetric_test_metrics.csv
│   ├── pred_brain_001.nii.gz … pred_brain_037.nii.gz  (gitignored)
│   └── gt_brain_001.nii.gz … gt_brain_037.nii.gz      (gitignored)
│
└── results/                       # Training-time NIfTI samples (gitignored)
    └── sct_epoch*.nii.gz
```

---

## End-to-End Architecture

```mermaid
flowchart TD
    MRI["MRI Condition · (B, 1, 64, 64, 4)"]
    CT["Noisy CT x_t · (B, 1, 64, 64, 4)"]
    T["Timestep t"]

    MRI & CT --> Cat["Concat → (B, 2, 64, 64, 4)"]
    T --> TEmb["Sinusoidal Embed + MLP\ndim = 256"]

    Cat --> Stem["Stem · ConvNormAct\n2 → 64 ch"]

    subgraph ENC["Encoder"]
        E1["UMambaBlock · 64 ch"]
        E1 --> D1["Down ×2,2,2 → 128 ch"]
        D1 --> E2["UMambaBlock · 128 ch"]
        E2 --> D2["Down ×2,2,1 → 256 ch"]
        D2 --> E3["UMambaBlock · 256 ch"]
        E3 --> D3["Down ×2,2,1 → 512 ch"]
    end

    subgraph BOT["Bottleneck"]
        E4["UMambaBlock · 512 ch"]
    end

    subgraph DEC["Decoder"]
        U3["ConvTranspose → 256 ch"]
        U3 --> Dec3["UMambaBlock · 512→256 ch + skip"]
        Dec3 --> U2["ConvTranspose → 128 ch"]
        U2 --> Dec2["UMambaBlock · 256→128 ch + skip"]
        Dec2 --> U1["ConvTranspose → 64 ch"]
        U1 --> Dec1["UMambaBlock · 128→64 ch + skip"]
    end

    Stem --> E1
    D3 --> E4
    E4 --> U3
    Dec1 --> Head["Head · Conv 1×1×1\n64 → 2 ch"]
    Head --> Out["Output · (B, 2, 64, 64, 4)\nPredicted ε + Variance"]

    TEmb -. "injected into every UMambaBlock" .-> ENC
    TEmb -. "injected into every UMambaBlock" .-> BOT
    TEmb -. "injected into every UMambaBlock" .-> DEC
```

### UMambaBlock (noise-prediction blocks)

```mermaid
flowchart LR
    In["Input · (B, C, D, H, W)"] --> CNN["ResBlock\nGroupNorm + SiLU + Conv3d"]
    TEmb2["Timestep embedding\n(B, dim)"] --> Proj["Linear → (B, 2C)\nscale + shift"]
    CNN & Proj --> FiLM["FiLM conditioning\nscale · feat + shift"]
    FiLM --> Flat["Flatten volume\n→ (B, D·H·W, C)"]
    Flat --> SSM["Bidirectional Mamba SSM\nd_state=16 · O(N) complexity"]
    SSM --> Reshape["Reshape → (B, C, D, H, W)"]
    Reshape --> Add["+ Residual skip"]
    Add --> Out["Output · (B, C, D, H, W)"]
```

### Diffusion Process

```mermaid
flowchart LR
    subgraph FWD["Forward (Training)"]
        CT2["CT ground truth x_0"] --> Add2["Add Gaussian noise\nat timestep t\nx_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε"]
        Add2 --> Inp["Input to UMamba\n[x_t ‖ MRI]"]
    end

    subgraph REV["Reverse (Inference · 1000 steps)"]
        Noise["Pure Gaussian noise x_T"] --> Loop["Iterative denoising\nx_{t-1} = μ_θ(x_t, t, MRI)\n+ σ_t z"]
        Loop --> SCT["Synthetic CT x_0"]
    end
```

---

## Training Pipeline

```mermaid
flowchart LR
    Data["brain_npy\n(MRI + CT pairs)\nshape: (2, 192, 192, 96)"]
    Data --> Patch["Random Patch\n64 × 64 × 4\npatch_num = 2 per volume"]
    Patch --> Aug["MONAI Augmentation\nRandFlip · RandRotate90\nRandGaussianNoise · RandAffine"]
    Aug --> DDPM["DDPM Forward\nsample t ~ Uniform(0, 999)\nadd noise to CT"]
    DDPM --> Model["UMamba Diffusion\n~45 M params"]
    Model --> Loss["Hybrid ELBO Loss\nε-MSE + VLB (learned variance)"]
    Loss --> Opt["Adam\nlr₀ = 3 × 10⁻⁵\nweight_decay = 1 × 10⁻⁵"]
    Opt -->|"next epoch"| Model
```

### Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Initial LR | 3 × 10⁻⁵ |
| Weight decay | 1 × 10⁻⁵ |
| LR schedule | None (fixed LR) |
| Diffusion steps (train) | 1000 |
| Diffusion steps (infer) | 1000 |
| Epochs | 500 |
| Batch size | 8 |
| Patch size | (64, 64, 4) |
| Patches per volume | 2 |
| Input channels | 2 (MRI + noisy CT concat) |
| Output channels | 2 (ε + variance) |
| Base channels | 64 → 128 → 256 → 512 |
| SSM state dim | 16 |
| Timestep embed dim | 256 |
| Mixed precision | AMP (fp16) |
| Checkpoint save | Best val loss + latest |

### Why UMamba over Swin Transformer?

| Property | Swin Transformer | UMamba |
|---|---|---|
| Complexity | O(N²) within windows | O(N) — linear |
| Receptive field | Local window (4×4×4) | Infinite (full sequence) |
| 3D global context | Via shifted windows (approximation) | Direct 1D SSM scan |
| Memory at 64×64×4 patch | High (window attention) | Low (SSM state-based) |

---

## Running

### Train

```bash
bash run_umamba_diffusion.sh

# Or directly:
python main_umamba_diffusion.py

# Resume from latest checkpoint:
python main_umamba_diffusion.py --resume
```

### Inference / Test

```bash
bash run_test_umamba_diffusion.sh

# Or directly:
python test_umamba_diffusion.py
```

### Dosimetric Evaluation

```bash
python evaluate_dosimetry.py
# Output: inference_results/dosimetric_metrics_all.csv
```

---

## Results

### Image Quality (38 test cases)

| Metric | Score | Std Dev |
|---|---|---|
| PSNR (3D) | 22.49 dB | ± 0.82 dB |
| SSIM | 0.7678 | ± 0.0318 |

### Dosimetric Performance

| Metric | Score |
|---|---|
| Air MAE | 73.00 HU |
| Soft Tissue MAE | 49.81 HU |
| Bone MAE | 340.88 HU |
| RED MAE | 0.06597 |
| Gamma (1% / 1mm) | 90.52% |
| Gamma (2% / 2mm) | 99.03% |

### Comparison vs Other Approaches

| Metric | Diffusion UMamba | TriPlane (best) | Pretrained Enc. Diffusion |
|---|---|---|---|
| PSNR | 22.49 dB | **25.79 dB** | 24.12 dB |
| SSIM | 0.7678 | **0.8561** | 0.8037 |
| Gamma (1%/1mm) | 90.52% | 90.61% | — |

> Diffusion UMamba crosses the clinical **90% Gamma threshold** (90.52%) — competitive with the best Mamba variant on dosimetric criteria despite lower pixel-level PSNR. This suggests the generated CT preserves clinically relevant dose distribution properties even when absolute HU accuracy is lower.

---

## Sample Results

Training loss and validation PSNR over 500 epochs:

![Training Metrics](visualizations/training_metrics.png)

Epoch 490 sample (MRI Input · CT Ground Truth · Generated CT):

![Epoch 490](visualizations/epoch_490_comparison.png)

Epoch 10 (early training comparison):

![Epoch 10](visualizations/epoch_10_comparison.png)

> All epoch visualizations: [`visualizations/`](visualizations/)

Full per-case dosimetric results: [`inference_results/dosimetric_metrics_all.csv`](inference_results/dosimetric_metrics_all.csv)
