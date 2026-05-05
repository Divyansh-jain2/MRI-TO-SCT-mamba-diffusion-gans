# SegMamba — MRI-to-CT Synthesis

SegMamba is a hybrid 3D U-Net that fuses CNN local feature extraction with Mamba State Space Models (SSM) for long-range sequence modeling, adapted for single-channel MRI → Synthetic CT translation on brain data.

---

## Folder Contents

```
SegMamba/
├── README.md
├── run_training.sh                # Training launch script
├── run_eval.sh                    # Evaluation launch script
├── run_viz.sh                     # Visualization generation script
├── segmamba_report.md             # Detailed architecture report
│
├── checkpoints/
│   ├── segmamba_best.pth          # Best model weights (lowest val loss)
│   ├── segmamba_epoch50.pth … segmamba_epoch500.pth
│   ├── segmamba_train_log.txt
│   └── segmamba_test_results.txt
│
├── predictions/                   # Test-set .npy arrays (37 cases)
│
└── visualizations/                # Side-by-side MRI | Pred CT | GT CT (37 PNGs)
    ├── brain_001_comparison.png
    └── …  brain_037_comparison.png
```

> Shared source code: [`../src/`](../src/) — `models.py`, `train.py`, `evaluate.py`, `dataset.py`, `losses.py`, `visualize.py`, `dosometric.py`

---

## End-to-End Architecture

```mermaid
flowchart TD
    Input["MRI Input · (1, 64, 192, 192)"]
    Input --> Stem["Stem · 2× ConvNormAct\n1 → 32 ch"]

    Stem --> E1["Enc1 · SegMambaBlock\n32 ch · full res"]
    E1   --> D1["Down1 · Stride-2 Conv"]
    D1   --> E2["Enc2 · SegMambaBlock\n64 ch · ½ res"]
    E2   --> D2["Down2 · Stride-2 Conv"]
    D2   --> E3["Enc3 · SegMambaBlock\n128 ch · ¼ res"]
    E3   --> D3["Down3 · Stride-2 Conv"]
    D3   --> E4["Enc4 · Bottleneck · SegMambaBlock\n256 ch · ⅛ res"]

    E4   --> U3["Up3 · Trilinear + Conv"]
    E3   -->|skip concat| U3
    U3   --> Dec3["Dec3 · SegMambaBlock · 128 ch"]

    Dec3 --> U2["Up2 · Trilinear + Conv"]
    E2   -->|skip concat| U2
    U2   --> Dec2["Dec2 · SegMambaBlock · 64 ch"]

    Dec2 --> U1["Up1 · Trilinear + Conv"]
    E1   -->|skip concat| U1
    U1   --> Dec1["Dec1 · SegMambaBlock · 32 ch"]

    Dec1 --> Head["Output Head · Conv3d + Tanh"]
    Head --> Out["Synthetic CT · (1, 64, 192, 192)"]
```

### SegMambaBlock (per encoder/decoder stage)

```mermaid
flowchart LR
    In["Input\n(B, C, D, H, W)"] --> CNN["ResBlock\nGroupNorm + ReLU + Conv3d"]
    CNN --> Flat["Flatten spatial\n→ (B, D·H·W, C)"]
    Flat --> SSM["Mamba SSM\nd_state = 16"]
    SSM --> Reshape["Reshape\n→ (B, C, D, H, W)"]
    Reshape --> Add["+ Residual skip"]
    Add --> Out["Output\n(B, C, D, H, W)"]
```

---

## Training Pipeline

```mermaid
flowchart LR
    Data["brain_npy\n(MRI + CT pairs)\nshape: (2, 192, 192, 96)"]
    Data --> Patch["Random Patch\n64 × 192 × 192"]
    Patch --> Model["SegMamba\n~18 M params"]
    Model --> Loss["Loss function\nepoch < 100 → wMAE\nepoch ≥ 100 → wMAE + SSIM + AFP"]
    Loss --> Opt["Adam\nβ₁=0.9 β₂=0.999 ε=1e-8\nlr₀ = 5 × 10⁻⁴"]
    Opt --> Sched["CosineAnnealingLR\nT_max = 500 · η_min = 1 × 10⁻⁶"]
    Sched -->|"next epoch"| Model
```

### Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | Adam (β₁=0.9, β₂=0.999) |
| Initial LR | 5 × 10⁻⁴ |
| LR schedule | Cosine annealing · T_max=500 · η_min=1×10⁻⁶ |
| Epochs | 500 |
| Batch size | 2 |
| Patch size | (64, 192, 192) D×H×W |
| Base channels | 32 → 64 → 128 → 256 |
| SSM state dim | 16 |
| Parameters | ~18 M |
| Mixed precision | AMP (fp16) |
| Checkpoint save | Every 50 epochs + best val |

### Loss Schedule

| Phase | Epochs | Components | HU tissue weights |
|---|---|---|---|
| Warmup | 1 – 99 | wMAE | Bone 3.0 · Soft tissue 1.5 · Air 0.5 |
| Full | 100 – 500 | wMAE + SSIM + AFP | same |

**AFP** = Anatomical Feature Preservation loss on high-gradient regions.

---

## Running

```bash
# From inside SegMamba/
bash run_training.sh

# Or directly:
python ../src/train.py \
    --data_dir /DATA/divyansh/mc_ddpm_data/brain_npy \
    --model segmamba \
    --epochs 500 \
    --batch_size 2 \
    --lr 5e-4 \
    --base_ch 32 \
    --save_dir ./checkpoints
```

### Evaluate

```bash
bash run_eval.sh

# Or directly:
python ../src/evaluate.py \
    --data_dir /DATA/divyansh/mc_ddpm_data/brain_npy \
    --checkpoint ./checkpoints/segmamba_best.pth \
    --model segmamba \
    --save_preds
```

---

## Results

### Image Quality (37 test cases)

| Metric | Score | Std Dev |
|---|---|---|
| MAE | 0.0480 | ± 0.0079 |
| PSNR | 24.79 dB | ± 1.19 dB |
| SSIM | 0.8432 | ± 0.0369 |

### Dosimetric Performance

| Metric | SegMamba |
|---|---|
| PSNR (3D) | 24.79 dB |
| PSNR (2D) | 25.42 dB |
| PSNR (1D) | 32.84 dB |
| SSIM | 0.8374 |
| Air MAE | 65.74 HU |
| Soft Tissue MAE | 38.15 HU |
| Bone MAE | 208.52 HU |
| RED MAE | 0.05208 |
| Gamma (1% / 1mm) | 91.61% |
| Gamma (2% / 2mm) | 99.35% |

---

## Sample Results

Best test case (brain_005 — PSNR 26.40 dB):

![SegMamba brain_005](visualizations/brain_005_comparison.png)

![SegMamba brain_010](visualizations/brain_010_comparison.png)

![SegMamba brain_020](visualizations/brain_020_comparison.png)

> All 37 test comparisons: [`visualizations/`](visualizations/)

Final epoch training dashboard:

![Final epoch dashboard](results/dashboard_final.png)
