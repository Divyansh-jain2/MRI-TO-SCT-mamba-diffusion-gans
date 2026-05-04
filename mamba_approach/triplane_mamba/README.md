# TriPlane Mamba (TriPlaneMamba-UNet) — MRI-to-CT Synthesis

TriPlane Mamba improves on TriAxial by replacing 1D axial SSM scans with **2D planar Mamba scans** across the Axial (HW), Coronal (DW), and Sagittal (DH) planes. A parallel `MultiScaleDepthConv` branch captures local structure at 4 dilation scales. Combined, this delivers richer spatial context than single-axis scanning.

---

## Folder Structure

```
triplane_mamba/
├── README.md                          ← you are here
├── Triplane_Mamba_Report.md           # Full technical report with source code
├── models.py                          # TriPlaneMamba-UNet architecture
├── train.py                           # Training script
├── evaluate.py                        # Inference + MAE / PSNR / SSIM
├── evaluate_dosimetric.py             # RED / Gamma-index dosimetric analysis
├── dataset.py                         # Data loader
├── losses.py                          # Loss functions
├── visualize.py                       # Generates comparison visualizations
├── environment.yml                    # Conda environment spec
├── run_training_triplane.sh           # Training shell script
├── resume_training_triplane.sh        # Resume from checkpoint
├── run_eval_trimamba.sh               # Evaluation shell script
├── training_triplane_output.log       # Full training log
├── architecture.html                  # Interactive architecture diagram
│
├── checkpoints_triplane/
│   ├── triplane_best.pth              # Best model weights
│   ├── triplane_epoch50.pth           # Epoch checkpoints (50 … 500)
│   ├── ...
│   └── visuals/                       # Per-epoch training dashboards
│       ├── dashboard_epoch_001.png
│       └── ...  dashboard_epoch_500.png
│
└── predictions_triplane/
    ├── dosimetric_metrics.csv         # Full per-case dosimetric results
    └── brain_001.npy … brain_037.npy # Test-set prediction arrays
```

---

## Architecture

### TriPlaneMambaBlock

Each block runs two parallel branches:

```
Input (B, C, D, H, W)
    ├─ Branch 1 (Local): MultiScaleDepthConv
    │       └─ 4 parallel dilated Conv3d [d=1, 2, 4, 8] along Depth → concat → project
    │
    └─ Branch 2 (Global): Tri-plane bidirectional Mamba scans
            ├─ HW plane  (Axial):    reshape → BiMamba → restore
            ├─ DW plane  (Coronal):  reshape → BiMamba → restore
            └─ DH plane  (Sagittal): reshape → BiMamba → restore
                        ↓
               Fusion Conv3d(y_hw + y_dw + y_dh)
                        ↓
    Output = Residual + Local + Global
```

### U-Net Topology

```
MRI (1, D, H, W)
    └─ Stem
        └─ Enc1 → Down1 → Enc2 → Down2 → Enc3 → Down3 → Enc4
                                                              ↓
                                              Up3 + CBAM3D(Enc3) → Dec3
                                              Up2 + CBAM3D(Enc2) → Dec2
                                              Up1 + CBAM3D(Enc1) → Dec1
                                                              ↓
                                               Auxiliary heads (Dec2, Dec3)
                                               for deep supervision (train only)
                                                              ↓
                                                Head (Conv3d + Tanh)
                                                              ↓
                                                  Synthetic CT (1, D, H, W)
```

| Hyperparameter | Value |
|---|---|
| Base channels | 32 → 64 → 128 → 256 |
| SSM state dim (`d_state`) | 16 |
| Parameters | ~20–22 M |
| Multi-scale depth conv dilations | [1, 2, 4, 8] |
| Upsampling | Trilinear interpolation + Conv3d |
| Gradient checkpointing | Enabled |
| Deep supervision | Auxiliary heads at Dec2 and Dec3 |
| Test-Time Augmentation | Enabled |

---

## Environment Setup

```bash
conda env create -f environment.yml
conda activate triplane
```

Or manually:

```bash
conda create -n triplane python=3.10 -y
conda activate triplane
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy scikit-image monai
pip install causal-conv1d>=1.2.0 mamba-ssm
```

---

## Training

```bash
bash run_training_triplane.sh
```

Or directly:

```bash
python train.py \
    --data_dir /DATA/divyansh/mc_ddpm_data/brain_npy \
    --epochs 500 \
    --batch_size 1 \
    --lr 5e-4 \
    --base_ch 32 \
    --save_dir ./checkpoints_triplane
```

### Resume from checkpoint

```bash
bash resume_training_triplane.sh
```

---

## Evaluation

```bash
bash run_eval_trimamba.sh
```

Or directly:

```bash
python evaluate.py \
    --data_dir /DATA/divyansh/mc_ddpm_data/brain_npy \
    --checkpoint ./checkpoints_triplane/triplane_best.pth \
    --save_preds

python evaluate_dosimetric.py \
    --pred_dir ./predictions_triplane \
    --output_csv ./predictions_triplane/dosimetric_metrics.csv
```

---

## Results

### Test-Set Performance

| Metric | Score | Std Dev |
|---|---|---|
| **MAE** | **0.0445** | ± 0.0074 |
| **RMSE** | **0.1041** | ± 0.0178 |
| **PSNR** | **25.79 dB** | ± 1.42 dB |
| **SSIM** | **0.8561** | ± 0.0358 |

TriPlane achieves the **highest SSIM (0.8561)** among all Mamba variants, indicating superior structural retention.

### Dosimetric Metrics

| Metric | TriPlane Mamba | TriAxial Mamba | Best |
|---|---|---|---|
| PSNR (3D) | **25.79 dB** | 25.71 dB | TriPlane |
| PSNR (2D) | **26.40 dB** | 26.32 dB | TriPlane |
| PSNR (1D) | **33.77 dB** | 33.32 dB | TriPlane |
| SSIM | **0.8502** | 0.8483 | TriPlane |
| Air MAE | **57.36 HU** | 60.77 HU | TriPlane |
| Soft Tissue MAE | 38.87 HU | **38.31 HU** | TriAxial |
| Bone MAE | **189.39 HU** | 196.20 HU | TriPlane |
| RED MAE | **0.04837** | 0.05012 | TriPlane |
| Gamma (1% / 1mm) | **90.61%** | 88.71% | TriPlane |
| Gamma (2% / 2mm) | **99.14%** | 98.83% | TriPlane |

> TriPlane Mamba is the best overall performer, crossing the clinical 90% threshold for the strict 1%/1mm Gamma criterion.

---

## Sample Visualizations

Training dashboard at epoch 500:

![Training Dashboard epoch 500](checkpoints_triplane/visuals/dashboard_epoch_500.png)

Earlier epoch (epoch 100) for comparison:

![Training Dashboard epoch 100](checkpoints_triplane/visuals/dashboard_epoch_100.png)

> All epoch dashboards are in [`checkpoints_triplane/visuals/`](checkpoints_triplane/visuals/).

---

## Model Weights

| File | Notes |
|---|---|
| `checkpoints_triplane/triplane_best.pth` | Best validation loss — use for inference |
| `checkpoints_triplane/triplane_epoch500.pth` | Final epoch |
| `checkpoints_triplane/triplane_epoch*.pth` | Intermediate checkpoints every 50 epochs |

---

## Full Technical Report

See [Triplane_Mamba_Report.md](Triplane_Mamba_Report.md) for the complete architecture source code, training details, and methodology including the full `TriPlaneMambaUNet` implementation.
