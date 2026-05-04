# TriAxial Mamba (TriMamba-UNet V2) — MRI-to-CT Synthesis

TriAxial Mamba scans the 3D feature volume along the **D (Depth)**, **H (Height)**, and **W (Width)** axes sequentially and bidirectionally using Mamba SSMs. This avoids flattening the entire 3D volume at once, dramatically reducing peak VRAM usage while still capturing full 3D global context.

---

## Folder Structure

```
triaxial_mamba/
├── README.md                          ← you are here
├── Triaxial_Mamba_Report.md           # Full technical report with source code
├── models.py                          # TriMamba-UNet V2 architecture
├── train.py                           # Training script
├── evaluate.py                        # Inference + MAE / PSNR / SSIM
├── evaluate_dosimetric.py             # RED / Gamma-index dosimetric analysis
├── dataset.py                         # Data loader
├── losses.py                          # Loss functions
├── visualize.py                       # Generates comparison visualizations
├── environment.yml                    # Conda environment spec
├── run_training_trimamba.sh           # Training shell script
├── resume_training_trimamba.sh        # Resume from checkpoint
├── run_eval_trimamba.sh               # Evaluation shell script
├── training_trimamba_output.log       # Full training log
├── architecture.html                  # Interactive architecture diagram
│
├── checkpoints_trimamba/
│   ├── trimamba_best.pth              # Best model weights
│   ├── trimamba_epoch50.pth           # Epoch checkpoints (50 … 500)
│   ├── ...
│   └── visuals/                       # Per-epoch training dashboards
│       ├── dashboard_epoch_001.png
│       └── ...  dashboard_epoch_500.png
│
└── predictions_trimamba/
    ├── dosimetric_metrics.csv         # Full per-case dosimetric results
    └── brain_001.npy … brain_037.npy # Test-set prediction arrays
```

---

## Architecture

### TriAxialMambaBlock

The core innovation: instead of one monolithic 3D scan, the block runs **three independent bidirectional Mamba scans** — one per axis — then fuses the results with a 1×1 Conv3d:

```
Input (B, C, D, H, W)
    ├─ SSM_D_fwd / SSM_D_bwd  →  y_d  (scan along Depth)
    ├─ SSM_H_fwd / SSM_H_bwd  →  y_h  (scan along Height)
    └─ SSM_W_fwd / SSM_W_bwd  →  y_w  (scan along Width)
                  ↓
         Fusion Conv3d([y_d, y_h, y_w]) + Residual
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

**CBAM3D** on skip connections applies Channel + Spatial Attention before concatenation, filtering irrelevant encoder features.

| Hyperparameter | Value |
|---|---|
| Base channels | 32 → 64 → 128 → 256 |
| SSM state dim (`d_state`) | 16 |
| Parameters | ~18 M |
| Upsampling | Trilinear interpolation + Conv3d (no checkerboard artifacts) |
| Gradient checkpointing | Enabled (~60% activation memory saved) |
| Deep supervision | Auxiliary heads at Dec2 and Dec3 |
| Test-Time Augmentation | Enabled |

---

## Environment Setup

```bash
conda env create -f environment.yml
conda activate trimamba
```

Or manually:

```bash
conda create -n trimamba python=3.10 -y
conda activate trimamba
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy scikit-image monai
pip install causal-conv1d>=1.2.0 mamba-ssm
```

---

## Training

```bash
bash run_training_trimamba.sh
```

Or directly:

```bash
python train.py \
    --data_dir /DATA/divyansh/mc_ddpm_data/brain_npy \
    --epochs 500 \
    --batch_size 1 \
    --lr 5e-4 \
    --base_ch 32 \
    --save_dir ./checkpoints_trimamba
```

### Resume from checkpoint

```bash
bash resume_training_trimamba.sh
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
    --checkpoint ./checkpoints_trimamba/trimamba_best.pth \
    --save_preds

python evaluate_dosimetric.py \
    --pred_dir ./predictions_trimamba \
    --output_csv ./predictions_trimamba/dosimetric_metrics.csv
```

---

## Results

### Test-Set Performance

| Metric | Score | Std Dev |
|---|---|---|
| **MAE** | **0.0458** | ± 0.0070 |
| **RMSE** | **0.1041** | — |
| **PSNR** | **25.71 dB** | ± 1.31 dB |
| **SSIM** | **0.8540** | ± 0.0341 |

### Dosimetric Metrics

| Metric | TriAxial Mamba |
|---|---|
| PSNR (3D) | 25.71 dB |
| PSNR (2D) | 26.32 dB |
| PSNR (1D) | 33.32 dB |
| SSIM | 0.8483 |
| Air MAE | 60.77 HU |
| Soft Tissue MAE | **38.31 HU** ← best among Mamba variants |
| Bone MAE | 196.20 HU |
| RED MAE | 0.05012 |
| Gamma (1% / 1mm) | 88.71% |
| Gamma (2% / 2mm) | 98.83% |

> TriAxial Mamba achieves the best **Soft Tissue MAE** among all Mamba variants tested.

---

## Sample Visualizations

Training dashboard at epoch 500:

![Training Dashboard epoch 500](checkpoints_trimamba/visuals/dashboard_epoch_500.png)

Earlier epoch (epoch 100) for comparison:

![Training Dashboard epoch 100](checkpoints_trimamba/visuals/dashboard_epoch_100.png)

> All epoch dashboards are in [`checkpoints_trimamba/visuals/`](checkpoints_trimamba/visuals/).

---

## Model Weights

| File | Notes |
|---|---|
| `checkpoints_trimamba/trimamba_best.pth` | Best validation loss — use for inference |
| `checkpoints_trimamba/trimamba_epoch500.pth` | Final epoch |
| `checkpoints_trimamba/trimamba_epoch*.pth` | Intermediate checkpoints every 50 epochs |

---

## Full Technical Report

See [Triaxial_Mamba_Report.md](Triaxial_Mamba_Report.md) for complete architecture source code, training details, and methodology.
