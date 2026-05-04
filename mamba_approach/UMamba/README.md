# U-Mamba — MRI-to-CT Synthesis

U-Mamba replaces transformer self-attention blocks inside the U-Net encoder with **Mamba State Space Model (SSM)** blocks, giving linear-time long-range context modeling at lower memory cost. Trained for MRI → Synthetic CT on brain data.

---

## Folder Contents

```
UMamba/
├── README.md
├── run_training.sh                # Training launch script
├── run_eval.sh                    # Evaluation launch script
├── run_viz.sh                     # Visualization generation script
├── training_output.log            # Full training log (500 epochs)
├── umamba_report.md               # U-Mamba architecture report
├── unet_umamba_report.md          # UNet-Mamba variant report
├── diffusion_mamba_models.py      # Diffusion-UMamba model definitions
├── main_diffusionUmamba.py        # Diffusion-UMamba training entry point
│
├── checkpoints/
│   ├── umamba_best.pth            # Best model weights (lowest val loss)
│   ├── umamba_epoch50.pth         # Checkpoints every 50 epochs
│   ├── ...
│   ├── umamba_epoch500.pth
│   ├── umamba_train_log.txt       # Per-epoch loss values
│   ├── umamba_test_results.txt    # Final test-set metrics
│   └── visuals/                   # Per-epoch training dashboards (500 PNGs)
│       ├── dashboard_epoch_001.png
│       └── ...  dashboard_epoch_500.png
│
└── predictions/                   # Test-set prediction arrays
    └── brain_001.npy … brain_037.npy
```

> Shared source code lives in [`../src/`](../src/) — `models.py`, `train.py`, `evaluate.py`, `dataset.py`, `losses.py`, `visualize.py`, `dosometric.py`.

---

## Architecture

U-Mamba uses the same 4-level U-Net skeleton as SegMamba. The core block is a **UMambaBlock**: residual CNN followed by a Mamba SSM that processes flattened 3D tokens as a 1D sequence.

```
MRI (1, D, H, W)
    └─ Stem (ConvNormAct × 2)
        ├─ Enc1 ──Down1──> Enc2 ──Down2──> Enc3 ──Down3──> Enc4 (bottleneck)
        │                                                        ↓
        └──────────────────────────────────────────────  Up3 + skip(Enc3) → Dec3
                                                         Up2 + skip(Enc2) → Dec2
                                                         Up1 + skip(Enc1) → Dec1
                                                                  ↓
                                                          Head (Conv3d + Tanh)
                                                                  ↓
                                                         Synthetic CT (1, D, H, W)
```

| Hyperparameter | Value |
|---|---|
| Base channels | 32 → 64 → 128 → 256 |
| SSM state dim (`d_state`) | 16 |
| Patch size | (64, 192, 192) D×H×W |
| Batch size | 1 |

---

## Training

```bash
# From inside UMamba/
bash run_training.sh

# Or directly:
python ../src/train.py \
    --data_dir /DATA/divyansh/mc_ddpm_data/brain_npy \
    --model umamba \
    --epochs 500 \
    --batch_size 1 \
    --lr 5e-4 \
    --base_ch 32 \
    --save_dir ./checkpoints
```

### Loss Schedule

| Phase | Epochs | Loss Components |
|---|---|---|
| Warmup | 1 – 99 | Weighted HU-aware MAE |
| Full | 100 – 500 | wMAE + SSIM + AFP |

---

## Evaluation

```bash
bash run_eval.sh

# Or directly:
python ../src/evaluate.py \
    --data_dir /DATA/divyansh/mc_ddpm_data/brain_npy \
    --checkpoint ./checkpoints/umamba_best.pth \
    --model umamba \
    --save_preds
```

---

## Results

### Test-Set Metrics

| Metric | Score | Std Dev |
|---|---|---|
| MAE | 0.0443 | ± 0.0075 |
| PSNR | 25.23 dB | ± 1.30 dB |
| SSIM | 0.8531 | ± 0.0358 |

U-Mamba outperforms SegMamba on all three standard metrics.

### Dosimetric Performance vs SegMamba

| Metric | UMamba | SegMamba | Delta |
|---|---|---|---|
| PSNR (3D) | **25.23 dB** | 24.79 dB | +0.44 dB |
| PSNR (2D) | **25.78 dB** | 25.42 dB | +0.36 dB |
| PSNR (1D) | **33.88 dB** | 32.84 dB | +1.04 dB |
| SSIM | **0.8509** | 0.8374 | +0.0135 |
| Air MAE | **60.53 HU** | 65.74 HU | −5.21 HU |
| Soft Tissue MAE | **35.43 HU** | 38.15 HU | −2.72 HU |
| Bone MAE | **192.50 HU** | 208.52 HU | −16.02 HU |
| RED MAE | **0.04794** | 0.05208 | −0.00414 |
| Gamma (1% / 1mm) | **93.26%** | 91.61% | +1.65% |
| Gamma (2% / 2mm) | **99.55%** | 99.35% | +0.20% |

---

## Visualizations

Training dashboard at epoch 500:

![Training Dashboard epoch 500](checkpoints/visuals/dashboard_epoch_500.png)

---

## Model Weights

| File | Notes |
|---|---|
| `checkpoints/umamba_best.pth` | Best validation checkpoint — use for inference |
| `checkpoints/umamba_epoch500.pth` | Final epoch |
| `checkpoints/umamba_epoch*.pth` | Intermediate saves every 50 epochs |
