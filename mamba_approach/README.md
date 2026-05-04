# Mamba Approach — MRI-to-CT Synthesis

This folder contains **four Mamba-based architectures** for 3D MRI → Synthetic CT synthesis. All share the same task (brain MRI → CT) and dataset, enabling direct comparison.

---

## Sub-approaches at a Glance

| Folder | Architecture | MAE | PSNR | SSIM |
|---|---|---|---|---|
| [SegMamba/](SegMamba/) | Hybrid CNN + Mamba U-Net | 0.0480 | 24.79 dB | 0.8432 |
| [UMamba/](UMamba/) | U-Net with Mamba SSM blocks | 0.0443 | 25.23 dB | **0.8531** |
| [triaxial_mamba/](triaxial_mamba/) | TriAxial bidirectional Mamba scans | 0.0458 | 25.71 dB | 0.8540 |
| [triplane_mamba/](triplane_mamba/) | TriPlane 2D Mamba scans + multiscale depth conv | **0.0445** | **25.79 dB** | **0.8561** |

**TriPlane Mamba** is the best overall performer. **U-Mamba** is the best among the two baseline variants.

---

## Repository Structure

```
mamba_approach/
├── README.md                          ← you are here
│
├── src/                               # Shared source code (SegMamba + UMamba)
│   ├── models.py                      # SegMamba + UMamba architecture definitions
│   ├── train.py                       # Training script (--model flag selects arch)
│   ├── evaluate.py                    # Inference + MAE / PSNR / SSIM
│   ├── dataset.py                     # Brain NPY data loader
│   ├── losses.py                      # wMAE + SSIM + AFP compound loss
│   ├── visualize.py                   # Comparison PNG generation
│   └── dosometric.py                  # Dosimetric / RED / Gamma analysis
│
├── docs/                              # Reports and interactive diagrams
│   ├── architecture.html              # Interactive architecture diagram
│   └── dosimetric_analysis.md        # SegMamba vs UMamba dosimetric report
│
├── SegMamba/                          ← SegMamba model, results, scripts
│   ├── README.md
│   ├── run_training.sh
│   ├── run_eval.sh
│   ├── run_viz.sh
│   ├── training_output.log
│   ├── segmamba_report.md
│   ├── checkpoints/
│   │   ├── segmamba_best.pth
│   │   ├── segmamba_epoch50.pth … segmamba_epoch500.pth
│   │   ├── segmamba_train_log.txt
│   │   ├── segmamba_test_results.txt
│   │   └── visuals/                   # 500 epoch training dashboards
│   ├── predictions/                   # 37 test .npy prediction files
│   └── visualizations/                # 37 MRI|PredCT|RealCT comparison PNGs
│
├── UMamba/                            ← U-Mamba model, results, scripts
│   ├── README.md
│   ├── run_training.sh
│   ├── run_eval.sh
│   ├── run_viz.sh
│   ├── training_output.log
│   ├── umamba_report.md
│   ├── unet_umamba_report.md
│   ├── diffusion_mamba_models.py
│   ├── main_diffusionUmamba.py
│   ├── checkpoints/
│   │   ├── umamba_best.pth
│   │   ├── umamba_epoch50.pth … umamba_epoch500.pth
│   │   ├── umamba_train_log.txt
│   │   ├── umamba_test_results.txt
│   │   └── visuals/                   # 500 epoch training dashboards
│   └── predictions/                   # 37 test .npy prediction files
│
├── triaxial_mamba/                    ← TriAxial Mamba model, results, scripts
│   ├── README.md
│   ├── Triaxial_Mamba_Report.md
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── evaluate_dosimetric.py
│   ├── dataset.py
│   ├── losses.py
│   ├── visualize.py
│   ├── environment.yml
│   ├── run_training_trimamba.sh
│   ├── resume_training_trimamba.sh
│   ├── run_eval_trimamba.sh
│   ├── training_trimamba_output.log
│   ├── architecture.html
│   ├── checkpoints_trimamba/
│   │   ├── trimamba_best.pth
│   │   ├── trimamba_epoch*.pth
│   │   └── visuals/                   # Epoch training dashboards
│   └── predictions_trimamba/
│       ├── dosimetric_metrics.csv
│       └── brain_*.npy
│
└── triplane_mamba/                    ← TriPlane Mamba model, results, scripts
    ├── README.md
    ├── Triplane_Mamba_Report.md
    ├── models.py
    ├── train.py
    ├── evaluate.py
    ├── evaluate_dosimetric.py
    ├── dataset.py
    ├── losses.py
    ├── visualize.py
    ├── environment.yml
    ├── run_training_triplane.sh
    ├── resume_training_triplane.sh
    ├── run_eval_trimamba.sh
    ├── training_triplane_output.log
    ├── architecture.html
    ├── checkpoints_triplane/
    │   ├── triplane_best.pth
    │   ├── triplane_epoch*.pth
    │   └── visuals/                   # Epoch training dashboards
    └── predictions_triplane/
        ├── dosimetric_metrics.csv
        └── brain_*.npy
```

---

## Shared Source (SegMamba & UMamba) — `src/`

| File | Purpose |
|---|---|
| `src/models.py` | SegMamba and UMamba class definitions |
| `src/train.py` | Training loop — select model with `--model segmamba` or `--model umamba` |
| `src/evaluate.py` | Sliding-window inference, computes MAE / PSNR / SSIM |
| `src/dataset.py` | Loads brain `.npy` files, shape `(2, 192, 192, 96)` |
| `src/losses.py` | Stage-1 wMAE + Stage-2 wMAE+SSIM+AFP |
| `src/visualize.py` | Generates MRI | Pred CT | Real CT comparison PNGs |
| `src/dosometric.py` | Tissue-specific MAE, RED, Gamma-Index analysis |

TriAxial and TriPlane variants have their **own self-contained** source in their subfolders.

### Documentation — `docs/`

| File | Purpose |
|---|---|
| `docs/architecture.html` | Interactive architecture diagram (open in browser) |
| `docs/dosimetric_analysis.md` | SegMamba vs UMamba dosimetric comparison report |

---

## Quick Start — SegMamba / UMamba

### Environment

```bash
conda create -n mamba_ct python=3.10 -y
conda activate mamba_ct
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy scikit-image monai
pip install causal-conv1d>=1.2.0 mamba-ssm
```

### Train

```bash
# SegMamba (run from mamba_approach/)
python src/train.py --data_dir /DATA/divyansh/mc_ddpm_data/brain_npy \
    --model segmamba --epochs 500 --batch_size 2 --lr 5e-4 \
    --save_dir ./SegMamba/checkpoints

# UMamba (run from mamba_approach/)
python src/train.py --data_dir /DATA/divyansh/mc_ddpm_data/brain_npy \
    --model umamba --epochs 500 --batch_size 1 --lr 5e-4 \
    --save_dir ./UMamba/checkpoints
```

Or use the launch scripts from within each subfolder:

```bash
cd SegMamba && bash run_training.sh
cd UMamba  && bash run_training.sh
```

### Evaluate

```bash
# Run from mamba_approach/
python src/evaluate.py --data_dir /DATA/divyansh/mc_ddpm_data/brain_npy \
    --checkpoint ./SegMamba/checkpoints/segmamba_best.pth \
    --model segmamba --save_preds

python src/evaluate.py --data_dir /DATA/divyansh/mc_ddpm_data/brain_npy \
    --checkpoint ./UMamba/checkpoints/umamba_best.pth \
    --model umamba --save_preds
```

---

## Full Metrics Comparison (All Variants)

| Metric | SegMamba | UMamba | TriAxial | TriPlane |
|---|---|---|---|---|
| MAE | 0.0480 | 0.0443 | 0.0458 | **0.0445** |
| PSNR (3D) | 24.79 dB | 25.23 dB | 25.71 dB | **25.79 dB** |
| SSIM | 0.8432 | 0.8509 | 0.8540 | **0.8561** |
| Bone MAE | 208.52 HU | 192.50 HU | 196.20 HU | **189.39 HU** |
| Soft Tissue MAE | 38.15 HU | 35.43 HU | **38.31 HU** | 38.87 HU |
| RED MAE | 0.05208 | 0.04794 | 0.05012 | **0.04837** |
| Gamma (1%/1mm) | 91.61% | 93.26% | 88.71% | **90.61%** |
| Gamma (2%/2mm) | 99.35% | 99.55% | 98.83% | **99.14%** |

---

## Sample Output

![SegMamba brain 001 comparison](SegMamba/visualizations/brain_001_comparison.png)

> More comparisons: [SegMamba/visualizations/](SegMamba/visualizations/)
