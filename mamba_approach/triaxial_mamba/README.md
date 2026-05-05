# TriAxial Mamba (TriMamba-UNet V2) — MRI-to-CT Synthesis

TriAxial Mamba scans the 3D feature volume along the **D (Depth)**, **H (Height)**, and **W (Width)** axes sequentially and bidirectionally using Mamba SSMs. This avoids flattening the entire 3D volume at once, reducing peak VRAM while still capturing full 3D global context.

---

## Folder Structure

```
triaxial_mamba/
├── README.md
├── Triaxial_Mamba_Report.md           # Full technical report
├── models.py                          # TriMamba-UNet V2 architecture
├── train.py                           # Training script
├── evaluate.py                        # Inference + MAE / PSNR / SSIM
├── evaluate_dosimetric.py             # RED / Gamma-index dosimetric analysis
├── dataset.py                         # Data loader
├── losses.py                          # Loss functions
├── visualize.py                       # Generates comparison PNGs
├── environment.yml                    # Conda environment spec
├── run_training_trimamba.sh
├── resume_training_trimamba.sh
├── run_eval_trimamba.sh
├── architecture.html                  # Interactive architecture diagram
│
├── checkpoints_trimamba/
│   ├── trimamba_best.pth
│   ├── trimamba_epoch50.pth … trimamba_epoch500.pth
│   └── visuals/                       # Per-epoch training dashboards
│
├── predictions_trimamba/
│   ├── dosimetric_metrics.csv
│   └── brain_001.npy … brain_037.npy
│
└── results/
    └── dashboard_final.png            # Final epoch training dashboard
```

---

## End-to-End Architecture

```mermaid
flowchart TD
    Input["MRI Input · (1, 64, 192, 192)"]
    Input --> Stem["Stem · 2× ConvNormAct\n1 → 32 ch"]

    Stem --> E1["Enc1 · TriAxialMambaBlock\n32 ch · full res"]
    E1   --> D1["Down1 · Stride-2 Conv"]
    D1   --> E2["Enc2 · TriAxialMambaBlock\n64 ch · ½ res"]
    E2   --> D2["Down2 · Stride-2 Conv"]
    D2   --> E3["Enc3 · TriAxialMambaBlock\n128 ch · ¼ res"]
    E3   --> D3["Down3 · Stride-2 Conv"]
    D3   --> E4["Enc4 · Bottleneck · TriAxialMambaBlock\n256 ch · ⅛ res"]

    E4   --> U3["Up3 · Trilinear + Conv"]
    E3   -->|"CBAM3D → skip concat"| U3
    U3   --> Dec3["Dec3 · TriAxialMambaBlock · 128 ch"]
    Dec3 --> Aux3["Aux Head 3\n(deep supervision · train only)"]

    Dec3 --> U2["Up2 · Trilinear + Conv"]
    E2   -->|"CBAM3D → skip concat"| U2
    U2   --> Dec2["Dec2 · TriAxialMambaBlock · 64 ch"]
    Dec2 --> Aux2["Aux Head 2\n(deep supervision · train only)"]

    Dec2 --> U1["Up1 · Trilinear + Conv"]
    E1   -->|"CBAM3D → skip concat"| U1
    U1   --> Dec1["Dec1 · TriAxialMambaBlock · 32 ch"]

    Dec1 --> Head["Output Head · Conv3d + Tanh"]
    Head --> Out["Synthetic CT · (1, 64, 192, 192)"]
```

### TriAxialMambaBlock — per-axis bidirectional scanning

```mermaid
flowchart TD
    In["Input · (B, C, D, H, W)"]

    In --> D_fwd["SSM_D_fwd\nscan along Depth axis"]
    In --> D_bwd["SSM_D_bwd\n← reverse scan"]
    D_fwd & D_bwd --> Yd["y_d = fwd + bwd"]

    In --> H_fwd["SSM_H_fwd\nscan along Height axis"]
    In --> H_bwd["SSM_H_bwd\n← reverse scan"]
    H_fwd & H_bwd --> Yh["y_h = fwd + bwd"]

    In --> W_fwd["SSM_W_fwd\nscan along Width axis"]
    In --> W_bwd["SSM_W_bwd\n← reverse scan"]
    W_fwd & W_bwd --> Yw["y_w = fwd + bwd"]

    Yd & Yh & Yw --> Fuse["Fusion Conv3d\n[y_d ‖ y_h ‖ y_w] → C channels"]
    Fuse --> Add["+ Residual skip"]
    Add --> Out["Output · (B, C, D, H, W)"]
```

### CBAM3D on skip connections

```mermaid
flowchart LR
    Enc["Encoder feature\n(B, C, D, H, W)"]
    Enc --> CA["Channel Attention\nGlobalAvgPool + FC + Sigmoid"]
    CA  --> SA["Spatial Attention\nAvgPool + MaxPool → Conv → Sigmoid"]
    SA  --> Filtered["Filtered skip feature"]
    Filtered --> Concat["Concat with decoder feature"]
```

---

## Training Pipeline

```mermaid
flowchart LR
    Data["brain_npy\n(MRI + CT pairs)"]
    Data --> Patch["Random Patch\n64 × 192 × 192"]
    Patch --> Aug["Test-Time Aug\n(flip · rot · enabled at eval)"]
    Aug --> Model["TriAxial Mamba\n~18 M params\nGrad Checkpointing ON"]
    Model --> Loss["Loss\nepoch < 100 → wMAE\nepoch ≥ 100 → wMAE + SSIM + AFP\n+ deep supervision weights 0.4 · 0.2"]
    Loss --> Opt["Adam\nβ₁=0.9 β₂=0.999\nlr₀ = 5 × 10⁻⁴"]
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
| Batch size | 1 |
| Patch size | (64, 192, 192) D×H×W |
| Base channels | 32 → 64 → 128 → 256 |
| SSM state dim | 16 |
| Parameters | ~18 M |
| Gradient checkpointing | Enabled (~60% activation memory saved) |
| Deep supervision | Aux heads at Dec2 and Dec3 (weights 0.4, 0.2) |
| Test-Time Augmentation | Enabled at inference |
| Mixed precision | AMP (fp16) |
| Upsampling | Trilinear interpolation + Conv3d |
| Checkpoint save | Every 50 epochs + best val |

### Loss Schedule

| Phase | Epochs | Components |
|---|---|---|
| Warmup | 1 – 99 | wMAE (Bone 3.0 · Soft tissue 1.5 · Air 0.5) |
| Full | 100 – 500 | wMAE + SSIM + AFP + deep supervision terms |

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

> If `mamba-ssm` fails (CUDA mismatch), the code falls back to a GRU-based SSM block automatically.

---

## Running

```bash
bash run_training_trimamba.sh

# Or directly:
python train.py \
    --data_dir /DATA/divyansh/mc_ddpm_data/brain_npy \
    --epochs 500 \
    --batch_size 1 \
    --lr 5e-4 \
    --base_ch 32 \
    --save_dir ./checkpoints_trimamba
```

### Resume

```bash
bash resume_training_trimamba.sh
```

### Evaluate

```bash
bash run_eval_trimamba.sh

# Or directly:
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

### Image Quality (37 test cases)

| Metric | Score | Std Dev |
|---|---|---|
| MAE | 0.0458 | ± 0.0070 |
| RMSE | 0.1041 | — |
| PSNR | 25.71 dB | ± 1.31 dB |
| SSIM | 0.8540 | ± 0.0341 |

### Dosimetric Metrics

| Metric | TriAxial Mamba |
|---|---|
| PSNR (3D) | 25.71 dB |
| PSNR (2D) | 26.32 dB |
| PSNR (1D) | 33.32 dB |
| SSIM | 0.8483 |
| Air MAE | 60.77 HU |
| Soft Tissue MAE | **38.31 HU** ← best among all Mamba variants |
| Bone MAE | 196.20 HU |
| RED MAE | 0.05012 |
| Gamma (1% / 1mm) | 88.71% |
| Gamma (2% / 2mm) | 98.83% |

---

## Sample Results

Final epoch training dashboard (Input MRI · Generated CT · Target CT · Error Map):

![Final epoch dashboard](results/dashboard_final.png)

---

## Full Technical Report

See [Triaxial_Mamba_Report.md](Triaxial_Mamba_Report.md) for complete architecture source, training details, and ablation analysis.
