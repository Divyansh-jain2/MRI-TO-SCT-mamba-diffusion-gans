# GAN Approaches — MRI-to-CT Synthesis

Two **paired and unpaired image-to-image translation** models for MRI → Synthetic CT generation: **Pix2Pix** (supervised, paired) and **UNIT** (unsupervised, shared-latent-space). Both operate on 2D slices extracted from 3D brain volumes and are evaluated under 5-fold cross-validation.

---

## Folder Structure

```
gan_approaches/
├── README.md
├── requirements.txt
├── GAN_families.csv              # Summary table of GAN families
│
├── configs/                       # YAML experiment configs
│   ├── pix2pix_train.yaml
│   ├── pix2pix_test.yaml
│   ├── unit_train.yaml
│   ├── unit_test.yaml
│   └── create_CrossValidation_2d.yaml
│
├── data/                          # Data path references for k-fold CSVs
│
├── src/
│   ├── model/
│   │   ├── Pix2Pix/               # Pix2Pix model code
│   │   │   ├── generator_model.py
│   │   │   ├── discriminator_model.py
│   │   │   ├── util_model.py
│   │   │   ├── train_kfold.py
│   │   │   └── test.py
│   │   └── Unit/                  # UNIT model code
│   │       ├── models.py          # Encoder, Generator, Discriminator
│   │       ├── generator_model.py
│   │       ├── discriminator_model.py
│   │       ├── util_model.py
│   │       ├── train_kfold.py
│   │       └── test.py
│   └── utils/
│       ├── util_general.py        # Checkpoint save/load, LR update
│       └── util_data.py           # Data loading and preprocessing
│
├── models/                        # Saved model checkpoints
│   ├── pix2pix/0/                 # Fold 0 weights
│   └── unit/0/
│
├── results/                       # Training outputs and test predictions
├── logs/                          # Training logs
└── scripts/                       # Evaluation and export utilities
    ├── evaluate_dosimetry_gan.py
    ├── evaluate_pix2pix_volume_psnr.py
    ├── evaluate_unit_volume_psnr.py
    ├── export_pix2pix_comparison_panels.py
    └── export_unit_comparison_panels.py
```

---

## Model 1 — Pix2Pix

A **supervised paired image-to-image** translation model. The generator learns a direct MRI → CT mapping from aligned pairs; the discriminator enforces photorealism via a PatchGAN loss.

### Architecture

```mermaid
flowchart TD
    MRI["MRI slice · (B, 1, 256, 256)"]

    subgraph Gen["U-Net Generator"]
        E0["InitDown · Conv4×4 · 64 ch · LeakyReLU"]
        E1["Down1 · Conv4×4↓ · 128 ch · BN · LeakyReLU"]
        E2["Down2 · 256 ch"]
        E3["Down3 · 512 ch"]
        E4["Down4 · 512 ch"]
        E5["Down5 · 512 ch"]
        E6["Down6 · 512 ch"]
        BN["Bottleneck · Conv4×4 · 512 ch · ReLU → 1×1"]
        U1["Up1 · ConvTranspose4×4 · 512 ch · Dropout 0.5"]
        U2["Up2 · 512+512 → 512 ch · Dropout 0.5"]
        U3["Up3 · 512+512 → 512 ch · Dropout 0.5"]
        U4["Up4 · 512+512 → 512 ch"]
        U5["Up5 · 512+512 → 256 ch"]
        U6["Up6 · 256+256 → 128 ch"]
        U7["Up7 · 128+128 → 64 ch"]
        FU["FinalUp · ConvTranspose4×4 · 1 ch · Sigmoid"]
    end

    subgraph Disc["PatchGAN Discriminator"]
        Cat["Concat(MRI, CT) → 2 ch"]
        D1["Conv4×4 s2 · 64 ch · LeakyReLU"]
        D2["Conv4×4 s2 · 128 ch · BN · LeakyReLU"]
        D3["Conv4×4 s2 · 256 ch · BN · LeakyReLU"]
        D4["Conv4×4 s1 · 512 ch · BN · LeakyReLU"]
        D5["Conv4×4 s1 · 1 ch · patch score"]
    end

    MRI --> E0 --> E1 --> E2 --> E3 --> E4 --> E5 --> E6 --> BN
    BN --> U1 --> U2 --> U3 --> U4 --> U5 --> U6 --> U7 --> FU
    FU --> SCT["Synthetic CT"]

    MRI & SCT --> Cat --> D1 --> D2 --> D3 --> D4 --> D5
```

### Loss

```
L_total = BCE(D(MRI, G(MRI)), 1) + λ · L1(G(MRI), CT_gt)
λ = 100
```

### Pix2Pix Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | Adam (β₁=0.5, β₂=0.999) |
| Learning rate | 2 × 10⁻⁴ |
| Epochs | 40 |
| Early stopping | patience = 8 |
| Warm-up epochs | 5 |
| Batch size | 16 |
| Image size | 256 × 256 |
| L1 lambda | 100 |
| GP lambda | 10 |
| Cross-validation | 5-fold |
| Generator features | 64 → 128 → 256 → 512 |
| Dropout in decoder | 0.5 (first 3 up-blocks) |

---

## Model 2 — UNIT (Unsupervised Image-to-Image Translation)

An **unsupervised** model that learns a shared latent space between MRI and CT domains. Each domain has a private encoder+generator pair; a shared residual block enforces domain-invariant features. Translation requires no paired training data — alignment is enforced via cycle-consistency and KL divergence.

### Architecture

```mermaid
flowchart TD
    MRI["MRI · domain X₁"]
    CT["CT · domain X₂"]

    subgraph E1_block["Encoder E1 (MRI)"]
        ReflPad1["ReflectionPad2d(3)"]
        Conv7_1["Conv 7×7 · 64 ch · InstanceNorm · LeakyReLU"]
        Down1A["Conv4×4 s2 · 128 ch · InstanceNorm · ReLU"]
        Down1B["Conv4×4 s2 · 256 ch · InstanceNorm · ReLU"]
        Res1["3× ResidualBlock · 256 ch"]
        Shared1["Shared ResBlock"]
        Reparam1["Reparameterization · z₁ = μ₁ + ε"]
    end

    subgraph E2_block["Encoder E2 (CT)"]
        ReflPad2["ReflectionPad2d(3)"]
        Conv7_2["Conv 7×7 · 64 ch · InstanceNorm · LeakyReLU"]
        Down2A["Conv4×4 s2 · 128 ch · InstanceNorm · ReLU"]
        Down2B["Conv4×4 s2 · 256 ch · InstanceNorm · ReLU"]
        Res2["3× ResidualBlock · 256 ch"]
        Shared2["Shared ResBlock (same weights)"]
        Reparam2["Reparameterization · z₂ = μ₂ + ε"]
    end

    subgraph G1_block["Generator G1 (→ MRI)"]
        SharedG1["Shared ResBlock"]
        ResG1["3× ResidualBlock"]
        Up1A["ConvTranspose4×4 s2 · 128 ch · InstanceNorm · LeakyReLU"]
        Up1B["ConvTranspose4×4 s2 · 64 ch · InstanceNorm · LeakyReLU"]
        Out1["ReflPad + Conv7×7 · 1 ch · Tanh"]
    end

    subgraph G2_block["Generator G2 (→ CT)"]
        SharedG2["Shared ResBlock (same weights)"]
        ResG2["3× ResidualBlock"]
        Up2A["ConvTranspose4×4 s2 · 128 ch · InstanceNorm · LeakyReLU"]
        Up2B["ConvTranspose4×4 s2 · 64 ch · InstanceNorm · LeakyReLU"]
        Out2["ReflPad + Conv7×7 · 1 ch · Tanh"]
    end

    MRI --> E1_block --> z1["z₁ (shared space)"]
    CT --> E2_block --> z2["z₂ (shared space)"]

    z1 --> G2_block --> SCT["MRI → Synthetic CT"]
    z2 --> G1_block --> SMRI["CT → Synthetic MRI (cycle)"]
    z1 --> G1_block
    z2 --> G2_block
```

### Loss

```
L_total = λ₀·L_GAN(D₁, D₂)
        + λ₁·KL(z₁) + λ₂·L1(G₁(E₂(CT)), MRI)   ← cycle MRI
        + λ₃·KL(z₂) + λ₄·L1(G₂(E₁(MRI)), CT)   ← cycle CT
```

### UNIT Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1 × 10⁻⁴ |
| Epochs | 40 |
| Early stopping | patience = 8 |
| Warm-up epochs | 5 |
| Batch size | 8 |
| Image size | 256 × 256 |
| Encoder base dim | 64 |
| Downsample levels | 2 |
| λ₀ (GAN) | 10 |
| λ₁ (KL₁) | 0.1 |
| λ₂ (cycle₁) | 100 |
| λ₃ (KL₂) | 0.1 |
| λ₄ (cycle₂) | 100 |
| Cross-validation | 5-fold |

---

## Running

### Train Pix2Pix

```bash
python src/model/Pix2Pix/train_kfold.py --config configs/pix2pix_train.yaml

# Single fold (e.g. fold 0):
FOLD_IDX=0 python src/model/Pix2Pix/train_kfold.py --config configs/pix2pix_train.yaml
```

### Train UNIT

```bash
python src/model/Unit/train_kfold.py --config configs/unit_train.yaml

# Single fold:
FOLD_IDX=0 python src/model/Unit/train_kfold.py --config configs/unit_train.yaml
```

### Test / Inference

```bash
python src/model/Pix2Pix/test.py --config configs/pix2pix_test.yaml
python src/model/Unit/test.py    --config configs/unit_test.yaml
```

### Volumetric Evaluation

```bash
python scripts/evaluate_pix2pix_volume_psnr.py
python scripts/evaluate_unit_volume_psnr.py
python scripts/evaluate_dosimetry_gan.py
```

---

## Documents

- `GAN_families.csv` — table summarising the characteristics of the two GAN families implemented here
- `models/pix2pix/info.csv` and `models/unit/info.csv` — per-run experiment metadata (batch size, epochs, img dim)

---

## Contact

For questions and comments, feel free to contact: b23193@students.iitmandi.ac.in, b23334@students.iitmandi.ac.in