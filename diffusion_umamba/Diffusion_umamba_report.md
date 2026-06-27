# Optimized UMamba Guided Diffusion for Synthetic CT Generation

This report provides a comprehensive technical and architectural overview of the optimized MRI-to-CT generation pipeline. It details the integration of **Denoising Diffusion Probabilistic Models (DDPM)** with a 3D **UMamba (State Space Model)** architecture, replacing the baseline Swin Transformer.

---

## 1. Project Overview & Motivation

The primary objective is **Image-to-Image translation**—converting anatomical MRI scans into Synthetic CT (sCT) volumes. This is a crucial task in medical physics (e.g., for dosimetric radiation therapy planning) as it bypasses the need for actual patient CT scans, minimizing radiation exposure and streamlining workflows.

Initially, the project utilized a 3D Swin-Transformer DDPM. However, Vision Transformers compute self-attention that scales quadratically $O(N^2)$ with the sequence length (spatial volume), imposing severe memory bottlenecks. To resolve this, the architecture has been upgraded to use **UMamba**, which integrates **State Space Models (SSMs)** that scale linearly $O(N)$ with sequence length, unlocking more efficient long-range contextual awareness for 3D medical volumes.

---

## 2. The Approach: Diffusion + UMamba

### 2.1 The Conditional Diffusion Process
The generative framework is a conditional Monte Carlo Improved DDPM. 
- **Training Phase:** The pipeline iteratively adds Gaussian noise to the Ground Truth CT patches over $T=1000$ timesteps. The UMamba network acts as the "noise-predictor" (or mean/variance predictor), tasked with reversing this process by conditioning on both the noisy CT and the corresponding structural MRI patch.
- **Inference Phase:** Starting from pure Gaussian noise, the model iteratively denoises the volume over $T$ steps, guided by the input MRI, to synthesize a highly realistic, structurally accurate CT volume.

### 2.2 Why Replace Swin Transformer with UMamba?
1. **Receptive Field Constraints:** Swin Transformers restrict self-attention to rigid local 3D windows (e.g., $4 \times 4 \times 4$). To communicate globally, they rely on complex Shifted-Window mechanisms which can lead to boundary artifacts.
2. **Infinite Receptive Field:** Mamba flattens the 3D volume into a 1D sequence and processes it dynamically using Bidirectional SSMs. This provides a theoretically infinite receptive field, allowing the network to observe global anatomical structures (like continuous bone boundaries) without rigid constraints.
3. **Linear Complexity:** Mamba processes sequences in $O(N)$ time. This allows us to achieve superior representation capabilities using a drop-in replacement that easily handles 3D volumetric data without the crushing memory overhead of global attention.

---

## 3. Architecture Diagrams: What is it?

The architecture merges the hierarchical spatial encoding of a **U-Net** with the sequential modeling power of **Mamba** and the temporal conditioning of **Diffusion**.

### 3.1 The Global U-Mamba Diffusion Flow
The exact data traversal inside the `UMamba` framework for a single `64x64x4` patch chunk is summarised below.

- **Input:** It takes a 2-channel tensor (1 channel for the guiding MRI, 1 channel for the noisy CT at timestep $t$).
- **Time Conditioning:** The scalar timestep $t$ is projected into a high-dimensional sinusoidal embedding, which is injected into *every single UMamba block* across the Encoder, Bottleneck, and Decoder.
- **Hierarchical Features:** The network uses Strided Convolutions (`Down`) to compress the spatial resolution while increasing channels, extracting deep semantic features. Transposed Convolutions (`ConvTranspose`) are used to upscale back to the original resolution.

### 3.2 Inside the UMamba Block (The Core Engine)
Instead of standard CNN convolutions or Vision Transformer Attention, every feature map in the network goes through a `UMambaBlock`. 

What happens here?
1. **Residual Convolution:** Extracts local spatial features.
2. **Time Shift:** Injects the current diffusion timestep $t$ by shifting the channel values so the block "knows" the current noise level.
3. **MambaBlock3D:** Flattens the 3D volume, runs a bidirectional Mamba (SSM) scan to gather infinite-context global information, and reconstructs the 3D volume.

---

## 4. Dosimetric & Clinical Analyses Results

After optimizing the UMamba architecture and executing the dosimetric testing pipeline across 37 subjects, the UMamba Diffusion model successfully surpassed the baseline Swin Transformer model across all major clinical metrics. To ensure a fair head-to-head comparison, both models utilized the exact same patch size (`64 × 64 × 4`).

### 4.1 Global Performance Comparison

| Metric (Average) | Baseline (Swin DDPM) | Optimized UMamba Diffusion | Improvement |
| :--- | :--- | :--- | :--- |
| **PSNR (dB)** | 21.09 dB | **22.49 dB** | **+1.40 dB** |
| **SSIM** | 0.720 | **0.768** | **+0.048** |
| **Gamma Pass Rate (1%/1mm)** | 88.74% | **90.52%** | **+1.78%** |
| **Gamma Pass Rate (2%/2mm)** | 98.51% | **99.03%** | **+0.52%** |

> [!TIP]
> The **Gamma Pass Rate (2%/2mm)** hitting **99.03%** indicates that the UMamba-generated CT scans are highly accurate for radiation therapy planning, as >95% is typically considered clinically acceptable.

### 4.2 Tissue-Specific Dosimetric Breakdown (UMamba)

A critical factor for sCT fidelity is how well the model predicts Hounsfield Units (HU) across different density tissues.

| Tissue Type | UMamba Mean Absolute Error (MAE) | Notes |
| :--- | :--- | :--- |
| **Air** | 73.00 HU | Low-density cavities (e.g., sinuses). |
| **Soft Tissue** | 49.81 HU | Excellent fidelity for standard organs and muscles. |
| **Bone** | 340.88 HU | High-density structures. Mamba's bidirectional scan improves bone continuity. |
| **RED MAE** | 0.066 | Relative Electron Density Error; a crucial metric for dose calculation algorithms. |

---

## 5. Core Code Breakdown

### 5.1 The 3D Mamba Block (`models.py`)
The `MambaBlock3D` module flattens the 3D volume into a 1D sequence and runs the State Space Model to learn global relationships, then reforms the 3D volume.

```python
class MambaBlock3D(nn.Module):
    def __init__(self, channels, d_state=16):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.ssm  = get_ssm_block(d_model=channels, d_state=d_state)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x):
        B, C, D, H, W = x.shape
        # 1. Flatten spatial dimensions into a sequence
        x_flat = x.flatten(2).permute(0, 2, 1)  # (B, D*H*W, C)
        residual = x_flat
        
        # 2. Apply Normalization and the SSM (Mamba) scan
        x_norm = self.norm(x_flat)
        x_ssm  = self.ssm(x_norm)
        
        # 3. Residual connection
        x_out  = self.proj(x_ssm) + residual
        
        # 4. Reshape back to the 3D volumetric tensor
        x_out = x_out.permute(0, 2, 1).reshape(B, C, D, H, W)
        return x_out
```

### 5.2 The Diffusion Training Loop (`main_umamba_diffusion.py`)
During training, the conditional DDPM requires both the noisy CT (`traintarget`) and the raw MRI (`traincondition`). The model predicts the clean CT (or noise).

```python
# 1. Instantiate the UMamba Model for Diffusion
A_to_B_model = UMamba(
    in_ch=2,             # 1 for MRI condition + 1 for Noisy CT
    out_ch=2,            # Predicts Gaussian Mean and Variance
    base_ch=64,
    is_diffusion=True,   # Enables Timestep Embedding logic inside UMambaBlocks
    strides=((2,2,2), (2,2,1), (2,2,1))
).to(device)

# 2. Core Iteration
for i, (x1, y1) in enumerate(data_loader1):
    traintarget    = y1.view(-1, 1, 64, 64, 4).to(device)   # Ground Truth CT
    traincondition = x1.view(-1, 1, 64, 64, 4).to(device)   # MRI Condition
    
    # Sample a random timestep t
    t, weights = schedule_sampler.sample(traincondition.shape[0], device)
    
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        # diffusion.training_losses handles noising the GT CT and calculating the divergence
        all_loss = diffusion.training_losses(A_to_B_model, traintarget, traincondition, t)
        A_to_B_loss = (all_loss["loss"] * weights).mean()
```

---

## 6. Glossary

| Term | Definition |
| :--- | :--- |
| **DDPM** | Denoising Diffusion Probabilistic Model. A generative model that learns to reverse a gradual noising process. |
| **Mamba / SSM** | State Space Model architecture that maps sequences dynamically in linear time $O(N)$, allowing for infinite context windows without the memory cost of Transformers. |
| **Sinusoidal Embed** | Maps a scalar timestep `t` to a fixed high-dimensional vector. Gives the model a unique representation for every timestep 0–T. |
| **Time Shift** | A layer that projects the time embedding to match feature channels. The output is added element-wise (⊕) to spatial features so the block knows the noise level. |
| **Gamma Pass Rate** | A dosimetric QA metric used in radiation therapy. Checks if dose difference and spatial distance-to-agreement fall within a tolerance (e.g. 1%/1mm or 2%/2mm). |
| **RED** | Relative Electron Density. Derived from Hounsfield Units, RED is the physical value directly used by clinical dose calculation algorithms. |
