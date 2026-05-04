# Dosimetric Analysis: UMamba vs SegMamba

## Overview
This report provides a comprehensive dosimetric and image quality evaluation of two Mamba-based approaches for MRI-to-CT synthesis: **SegMamba** and **UMamba** (Diffusion UMamba). The evaluation assesses spatial fidelity, tissue-specific accuracy (focusing on bone and air, which are critical for clinical dosimetry), and overall structural similarity.

## Metric Definitions

- **PSNR (Peak Signal-to-Noise Ratio) & SSIM (Structural Similarity Index):** Standard computer vision metrics measuring the overall image quality, sharpness, and structural similarity between the synthetic CT and the real CT.
- **Tissue-Specific MAE (Mean Absolute Error):** Measures the average error in Hounsfield Units (HU) specifically within Air, Soft Tissue, and Bone regions. Accurate bone and air generation is critical because these tissues heavily influence how radiation is absorbed.
- **RED MAE (Relative Electron Density):** Evaluates the direct conversion of Hounsfield Units into electron density values, which are the exact physical properties used by clinical software to calculate radiation dose. Lower RED MAE means the synthetic CT interacts with radiation almost exactly like the real CT would.
- **Gamma Pass Rate:** The clinical gold standard for radiotherapy. It simultaneously checks both the dose difference and spatial distance difference. For example, a "1% / 1mm" pass rate checks what percentage of the volume falls within a 1% dose tolerance and a 1mm spatial tolerance. High scores (e.g., >90%) indicate clinical acceptability.

## Comparative Metrics

| Metric | SegMamba | UMamba | Improvement |
| :--- | :--- | :--- | :--- |
| **PSNR (3D)** | 24.79 dB | **25.23 dB** | +0.44 dB |
| **PSNR (2D)** | 25.42 dB | **25.78 dB** | +0.36 dB |
| **PSNR (1D)** | 32.84 dB | **33.88 dB** | +1.04 dB |
| **SSIM** | 0.8374 | **0.8509** | +0.0135 |
| **Air MAE** | 65.74 HU | **60.53 HU** | -5.21 HU |
| **Soft Tissue MAE** | 38.15 HU | **35.43 HU** | -2.72 HU |
| **Bone MAE** | 208.52 HU | **192.50 HU** | -16.02 HU |
| **RED MAE** | 0.05208 | **0.04794** | -0.00414 |
| **Gamma Pass Rate (1% / 1mm)** | 91.61% | **93.26%** | +1.65% |
| **Gamma Pass Rate (2% / 2mm)** | 99.35% | **99.55%** | +0.20% |

## Conclusion
**UMamba** demonstrates superior performance across all measured metrics compared to **SegMamba**. Notably, UMamba achieves significant reductions in tissue-specific Mean Absolute Error (MAE), particularly in challenging high-density (Bone) and low-density (Air) regions. This translates to a more accurate Relative Electron Density (RED) mapping and consistently higher Gamma-Index Pass Rates, making UMamba a much stronger candidate for clinical dosimetric applications.
