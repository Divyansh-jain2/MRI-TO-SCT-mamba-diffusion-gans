# Dosimetric Analysis Report: Triaxial vs. Triplane Mamba

This report summarizes the comprehensive dosimetric performance of the **Triaxial Mamba** and **Triplane Mamba** architectures for MRI-to-CT synthesis. The analysis evaluates tissue-specific Mean Absolute Error (MAE) in Hounsfield Units (HU), Relative Electron Density (RED) accuracy, and 3D Gamma-Index pass rates.

## 📊 Evaluation Metrics

The peak signal-to-noise ratio (PSNR) has been calculated in 1D (ray-wise), 2D (slice-by-slice), and 3D formats as requested.

| Metric | Triaxial Mamba | Triplane Mamba | Best Performer |
|--------|----------------|----------------|----------------|
| **PSNR (3D)** | 25.71 dB | 25.79 dB | Triplane |
| **PSNR (2D)** | 26.32 dB | 26.40 dB | Triplane |
| **PSNR (1D)** | 33.32 dB | 33.77 dB | Triplane |
| **SSIM** | 0.8483 | 0.8502 | Triplane |
| **Air MAE** | 60.77 HU | 57.36 HU | Triplane |
| **Soft Tissue MAE** | 38.31 HU | 38.87 HU | Triaxial |
| **Bone MAE** | 196.20 HU | 189.39 HU | Triplane |
| **RED MAE** | 0.05012 | 0.04837 | Triplane |
| **Gamma (1% / 1mm)** | 88.71% | 90.61% | Triplane |
| **Gamma (2% / 2mm)** | 98.83% | 99.14% | Triplane |

## 💡 Key Findings

1. **Overall Superiority**: **Triplane Mamba** slightly outperforms Triaxial Mamba across almost all dosimetric and image quality metrics. This demonstrates that its architecture captures spatial representations with slightly higher fidelity for this dataset.
2. **Clinical Accuracy**: Triplane Mamba achieved notably higher Gamma passing rates, breaking the 90% threshold for the strict 1%/1mm criteria (90.61%), and reaching 99.14% for the 2%/2mm criteria. 
3. **Tissue-Specific Performance**: 
   - Air and Bone tissue synthesis are significantly more accurate in Triplane Mamba. 
   - Triaxial Mamba demonstrated marginally better performance solely in Soft Tissue MAE.
   - The improved bone synthesis in Triplane directly contributes to its superior Relative Electron Density (RED) accuracy (0.04837).
4. **1D vs 2D vs 3D Validation**: 1D PSNR values are significantly higher (~33 dB), as they average error along a single ray. 2D PSNR values are consistently higher than 3D PSNR values for both architectures (averaging ~0.6 dB higher), accurately reflecting the inter-slice variations often present in 3D medical image synthesis.
