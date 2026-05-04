#!/usr/bin/env python3
"""
Standalone Dosimetric Evaluation Script
=======================================
This script exclusively computes crucial dosimetric metrics (Tissue-Specific MAE,
Relative Electron Density (RED) Accuracy, and Gamma-Index Pass Rates) on pre-generated
Synthetic CT (sCT) .npy files without re-running the diffusion inference loop.

Usage:
    python3 evaluate_dosimetric.py
"""

import os
import glob
import csv
import numpy as np
from natsort import natsorted
from scipy.ndimage import maximum_filter
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from tqdm import tqdm

# ── Constants ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = '/DATA/Munish_Synthetic_CT/mc_ddpm_data/brain_npy'
CT_CLIP = (-1024, 1650)

MODELS = [
    {
        'id': 'triplane_mamba',
        'name': 'Triplane Mamba',
        'pred_dir': os.path.join(BASE_DIR, 'predictions_triplane')
    }
]

# ═════════════════════════════════════════════════════════════════════════════
#  DOSIMETRIC FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def hu_to_red(hu):
    """Converts Hounsfield Units (HU) to Relative Electron Density (RED) using ICRU-44."""
    red = np.zeros_like(hu, dtype=np.float64)
    mask_air = hu < -100
    mask_bone = hu > 100
    mask_st = (~mask_air) & (~mask_bone)
    
    red[mask_air] = 0.001 + 1.049e-3 * (hu[mask_air] + 1000)
    red[mask_st] = 1.0 + 0.001 * hu[mask_st]
    red[mask_bone] = 1.0 + 0.0005 * hu[mask_bone]
    return red

def tissue_masks(hu):
    """Creates boolean masks for tissue density clustering based on HU ranges."""
    air = hu < -200
    bone = hu > 300
    soft = (~air) & (~bone)
    return air, soft, bone

def gamma_pass_rate(ref_hu, eval_hu, dose_tol_pct, dist_mm, voxel_mm=(1, 1, 1)):
    """Computes simplified 3D Gamma-index pass rate (using HU as a dose surrogate)."""
    dose_range = ref_hu.max() - ref_hu.min()
    if dose_range == 0: 
        return 1.0
        
    dose_thresh = dose_tol_pct / 100.0 * dose_range
    diff = np.abs(ref_hu.astype(np.float64) - eval_hu.astype(np.float64))
    dose_pass = diff <= dose_thresh
    
    r = [max(1, int(np.ceil(dist_mm / s))) for s in voxel_mm]
    kernel = tuple(2 * ri + 1 for ri in r)
    dist_pass = maximum_filter(dose_pass.astype(np.float32), size=kernel) > 0.5
    
    return float((dose_pass | dist_pass).sum()) / float(diff.size)

# ═════════════════════════════════════════════════════════════════════════════
#  EVALUATION PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_model_dosimetry(model_info):
    print(f"\n{'-'*70}")
    print(f" ☢️ Evaluating Dosimetry for: {model_info['name']}")
    print(f"{'-'*70}")

    pred_dir = model_info['pred_dir']
    if not os.path.isdir(pred_dir):
        print(f" ⚠ Prediction directory not found: {pred_dir}")
        return None

    # Load Ground Truths
    test_files = natsorted(glob.glob(os.path.join(DATA_ROOT, 'imagesTs', '*.npy')))
    if not test_files:
        print(f" ⚠ No ground truth .npy files found in {os.path.join(DATA_ROOT, 'imagesTs')}")
        return None

    all_metrics = []
    lo, hi = CT_CLIP
    voxel_spacing = (1.0, 1.0, 1.0) # Assume 1mm isotropic for brain data

    for i, gt_file in enumerate(tqdm(test_files, desc="Processing Samples", colour='green')):
        basename = os.path.basename(gt_file).replace('.npy', '')
        
        # Load Ground Truth CT
        data = np.load(gt_file)
        gt_ct_norm = data[1] # [1] is the CT, [0] is the MRI
        gt_hu = (gt_ct_norm + 1.0) / 2.0 * (hi - lo) + lo
        
        # Load Predicted NPY
        pred_file = os.path.join(pred_dir, f'{basename}_pred.npy')
        if not os.path.exists(pred_file):
            print(f"\n   -> Missing prediction: {pred_file}")
            continue
            
        sct_norm = np.load(pred_file)
        # Prediction shape is (96, 192, 192), transpose to match GT (192, 192, 96)
        sct_norm = np.transpose(sct_norm, (1, 2, 0))
        sct_hu = (sct_norm + 1.0) / 2.0 * (hi - lo) + lo
        
        # Ensure dimensions match
        if gt_hu.shape != sct_hu.shape:
            print(f"\n   -> Shape mismatch on Sample {i}: GT {gt_hu.shape} vs SCT {sct_hu.shape}")
            continue
            
        # ── Compute PSNR & SSIM
        dr = float(CT_CLIP[1] - CT_CLIP[0])
        psnr_3d = float(psnr_metric(gt_hu, sct_hu, data_range=dr))
        
        psnr_2d_vals = [psnr_metric(gt_hu[:,:,z], sct_hu[:,:,z], data_range=dr)
                        for z in range(gt_hu.shape[2])]
        psnr_2d = float(np.mean(psnr_2d_vals))
        
        mse_1d = np.mean((gt_hu - sct_hu)**2, axis=2)
        mse_1d = np.maximum(mse_1d, 1e-10)
        psnr_1d_vals = 10 * np.log10((dr**2) / mse_1d)
        psnr_1d = float(np.mean(psnr_1d_vals))
        
        ssim_vals = [ssim_metric(gt_hu[:,:,z], sct_hu[:,:,z], data_range=dr)
                     for z in range(gt_hu.shape[2])]
        ssim_val = float(np.mean(ssim_vals))
        
        # ── Compute Tissue-Specific MAE
        air_m, soft_m, bone_m = tissue_masks(gt_hu)
        mae_air = float(np.mean(np.abs(gt_hu[air_m] - sct_hu[air_m]))) if air_m.any() else 0.
        mae_soft = float(np.mean(np.abs(gt_hu[soft_m] - sct_hu[soft_m]))) if soft_m.any() else 0.
        mae_bone = float(np.mean(np.abs(gt_hu[bone_m] - sct_hu[bone_m]))) if bone_m.any() else 0.
        
        # ── Compute RED Accuracy
        red_gt = hu_to_red(gt_hu)
        red_sct = hu_to_red(sct_hu)
        red_mae = float(np.mean(np.abs(red_gt - red_sct)))
        red_me = float(np.mean(red_sct - red_gt))
        
        # ── Compute Gamma-Index Pass Rates
        gpr_1_1 = gamma_pass_rate(gt_hu, sct_hu, dose_tol_pct=1, dist_mm=1, voxel_mm=voxel_spacing)
        gpr_2_2 = gamma_pass_rate(gt_hu, sct_hu, dose_tol_pct=2, dist_mm=2, voxel_mm=voxel_spacing)
        
        metrics = {
            'sample': i,
            'PSNR_3D_dB': psnr_3d,
            'PSNR_2D_dB': psnr_2d,
            'PSNR_1D_dB': psnr_1d,
            'SSIM': ssim_val,
            'MAE_Air_HU': mae_air,
            'MAE_Soft_HU': mae_soft,
            'MAE_Bone_HU': mae_bone,
            'RED_MAE': red_mae,
            'RED_ME': red_me,
            'Gamma_1pct_1mm': gpr_1_1,
            'Gamma_2pct_2mm': gpr_2_2
        }
        all_metrics.append(metrics)

    if not all_metrics:
        return None

    # Calculate Summaries
    summary = {}
    metric_keys = [k for k in all_metrics[0].keys() if k != 'sample']
    for k in metric_keys:
        vals = [m[k] for m in all_metrics]
        summary[k] = float(np.mean(vals))
        summary[k + '_std'] = float(np.std(vals))
    
    summary['name'] = model_info['name']
    
    # Save purely dosimetric report to CSV
    csv_path = os.path.join(pred_dir, 'dosimetric_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['sample'] + metric_keys)
        writer.writeheader()
        for m in all_metrics:
            writer.writerow({k: (f'{m[k]:.6f}' if isinstance(m[k], float) else m[k]) for k in ['sample'] + metric_keys})
        
        writer.writerow({k: (f'{summary[k]:.6f}' if k in summary else 'mean') for k in ['sample'] + metric_keys})
        
    print(f"\n ✅ Done! Saved to: {csv_path}")
    print("\n   [Average Metrics]")
    print(f"   PSNR (3D):     {summary['PSNR_3D_dB']:.2f} dB")
    print(f"   PSNR (2D):     {summary['PSNR_2D_dB']:.2f} dB")
    print(f"   PSNR (1D):     {summary['PSNR_1D_dB']:.2f} dB")
    print(f"   SSIM:          {summary['SSIM']:.4f}")
    print(f"   Air MAE:       {summary['MAE_Air_HU']:.2f} HU")
    print(f"   Soft MAE:      {summary['MAE_Soft_HU']:.2f} HU")
    print(f"   Bone MAE:      {summary['MAE_Bone_HU']:.2f} HU")
    print(f"   RED MAE:       {summary['RED_MAE']:.5f}")
    print(f"   Gamma (1%/1mm): {summary['Gamma_1pct_1mm'] * 100:.2f}%")
    print(f"   Gamma (2%/2mm): {summary['Gamma_2pct_2mm'] * 100:.2f}%")
    
    return summary

def main():
    print("\n" + "="*70)
    print(" STANDALONE DOSIMETRIC ANALYSIS ".center(70, "="))
    print("="*70)
    
    for model_info in MODELS:
        evaluate_model_dosimetry(model_info)

if __name__ == '__main__':
    main()
