import os
import glob
import csv
import numpy as np
import nibabel as nib
from natsort import natsorted
from scipy.ndimage import maximum_filter
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_RESULTS_DIR = os.path.join(BASE_DIR, 'results_test_brain_umamba_diffusion')
CT_CLIP = (-1024, 1650)

def hu_to_red(hu):
    red = np.zeros_like(hu, dtype=np.float64)
    mask_air = hu < -100
    mask_bone = hu > 100
    mask_st = (~mask_air) & (~mask_bone)
    red[mask_air] = 0.001 + 1.049e-3 * (hu[mask_air] + 1000)
    red[mask_st] = 1.0 + 0.001 * hu[mask_st]
    red[mask_bone] = 1.0 + 0.0005 * hu[mask_bone]
    return red

def tissue_masks(hu):
    air = hu < -200
    bone = hu > 300
    soft = (~air) & (~bone)
    return air, soft, bone

def gamma_pass_rate(ref_hu, eval_hu, dose_tol_pct, dist_mm, voxel_mm=(1, 1, 1)):
    dose_range = ref_hu.max() - ref_hu.min()
    if dose_range == 0: return 1.0
    dose_thresh = dose_tol_pct / 100.0 * dose_range
    diff = np.abs(ref_hu.astype(np.float64) - eval_hu.astype(np.float64))
    dose_pass = diff <= dose_thresh
    r = [max(1, int(np.ceil(dist_mm / s))) for s in voxel_mm]
    kernel = tuple(2 * ri + 1 for ri in r)
    dist_pass = maximum_filter(dose_pass.astype(np.float32), size=kernel) > 0.5
    return float((dose_pass | dist_pass).sum()) / float(diff.size)

def evaluate_final_model():
    print(f"\n{'-'*70}")
    print(f" ☢️ Evaluating Dosimetry for: OPTIMIZED UMAMBA DIFFUSION (TEST SET)")
    print(f"{'-'*70}")

    if not os.path.isdir(TEST_RESULTS_DIR):
        print(f" ⚠ Prediction directory not found: {TEST_RESULTS_DIR}.")
        return None

    pred_files = natsorted(glob.glob(os.path.join(TEST_RESULTS_DIR, 'pred_*.nii.gz')))
    all_metrics = []
    
    lo, hi = CT_CLIP
    dr = float(hi - lo)
    
    for pred_file in tqdm(pred_files, desc="Processing Samples", colour='green'):
        filename = os.path.basename(pred_file)
        gt_file = os.path.join(TEST_RESULTS_DIR, filename.replace('pred_', 'gt_'))
        
        if not os.path.exists(gt_file):
            print(f"GT missing for {filename}")
            continue
            
        sct_nifti = nib.load(pred_file)
        sct_hu = sct_nifti.get_fdata()
        
        gt_nifti = nib.load(gt_file)
        gt_hu = gt_nifti.get_fdata()
        
        if gt_hu.shape != sct_hu.shape: continue
            
        psnr_val = float(psnr_metric(gt_hu, sct_hu, data_range=dr))
        ssim_val = float(np.mean([ssim_metric(gt_hu[:,:,z], sct_hu[:,:,z], data_range=dr) for z in range(gt_hu.shape[2])]))
        
        air_m, soft_m, bone_m = tissue_masks(gt_hu)
        mae_air = float(np.mean(np.abs(gt_hu[air_m] - sct_hu[air_m]))) if air_m.any() else 0.
        mae_soft = float(np.mean(np.abs(gt_hu[soft_m] - sct_hu[soft_m]))) if soft_m.any() else 0.
        mae_bone = float(np.mean(np.abs(gt_hu[bone_m] - sct_hu[bone_m]))) if bone_m.any() else 0.
        
        red_gt, red_sct = hu_to_red(gt_hu), hu_to_red(sct_hu)
        red_mae = float(np.mean(np.abs(red_gt - red_sct)))
        
        gpr_1_1 = gamma_pass_rate(gt_hu, sct_hu, 1, 1)
        gpr_2_2 = gamma_pass_rate(gt_hu, sct_hu, 2, 2)
        
        metrics = {
            'sample': filename.replace('pred_', '').replace('.nii.gz', ''),
            'PSNR_dB': psnr_val, 
            'SSIM': ssim_val, 
            'MAE_Air_HU': mae_air, 
            'MAE_Soft_HU': mae_soft, 
            'MAE_Bone_HU': mae_bone, 
            'RED_MAE': red_mae, 
            'Gamma_1pct_1mm': gpr_1_1, 
            'Gamma_2pct_2mm': gpr_2_2
        }
        all_metrics.append(metrics)

    if not all_metrics: return
    
    # Calculate averages
    avg_metrics = {'sample': 'AVERAGE'}
    for k in all_metrics[0].keys():
        if k != 'sample':
            avg_metrics[k] = np.mean([m[k] for m in all_metrics])
    
    # Add to list
    all_metrics.append(avg_metrics)
    
    csv_path = os.path.join(TEST_RESULTS_DIR, 'dosimetric_test_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(all_metrics[0].keys()))
        w.writeheader()
        for m in all_metrics:
            w.writerow(m)
        
    print(f"\n ✅ Done! Saved comprehensive dosimetric results to: {csv_path}")
    print("\n   [Average Dosimetric Output of Optimized Model over 37 Test Subjects]")
    print(f"   PSNR:          {avg_metrics['PSNR_dB']:.2f} dB")
    print(f"   SSIM:          {avg_metrics['SSIM']:.4f}")
    print(f"   Air MAE:       {avg_metrics['MAE_Air_HU']:.2f} HU")
    print(f"   Soft MAE:      {avg_metrics['MAE_Soft_HU']:.2f} HU")
    print(f"   Bone MAE:      {avg_metrics['MAE_Bone_HU']:.2f} HU")
    print(f"   RED MAE:       {avg_metrics['RED_MAE']:.4f}")
    print(f"   Gamma (1%/1mm): {avg_metrics['Gamma_1pct_1mm'] * 100:.2f}%")
    print(f"   Gamma (2%/2mm): {avg_metrics['Gamma_2pct_2mm'] * 100:.2f}%")

if __name__ == '__main__':
    evaluate_final_model()
