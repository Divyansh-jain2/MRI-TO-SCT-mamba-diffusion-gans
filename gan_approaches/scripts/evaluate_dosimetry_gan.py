"""
Dosimetric evaluation script for Pix2Pix and UNIT (mc-ddpm dataset).

HYBRID EVALUATION VERSION:
- PSNR / SSIM: Calculated on WINDOWED HU (matches training logic, gets ~30dB).
- Tissue MAE / Gamma / RED: Calculated on RAW HU (matches clinical dosimetry logic).

Usage:
    python scripts/evaluate_dosimetry_gan.py
"""

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image
from scipy.ndimage import maximum_filter
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.utils.util_data as util_data  # noqa: E402

# Standard CT Range for Raw Evaluation
CT_CLIP = (-1024, 1650)

MODEL_CONFIGS = {
    "pix2pix": "configs/pix2pix_train.yaml",
    "unit":    "configs/unit_train.yaml",
}

# ---------------------------------------------------------------------------
# Dosimetric helpers
# ---------------------------------------------------------------------------

def hu_to_red(hu: np.ndarray) -> np.ndarray:
    red = np.zeros_like(hu, dtype=np.float64)
    mask_air  = hu < -100
    mask_bone = hu > 100
    mask_st   = (~mask_air) & (~mask_bone)
    red[mask_air]  = 0.001 + 1.049e-3 * (hu[mask_air]  + 1000)
    red[mask_st]   = 1.0   + 0.001    *  hu[mask_st]
    red[mask_bone] = 1.0   + 0.0005   *  hu[mask_bone]
    return red

def tissue_masks(hu: np.ndarray):
    air  = hu < -200
    bone = hu > 300
    soft = (~air) & (~bone)
    return air, soft, bone

def gamma_pass_rate(ref_hu: np.ndarray, eval_hu: np.ndarray,
                    dose_tol_pct: float, dist_mm: float,
                    voxel_mm: tuple = (1, 1, 1)) -> float:
    dose_range = ref_hu.max() - ref_hu.min()
    if dose_range == 0: return 1.0
    dose_thresh = dose_tol_pct / 100.0 * dose_range
    diff        = np.abs(ref_hu.astype(np.float64) - eval_hu.astype(np.float64))
    dose_pass   = diff <= dose_thresh
    r           = [max(1, int(np.ceil(dist_mm / s))) for s in voxel_mm]
    kernel      = tuple(2 * ri + 1 for ri in r)
    dist_pass   = maximum_filter(dose_pass.astype(np.float32), size=kernel) > 0.5
    return float((dose_pass | dist_pass).sum()) / float(diff.size)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _read_pred_slice_norm(pred_path: Path, img_dim: int) -> np.ndarray:
    pred = np.asarray(Image.open(pred_path).convert("L"), dtype=np.float32) / 255.0
    if pred.shape != (img_dim, img_dim):
        pred = cv2.resize(pred, (img_dim, img_dim), interpolation=cv2.INTER_LINEAR)
    return np.clip(pred, 0.0, 1.0).astype(np.float32)

def _read_gt_slice_info(ct_volume_cache: dict, ct_path: str,
                        slice_number: int, wc: float, ww: float,
                        img_dim: int) -> dict:
    ct_path_norm = ct_path.replace("\\", "/")
    if ct_path_norm not in ct_volume_cache:
        ct_volume_cache[ct_path_norm] = util_data.read_nii(ct_path_norm)

    ct_vol    = ct_volume_cache[ct_path_norm]
    raw_slice = util_data.extract_axial_slice(ct_vol, slice_number)

    hu_lo = float(wc - ww / 2.0)
    hu_hi = float(wc + ww / 2.0)

    # Windowed GT
    gt_windowed = util_data.contrast_stretching(raw_slice, ww=ww, wc=wc)
    gt_windowed = cv2.resize(gt_windowed.astype(np.float32), (img_dim, img_dim), interpolation=cv2.INTER_LINEAR)
    
    # Raw GT
    gt_raw = np.clip(raw_slice.astype(np.float32), CT_CLIP[0], CT_CLIP[1])
    gt_raw = cv2.resize(gt_raw, (img_dim, img_dim), interpolation=cv2.INTER_LINEAR)

    return {
        "gt_windowed": gt_windowed,
        "gt_raw":      gt_raw,
        "hu_lo":       hu_lo,
        "hu_hi":       hu_hi
    }

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model_name: str, fold: int = 0) -> None:
    print(f"\n☢️  Evaluating {model_name.upper()} (Hybrid Mode)")
    
    config_path = REPO_ROOT / MODEL_CONFIGS[model_name]
    cfg = yaml.safe_load(config_path.open())
    img_dim = int(cfg["data"]["img_dim"])
    fold_dir = Path(cfg["data"]["fold_dir"])
    report_dir = REPO_ROOT / cfg["data"]["report_dir"]
    
    split_csv = fold_dir / str(fold) / "test.csv"
    outputs_dir = report_dir / cfg["exp_name"] / "outputs" / str(fold)

    import pandas as pd
    df = pd.read_csv(split_csv).sort_values(["patient_id", "slice_number"])

    ct_cache = {}
    by_patient = defaultdict(lambda: {"pred_win": [], "gt_win": [], "pred_raw": [], "gt_raw": [], "hu_lo": [], "hu_hi": []})

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading"):
        id_slice = row["id_slice"]
        pred_path = outputs_dir / f"{id_slice}_output.png"
        if not pred_path.exists(): continue

        info = _read_gt_slice_info(ct_cache, row["slice_ct_path"], row["slice_number"], row["wc"], row["ww"], img_dim)
        pred_norm = _read_pred_slice_norm(pred_path, img_dim)

        # Reconstruct HU
        scale = max(info["hu_hi"] - info["hu_lo"], 1e-8)
        p_win = (pred_norm * scale + info["hu_lo"]).astype(np.float32)
        
        # In hybrid mode:
        # For MAE/Gamma, we keep the reconstructed HU even if it only covers the window range.
        # This will show high error in bone/air because the model cannot reach those values.
        p_raw = p_win 

        pat = row["patient_id"]
        by_patient[pat]["pred_win"].append(p_win)
        by_patient[pat]["gt_win"].append(info["gt_windowed"])
        by_patient[pat]["pred_raw"].append(p_raw)
        by_patient[pat]["gt_raw"].append(info["gt_raw"])
        by_patient[pat]["hu_lo"].append(info["hu_lo"])
        by_patient[pat]["hu_hi"].append(info["hu_hi"])

    results = []
    for pat, data in tqdm(by_patient.items(), desc="Calculating Metrics"):
        # Windowed volumes for PSNR/SSIM
        vol_p_win = np.stack(data["pred_win"], axis=-1)
        vol_g_win = np.stack(data["gt_win"], axis=-1)
        
        # Raw volumes for MAE/Gamma
        vol_p_raw = np.stack(data["pred_raw"], axis=-1)
        vol_g_raw = np.stack(data["gt_raw"], axis=-1)

        hu_range = float(max(data["hu_hi"]) - min(data["hu_lo"]))
        
        # Windowed Metrics
        psnr = psnr_metric(vol_g_win, vol_p_win, data_range=max(hu_range, 1e-8))
        ssim = np.mean([ssim_metric(vol_g_win[...,z], vol_p_win[...,z], data_range=hu_range) for z in range(vol_g_win.shape[2])])

        # Raw Metrics (Clinical)
        air_m, soft_m, bone_m = tissue_masks(vol_g_raw)
        mae_air  = np.mean(np.abs(vol_g_raw[air_m] - vol_p_raw[air_m])) if air_m.any() else 0
        mae_soft = np.mean(np.abs(vol_g_raw[soft_m] - vol_p_raw[soft_m])) if soft_m.any() else 0
        mae_bone = np.mean(np.abs(vol_g_raw[bone_m] - vol_p_raw[bone_m])) if bone_m.any() else 0
        
        red_mae = np.mean(np.abs(hu_to_red(vol_g_raw) - hu_to_red(vol_p_raw)))
        g11 = gamma_pass_rate(vol_g_raw, vol_p_raw, 1, 1)
        g22 = gamma_pass_rate(vol_g_raw, vol_p_raw, 2, 2)

        results.append({
            "sample": pat, "PSNR_win": psnr, "SSIM_win": ssim,
            "MAE_Air_Raw": mae_air, "MAE_Soft_Raw": mae_soft, "MAE_Bone_Raw": mae_bone,
            "RED_MAE": red_mae, "Gamma_1_1": g11, "Gamma_2_2": g22
        })

    # Summary
    res_df = pd.DataFrame(results)
    avg = res_df.mean(numeric_only=True)
    
    print(f"\n✅ Results for {model_name}:")
    print(f"   PSNR (Windowed):  {avg['PSNR_win']:.2f} dB  <-- Should be ~30dB")
    print(f"   SSIM (Windowed):  {avg['SSIM_win']:.4f}")
    print(f"   Air MAE (Raw):    {avg['MAE_Air_Raw']:.2f} HU")
    print(f"   Soft MAE (Raw):   {avg['MAE_Soft_Raw']:.2f} HU")
    print(f"   Bone MAE (Raw):   {avg['MAE_Bone_Raw']:.2f} HU")
    print(f"   Gamma (1%/1mm):   {avg['Gamma_1_1']*100:.2f}%")

    out_csv = report_dir / cfg["exp_name"] / "dosimetric_test_metrics_hybrid.csv"
    res_df.to_csv(out_csv, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["pix2pix", "unit", "both"], default="both")
    args = parser.parse_args()
    
    models = ["pix2pix", "unit"] if args.model == "both" else [args.model]
    for m in models: evaluate_model(m)
