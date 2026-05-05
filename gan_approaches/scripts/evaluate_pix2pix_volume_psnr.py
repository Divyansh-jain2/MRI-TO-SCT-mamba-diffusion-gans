import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as sk_psnr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.utils.util_data as util_data  # noqa: E402


def _resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _load_cfg(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_pred_slice(pred_path: Path, img_dim: int) -> np.ndarray:
    pred = np.asarray(Image.open(pred_path).convert("L"), dtype=np.float32) / 255.0
    if pred.shape != (img_dim, img_dim):
        pred = cv2.resize(pred, (img_dim, img_dim), interpolation=cv2.INTER_LINEAR)
    return np.clip(pred, 0.0, 1.0).astype(np.float32)


def _read_gt_slice_info(
    ct_volume_cache: dict,
    ct_path: str,
    slice_number: int,
    wc: float,
    ww: float,
    img_dim: int,
) -> np.ndarray:
    ct_path = ct_path.replace("\\", "/")
    if ct_path not in ct_volume_cache:
        ct_volume_cache[ct_path] = util_data.read_nii(ct_path)

    ct_slice_raw = util_data.extract_axial_slice(ct_volume_cache[ct_path], slice_number)

    hu_lo = float(wc - ww / 2.0)
    hu_hi = float(wc + ww / 2.0)
    ct_slice_hu = util_data.contrast_stretching(ct_slice_raw, ww=ww, wc=wc)

    finite = np.isfinite(ct_slice_hu)
    if np.any(finite):
        hu_min = float(np.min(ct_slice_hu[finite]))
        hu_max = float(np.max(ct_slice_hu[finite]))
    else:
        hu_min = hu_lo
        hu_max = hu_hi

    gt_slice_norm = util_data.normalize_01(ct_slice_hu, ct_path)

    gt_slice_hu = cv2.resize(ct_slice_hu.astype(np.float32), (img_dim, img_dim), interpolation=cv2.INTER_LINEAR)
    gt_slice_norm = cv2.resize(gt_slice_norm.astype(np.float32), (img_dim, img_dim), interpolation=cv2.INTER_LINEAR)

    return {
        "gt_slice_hu": np.nan_to_num(gt_slice_hu, nan=hu_lo, posinf=hu_hi, neginf=hu_lo).astype(np.float32),
        "gt_slice_norm": np.clip(gt_slice_norm, 0.0, 1.0).astype(np.float32),
        "hu_lo": hu_lo,
        "hu_hi": hu_hi,
        "hu_min": hu_min,
        "hu_max": hu_max,
    }


def _psnr_from_mse(mse: float, data_range: float = 1.0) -> float:
    if mse <= 0.0:
        return math.inf
    return float(10.0 * math.log10((data_range ** 2) / mse))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Pix2Pix volume-level PSNR from saved per-slice outputs (no retraining)."
    )
    parser.add_argument("--config", default="configs/pix2pix_train.yaml", help="Path to Pix2Pix training YAML.")
    parser.add_argument("--fold", type=int, default=0, help="Fold index to evaluate (default: 0).")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Split CSV to evaluate.")
    parser.add_argument(
        "--outputs-dir",
        default=None,
        help="Optional override for folder with generated slices named <id_slice>_output.png.",
    )
    parser.add_argument(
        "--out-csv",
        default=None,
        help="Optional output CSV path. Default: results/<exp_name>/volume_psnr_fold<k>_<split>.csv",
    )
    parser.add_argument(
        "--hu-mapping",
        choices=["window", "slice-minmax"],
        default="window",
        help=(
            "How to map normalized prediction back to HU. "
            "window: pred_hu = pred_norm * (hu_hi - hu_lo) + hu_lo; "
            "slice-minmax: pred_hu = pred_norm * (hu_max - hu_min) + hu_min."
        ),
    )
    args = parser.parse_args()

    config_path = _resolve_repo_path(args.config)
    cfg = _load_cfg(config_path)

    img_dim = int(cfg["data"]["img_dim"])
    fold_dir = _resolve_repo_path(cfg["data"]["fold_dir"])
    split_csv = fold_dir / str(args.fold) / f"{args.split}.csv"
    if not split_csv.exists():
        raise FileNotFoundError(f"Split CSV not found: {split_csv}")

    if args.outputs_dir:
        outputs_dir = _resolve_repo_path(args.outputs_dir)
    else:
        outputs_dir = _resolve_repo_path(cfg["data"]["report_dir"]) / cfg["exp_name"] / "outputs" / str(args.fold)
    if not outputs_dir.exists():
        raise FileNotFoundError(f"Outputs directory not found: {outputs_dir}")

    if args.out_csv:
        out_csv = _resolve_repo_path(args.out_csv)
    else:
        out_csv = _resolve_repo_path(cfg["data"]["report_dir"]) / cfg["exp_name"] / f"volume_psnr_fold{args.fold}_{args.split}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(split_csv, index_col="id_slice")
    required_cols = {"patient_id", "slice_number", "slice_ct_path", "wc", "ww"}
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        raise ValueError(f"Missing columns in {split_csv}: {missing_cols}")

    df = df.sort_values(["patient_id", "slice_number"], kind="stable")

    ct_volume_cache = {}
    by_patient_pred_hu = defaultdict(list)
    by_patient_gt_hu = defaultdict(list)
    by_patient_pred_norm = defaultdict(list)
    by_patient_gt_norm = defaultdict(list)
    by_patient_slice_numbers = defaultdict(list)
    by_patient_hu_los = defaultdict(list)
    by_patient_hu_his = defaultdict(list)

    missing_preds = []

    for id_slice, row in df.iterrows():
        pred_path = outputs_dir / f"{id_slice}_output.png"
        if not pred_path.exists():
            missing_preds.append(str(pred_path))
            continue

        patient_id = str(row["patient_id"])
        slice_number = int(row["slice_number"])
        wc = float(row["wc"])
        ww = float(row["ww"])
        ct_path = str(row["slice_ct_path"])

        pred_slice_norm = _read_pred_slice(pred_path, img_dim)
        gt_info = _read_gt_slice_info(
            ct_volume_cache=ct_volume_cache,
            ct_path=ct_path,
            slice_number=slice_number,
            wc=wc,
            ww=ww,
            img_dim=img_dim,
        )

        if args.hu_mapping == "window":
            scale = max(gt_info["hu_hi"] - gt_info["hu_lo"], 1e-8)
            pred_slice_hu = pred_slice_norm * scale + gt_info["hu_lo"]
        else:
            scale = max(gt_info["hu_max"] - gt_info["hu_min"], 1e-8)
            pred_slice_hu = pred_slice_norm * scale + gt_info["hu_min"]

        pred_slice_hu = np.clip(pred_slice_hu.astype(np.float32), gt_info["hu_lo"], gt_info["hu_hi"])

        by_patient_pred_hu[patient_id].append(pred_slice_hu)
        by_patient_gt_hu[patient_id].append(gt_info["gt_slice_hu"])
        by_patient_pred_norm[patient_id].append(pred_slice_norm)
        by_patient_gt_norm[patient_id].append(gt_info["gt_slice_norm"])
        by_patient_slice_numbers[patient_id].append(slice_number)
        by_patient_hu_los[patient_id].append(gt_info["hu_lo"])
        by_patient_hu_his[patient_id].append(gt_info["hu_hi"])

    if missing_preds:
        preview = "\n".join(missing_preds[:10])
        raise FileNotFoundError(
            "Missing generated output slices. First missing files:\n"
            f"{preview}\n"
            f"Total missing: {len(missing_preds)}"
        )

    rows = []
    for patient_id in sorted(by_patient_gt_hu.keys()):
        gt_hu_list = by_patient_gt_hu[patient_id]
        pred_hu_list = by_patient_pred_hu[patient_id]
        gt_norm_list = by_patient_gt_norm[patient_id]
        pred_norm_list = by_patient_pred_norm[patient_id]
        if not gt_hu_list:
            continue

        gt_hu_vol = np.stack(gt_hu_list, axis=-1)
        pred_hu_vol = np.stack(pred_hu_list, axis=-1)
        gt_norm_vol = np.stack(gt_norm_list, axis=-1)
        pred_norm_vol = np.stack(pred_norm_list, axis=-1)

        hu_data_range = float(max(by_patient_hu_his[patient_id]) - min(by_patient_hu_los[patient_id]))
        hu_data_range = max(hu_data_range, 1e-8)

        mse_hu = float(np.mean((pred_hu_vol - gt_hu_vol) ** 2, dtype=np.float64))
        psnr_hu_db = float(sk_psnr(gt_hu_vol, pred_hu_vol, data_range=hu_data_range))

        mse_norm = float(np.mean((pred_norm_vol - gt_norm_vol) ** 2, dtype=np.float64))
        psnr_norm_db = _psnr_from_mse(mse_norm, data_range=1.0)

        rows.append(
            {
                "patient_id": patient_id,
                "n_slices": int(len(gt_hu_list)),
                "min_slice_number": int(np.min(by_patient_slice_numbers[patient_id])),
                "max_slice_number": int(np.max(by_patient_slice_numbers[patient_id])),
                "hu_data_range": hu_data_range,
                "mse_hu": mse_hu,
                "psnr_hu_db": psnr_hu_db,
                "mse_norm": mse_norm,
                "psnr_norm_db": psnr_norm_db,
            }
        )

    if not rows:
        raise RuntimeError("No volumes were evaluated. Check split CSV and outputs folder.")

    out_df = pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)
    out_df.to_csv(out_csv, index=False)

    # Compute HU-space statistics
    psnr_hu_vals = out_df["psnr_hu_db"].to_numpy(dtype=np.float64)
    finite_mask_hu = np.isfinite(psnr_hu_vals)
    finite_vals_hu = psnr_hu_vals[finite_mask_hu]

    if finite_vals_hu.size == 0:
        mean_psnr_hu = float("nan")
        std_psnr_hu = float("nan")
    else:
        mean_psnr_hu = float(np.mean(finite_vals_hu))
        std_psnr_hu = float(np.std(finite_vals_hu, ddof=1)) if finite_vals_hu.size > 1 else 0.0

    # Compute normalized-space statistics
    psnr_norm_vals = out_df["psnr_norm_db"].to_numpy(dtype=np.float64)
    finite_mask_norm = np.isfinite(psnr_norm_vals)
    finite_vals_norm = psnr_norm_vals[finite_mask_norm]

    if finite_vals_norm.size == 0:
        mean_psnr_norm = float("nan")
        std_psnr_norm = float("nan")
    else:
        mean_psnr_norm = float(np.mean(finite_vals_norm))
        std_psnr_norm = float(np.std(finite_vals_norm, ddof=1)) if finite_vals_norm.size > 1 else 0.0

    print("=" * 72)
    print(f"Pix2Pix volume-level PSNR | fold={args.fold} split={args.split} | mapping={args.hu_mapping}")
    print(f"Volumes evaluated: {len(out_df)}")
    print(f"Finite HU-PSNR volumes: {int(finite_mask_hu.sum())}")
    print(f"Finite Norm-PSNR volumes: {int(finite_mask_norm.sum())}")
    print(f"Volume HU-PSNR mean +/- std: {mean_psnr_hu:.4f} +/- {std_psnr_hu:.4f} dB")
    print(f"Volume Norm-PSNR mean +/- std: {mean_psnr_norm:.4f} +/- {std_psnr_norm:.4f} dB")
    print("CSV columns include both HU and normalized metrics per volume.")
    print(f"Per-volume CSV saved to: {out_csv}")
    print("=" * 72)


if __name__ == "__main__":
    main()
