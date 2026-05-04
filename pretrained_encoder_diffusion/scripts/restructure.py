"""
prepare_synthrad_data.py
========================
Converts SynthRAD2023 preprocessed NIfTI data into the folder structure
expected by the MC-DDPM training notebook.

SynthRAD2023 input structure:
    Task1/
      brain/
        1BA001/
          mr.nii.gz
          ct.nii.gz
          mask.nii.gz
        1BA002/ ...
      pelvis/
        1PA001/
          mr.nii.gz
          ct.nii.gz
          mask.nii.gz

MC-DDPM output structure (NIfTI mode):
    mc_ddpm_data/
      brain/
        imagesTr/   ← MRI volumes, training
        labelsTr/   ← CT volumes, training
        imagesVal/
        labelsVal/
        imagesTs/
        labelsTs/
      pelvis/
        imagesTr/
        labelsTr/
        imagesVal/
        labelsVal/
        imagesTs/
        labelsTs/

Usage:
    python prepare_synthrad_data.py \
        --synthrad_root /path/to/Task1 \
        --output_root   /path/to/mc_ddpm_data \
        --anatomy       brain          # or pelvis, or both
        --train_frac    0.7 \
        --val_frac      0.1

Notes:
  - The remaining fraction after train + val becomes the test set.
  - Files are COPIED (not moved) so the original data is untouched.
  - Voxel intensities are NOT modified here; normalisation happens
    inside the training notebook exactly as in the original paper.
  - Brain:  1x1x1 mm³  → no resampling needed
  - Pelvis: 1x1x2.5 mm³ → no resampling needed (model handles anisotropy
    via asymmetric window sizes N_L)
"""

import argparse
import os
import shutil
import random
from pathlib import Path


# ── Recommended train/val/test splits ────────────────────────────────────────
# SynthRAD2023 Task 1 provides:
#   Brain:  180 training patients  (we further split into tr/val/ts)
#   Pelvis: 180 training patients
# The original MC-DDPM paper used ~28 brain and ~20 prostate patients.
# For SynthRAD you have far more data which is better. Suggested split:
#   70% train / 10% val / 20% test
# ─────────────────────────────────────────────────────────────────────────────

ANATOMY_MAP = {
    "brain":  "brain",
    "pelvis": "pelvis",   # SynthRAD calls it pelvis; paper calls it prostate
}


def get_patient_dirs(synthrad_root: Path, anatomy: str):
    """Return sorted list of patient directories for a given anatomy."""
    anatomy_dir = synthrad_root / anatomy
    if not anatomy_dir.exists():
        raise FileNotFoundError(f"Could not find anatomy folder: {anatomy_dir}")
    patients = sorted([
        p for p in anatomy_dir.iterdir()
        if p.is_dir() and (p / "mr.nii.gz").exists() and (p / "ct.nii.gz").exists()
    ])
    if not patients:
        raise ValueError(f"No valid patient folders found in {anatomy_dir}")
    return patients


def split_patients(patients, train_frac, val_frac, seed=42):
    """Randomly split patients into train / val / test lists."""
    random.seed(seed)
    shuffled = patients.copy()
    random.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    train = shuffled[:n_train]
    val   = shuffled[n_train:n_train + n_val]
    test  = shuffled[n_train + n_val:]
    return train, val, test


def copy_patient(patient_dir: Path, images_dst: Path, labels_dst: Path, idx: int, anatomy: str):
    """
    Copy mr.nii.gz → images_dst/  and  ct.nii.gz → labels_dst/
    Files are renamed to a zero-padded index for consistency.
    e.g.  brain_001_mr.nii.gz  /  brain_001_ct.nii.gz
    """
    prefix = f"{anatomy}_{idx:03d}"
    src_mr = patient_dir / "mr.nii.gz"
    src_ct = patient_dir / "ct.nii.gz"

    dst_mr = images_dst / f"{prefix}_mr.nii.gz"
    dst_ct = labels_dst / f"{prefix}_ct.nii.gz"

    shutil.copy2(src_mr, dst_mr)
    shutil.copy2(src_ct, dst_ct)
    return prefix


def prepare_anatomy(synthrad_root: Path, output_root: Path, anatomy: str,
                    train_frac: float, val_frac: float, seed: int):
    print(f"\n{'='*60}")
    print(f"  Processing anatomy: {anatomy.upper()}")
    print(f"{'='*60}")

    patients = get_patient_dirs(synthrad_root, anatomy)
    print(f"  Found {len(patients)} patients with paired MR+CT")

    train, val, test = split_patients(patients, train_frac, val_frac, seed)
    print(f"  Split → train: {len(train)}  val: {len(val)}  test: {len(test)}")

    # Create output directories
    split_dirs = {
        "train": ("imagesTr", "labelsTr", train),
        "val":   ("imagesVal", "labelsVal", val),
        "test":  ("imagesTs",  "labelsTs",  test),
    }

    anatomy_out = output_root / anatomy
    for split_name, (img_folder, lbl_folder, pat_list) in split_dirs.items():
        img_dir = anatomy_out / img_folder
        lbl_dir = anatomy_out / lbl_folder
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  [{split_name.upper()}] → {img_folder} / {lbl_folder}")
        for i, patient_dir in enumerate(pat_list):
            prefix = copy_patient(patient_dir, img_dir, lbl_dir, i + 1, anatomy)
            print(f"    Copied {patient_dir.name} → {prefix}")

    print(f"\n  Done. Output written to: {anatomy_out}")
    return {
        "anatomy":    anatomy,
        "n_train":    len(train),
        "n_val":      len(val),
        "n_test":     len(test),
        "output_dir": str(anatomy_out),
    }


def print_summary(summaries):
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    for s in summaries:
        print(f"\n  {s['anatomy'].upper()}")
        print(f"    Train : {s['n_train']}")
        print(f"    Val   : {s['n_val']}")
        print(f"    Test  : {s['n_test']}")
        print(f"    Output: {s['output_dir']}")

    print("\n  NEXT STEPS:")
    print("  1. Open MC-IDDPM main.ipynb")
    print("  2. Update the data_root variable to your output_root path")
    print("  3. Set anatomy = 'brain' or 'pelvis'")
    print("  4. Adjust img_size, patch_size, and hyperparameters")
    print("     (see mc_ddpm_train_config.py for recommended values)")
    print("  5. Run training cells\n")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare SynthRAD2023 data for MC-DDPM training"
    )
    parser.add_argument("--synthrad_root", type=str, required=True,
                        help="Path to SynthRAD2023 Task1 root folder")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Where to write the MC-DDPM ready data")
    parser.add_argument("--anatomy", type=str, default="both",
                        choices=["brain", "pelvis", "both"],
                        help="Which anatomy to prepare (default: both)")
    parser.add_argument("--train_frac", type=float, default=0.70,
                        help="Fraction of patients for training (default: 0.70)")
    parser.add_argument("--val_frac", type=float, default=0.10,
                        help="Fraction of patients for validation (default: 0.10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible splits (default: 42)")
    args = parser.parse_args()

    synthrad_root = Path(args.synthrad_root)
    output_root   = Path(args.output_root)

    if not synthrad_root.exists():
        raise FileNotFoundError(f"synthrad_root not found: {synthrad_root}")

    if args.train_frac + args.val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1.0 to leave room for test set")

    anatomies = ["brain", "pelvis"] if args.anatomy == "both" else [args.anatomy] 
    summaries = []
    for anatomy in anatomies:
        s = prepare_anatomy(synthrad_root, output_root, anatomy,
                            args.train_frac, args.val_frac, args.seed)
        summaries.append(s)

    print_summary(summaries)


if __name__ == "__main__":
    main()
