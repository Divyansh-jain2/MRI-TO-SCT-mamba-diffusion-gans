import argparse
import os
import random
import re
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import pandas as pd
import yaml


SPLITS = {
    "train": ("imagesTr", "labelsTr"),
    "val": ("imagesVal", "labelsVal"),
    "test": ("imagesTs", "labelsTs"),
}


def case_id_from_name(path, suffix):
    name = path.name
    if name.endswith(".nii.gz"):
        name = name[:-7]
    else:
        name = path.stem
    return re.sub(f"{re.escape(suffix)}$", "", name)


def normalized_path(path):
    return str(path.resolve()).replace("\\", "/")


def find_pairs(dataset_root, split, mr_suffix, ct_suffix):
    image_dir_name, label_dir_name = SPLITS[split]
    image_dir = dataset_root / image_dir_name
    label_dir = dataset_root / label_dir_name

    if not image_dir.exists():
        raise FileNotFoundError(f"Missing MR directory: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Missing CT directory: {label_dir}")

    mr_files = sorted(image_dir.glob("*.nii.gz"))
    ct_files = sorted(label_dir.glob("*.nii.gz"))
    ct_by_case = {case_id_from_name(path, ct_suffix): path for path in ct_files}

    pairs = []
    missing = []
    for mr_path in mr_files:
        case_id = case_id_from_name(mr_path, mr_suffix)
        ct_path = ct_by_case.get(case_id)
        if ct_path is None:
            missing.append(mr_path.name)
            continue
        pairs.append((case_id, mr_path, ct_path))

    if missing:
        missing_preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"Missing CT pair for {len(missing)} MR files. First missing: {missing_preview}")

    return pairs


def axial_slice_count(path):
    shape = nib.load(str(path)).shape
    if len(shape) < 2:
        raise ValueError(f"Expected at least 2D NIfTI, got shape {shape}: {path}")
    if len(shape) == 2:
        return 1
    return shape[2]


def case_rows(patient_id, case_id, anatomy, mr_path, ct_path, wc, ww):
    rows = []
    mr_slices = axial_slice_count(mr_path)
    ct_slices = axial_slice_count(ct_path)
    if mr_slices != ct_slices:
        raise ValueError(
            f"Slice-count mismatch for {case_id}: MR has {mr_slices}, CT has {ct_slices}"
        )

    for z in range(mr_slices):
        rows.append(
            {
                "id_slice": f"{patient_id}_{z:03d}",
                "patient_id": patient_id,
                "center": anatomy,
                "slice_number": z,
                "slice_ct_path": normalized_path(ct_path),
                "slice_mri_path": normalized_path(mr_path),
                "wc": float(wc),
                "ww": float(ww),
            }
        )

    return rows


def build_split_dataframe(dataset_root, anatomy, split, wc, ww, mr_suffix, ct_suffix):
    rows = []
    for case_id, mr_path, ct_path in find_pairs(dataset_root, split, mr_suffix, ct_suffix):
        patient_id = f"{anatomy}_{split}_{case_id}"
        rows.extend(case_rows(patient_id, case_id, anatomy, mr_path, ct_path, wc, ww))

    if not rows:
        raise ValueError(f"No paired NIfTI files found for split '{split}' in {dataset_root}")

    return pd.DataFrame(rows)


def collect_cases(dataset_root, anatomy, wc, ww, mr_suffix, ct_suffix):
    cases = []
    for split in SPLITS:
        for case_id, mr_path, ct_path in find_pairs(dataset_root, split, mr_suffix, ct_suffix):
            patient_id = f"{anatomy}_{split}_{case_id}"
            rows = case_rows(patient_id, case_id, anatomy, mr_path, ct_path, wc, ww)
            cases.append({"patient_id": patient_id, "source_split": split, "rows": rows})

    if not cases:
        raise ValueError(f"No paired NIfTI files found in {dataset_root}")

    return cases


def split_cases_kfold(cases, cv, seed):
    shuffled = list(cases)
    random.Random(seed).shuffle(shuffled)
    folds = [shuffled[i::cv] for i in range(cv)]
    split_frames_by_fold = []

    for fold_idx in range(cv):
        test_cases = folds[fold_idx]
        val_cases = folds[(fold_idx + 1) % cv]
        train_cases = [
            case
            for idx, fold_cases in enumerate(folds)
            if idx not in {fold_idx, (fold_idx + 1) % cv}
            for case in fold_cases
        ]

        split_frames_by_fold.append(
            {
                "train": pd.DataFrame([row for case in train_cases for row in case["rows"]]),
                "val": pd.DataFrame([row for case in val_cases for row in case["rows"]]),
                "test": pd.DataFrame([row for case in test_cases for row in case["rows"]]),
            }
        )

    return split_frames_by_fold


def write_fold_csvs(fold_dir, split_frames_by_fold):
    fold_dir.mkdir(parents=True, exist_ok=True)
    for fold, split_frames in enumerate(split_frames_by_fold):
        current = fold_dir / str(fold)
        current.mkdir(parents=True, exist_ok=True)
        for split, frame in split_frames.items():
            frame.to_csv(current / f"{split}.csv", index=False)


def update_config(config_path, fold_dir, cv=None):
    with config_path.open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    cfg.setdefault("data", {})
    cfg["data"]["fold_dir"] = str(fold_dir).replace("\\", "/")
    if cv is not None:
        cfg["data"]["cv"] = int(cv)

    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(cfg, file, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(
        description="Create axial 2D slice CSVs from paired NIfTI MR/CT volumes and launch Pix2Pix training."
    )
    parser.add_argument("--data-root", required=True, help="Path to mc_ddpm_data, the folder containing brain/ and pelvis/.")
    parser.add_argument("--anatomy", default="brain", choices=["brain", "pelvis"], help="Dataset subfolder to train on.")
    parser.add_argument("--config", default="configs/pix2pix_train.yaml", help="Pix2Pix training YAML path.")
    parser.add_argument("--fold-dir", default=None, help="Where generated train/val/test CSVs should be written.")
    parser.add_argument("--cv", type=int, default=5, help="Number of cross-validation folds to create.")
    parser.add_argument("--split-mode", choices=["kfold", "fixed"], default="kfold", help="Use volume-level k-fold splits or keep the existing Tr/Val/Ts split.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used for k-fold volume shuffling.")
    parser.add_argument("--wc", type=float, default=50.0, help="CT window center written into the CSV.")
    parser.add_argument("--ww", type=float, default=400.0, help="CT window width written into the CSV.")
    parser.add_argument("--mr-suffix", default="_mr", help="Suffix removed from MR filenames to pair cases.")
    parser.add_argument("--ct-suffix", default="_ct", help="Suffix removed from CT filenames to pair cases.")
    parser.add_argument("--skip-train", action="store_true", help="Only create CSVs and update the config.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_root = Path(args.data_root).expanduser()
    dataset_root = data_root / args.anatomy
    config_path = (repo_root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    fold_dir = (
        Path(args.fold_dir).expanduser()
        if args.fold_dir
        else repo_root / "data" / "k_fold_cross_validation" / "folds_2d" / f"{args.anatomy}_mc_ddpm"
    )
    if not fold_dir.is_absolute():
        fold_dir = (repo_root / fold_dir).resolve()

    if args.split_mode == "kfold":
        cases = collect_cases(dataset_root, args.anatomy, args.wc, args.ww, args.mr_suffix, args.ct_suffix)
        if len(cases) < args.cv:
            raise ValueError(f"Need at least {args.cv} volumes for {args.cv}-fold CV, found {len(cases)}")
        split_frames_by_fold = split_cases_kfold(cases, args.cv, args.seed)
    else:
        split_frames = {
            split: build_split_dataframe(
                dataset_root=dataset_root,
                anatomy=args.anatomy,
                split=split,
                wc=args.wc,
                ww=args.ww,
                mr_suffix=args.mr_suffix,
                ct_suffix=args.ct_suffix,
            )
            for split in SPLITS
        }
        split_frames_by_fold = [split_frames for _ in range(args.cv)]

    write_fold_csvs(fold_dir, split_frames_by_fold)
    update_config(config_path, fold_dir, cv=args.cv)

    for fold, split_frames in enumerate(split_frames_by_fold):
        print(f"fold {fold}")
        for split, frame in split_frames.items():
            print(f"  {split}: {len(frame)} 2D slices from {frame['patient_id'].nunique()} volumes")
    print(f"CSV folds written to: {fold_dir}")
    print(f"Updated config: {config_path}")

    if args.skip_train:
        return

    subprocess.run(
        [sys.executable, "-m", "src.model.Pix2Pix.train_kfold", "--config", str(config_path)],
        cwd=str(repo_root),
        check=True,
        env={**os.environ, "PYTHONPATH": str(repo_root)},
    )


if __name__ == "__main__":
    main()
