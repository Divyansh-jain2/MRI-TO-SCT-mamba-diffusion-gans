import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_and_train_pix2pix import (  # noqa: E402
    SPLITS,
    build_split_dataframe,
    collect_cases,
    split_cases_kfold,
    write_fold_csvs,
)


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
        description="Create axial 2D slice CSVs from paired NIfTI MR/CT volumes and launch UNIT training."
    )
    parser.add_argument("--data-root", required=True, help="Path to mc_ddpm_data, the folder containing brain/ and pelvis/.")
    parser.add_argument("--anatomy", default="brain", choices=["brain", "pelvis"], help="Dataset subfolder to train on.")
    parser.add_argument("--config", default="configs/unit_train.yaml", help="UNIT training YAML path.")
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

    data_root = Path(args.data_root).expanduser()
    dataset_root = data_root / args.anatomy
    config_path = (REPO_ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    fold_dir = (
        Path(args.fold_dir).expanduser()
        if args.fold_dir
        else REPO_ROOT / "data" / "k_fold_cross_validation" / "folds_2d" / f"{args.anatomy}_unit_mc_ddpm"
    )
    if not fold_dir.is_absolute():
        fold_dir = (REPO_ROOT / fold_dir).resolve()

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
        [sys.executable, "-m", "src.model.Unit.train_kfold", "--config", str(config_path)],
        cwd=str(REPO_ROOT),
        check=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )


if __name__ == "__main__":
    main()
