import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torchvision.utils import save_image
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.Unit.generator_model import Encoder, Generator, ResidualBlock  # noqa: E402
import src.utils.util_data as util_data  # noqa: E402
import src.utils.util_general as util_general  # noqa: E402


def _load_cfg(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _resolve_repo_path(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _make_panel(mri, gt_ct, fake_ct):
    spacer = torch.ones((1, mri.shape[-2], 8), dtype=mri.dtype, device=mri.device)
    return torch.cat([mri, spacer, gt_ct, spacer, fake_ct], dim=-1)


def _sanitize_name(value):
    return str(value).replace("\\", "_").replace("/", "_").replace(" ", "_")


def main():
    parser = argparse.ArgumentParser(
        description="Export flat MRI | ground-truth CT | synthetic CT comparison panels from a trained UNIT fold."
    )
    parser.add_argument("--config", default="configs/unit_train.yaml", help="UNIT training YAML path.")
    parser.add_argument("--fold", type=int, default=0, help="Fold index to export from.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="CSV split to sample from.")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of panels to export.")
    parser.add_argument("--checkpoint-e1", default=None, help="Optional encoder E1 checkpoint path.")
    parser.add_argument("--checkpoint-g2", default=None, help="Optional generator G2 checkpoint path.")
    parser.add_argument(
        "--output-dir",
        default="results/unit/comparison_panels_flat/fold0",
        help="Folder where flat comparison panel PNGs and manifest.csv will be saved.",
    )
    parser.add_argument("--device", default=None, help="Override device, e.g. cuda or cpu.")
    parser.add_argument("--start-index", type=int, default=0, help="Start row in the selected split CSV.")
    args = parser.parse_args()

    cfg_path = _resolve_repo_path(args.config)
    cfg = _load_cfg(cfg_path)

    device = torch.device(
        args.device
        if args.device is not None
        else cfg["device"]["cuda_device"]
        if torch.cuda.is_available()
        else "cpu"
    )

    fold_csv = _resolve_repo_path(cfg["data"]["fold_dir"]) / str(args.fold) / f"{args.split}.csv"
    if not fold_csv.exists():
        raise FileNotFoundError(f"Split CSV not found: {fold_csv}")

    model_root = _resolve_repo_path(cfg["data"]["model_dir"]) / cfg["exp_name"] / str(args.fold)
    checkpoint_e1 = Path(args.checkpoint_e1).expanduser() if args.checkpoint_e1 else model_root / cfg["trainer"]["CHECKPOINT_E1"]
    checkpoint_g2 = Path(args.checkpoint_g2).expanduser() if args.checkpoint_g2 else model_root / cfg["trainer"]["CHECKPOINT_G2"]

    if not checkpoint_e1.is_absolute():
        checkpoint_e1 = REPO_ROOT / checkpoint_e1
    if not checkpoint_g2.is_absolute():
        checkpoint_g2 = REPO_ROOT / checkpoint_g2

    if not checkpoint_e1.exists():
        raise FileNotFoundError(f"UNIT E1 checkpoint not found: {checkpoint_e1}")
    if not checkpoint_g2.exists():
        raise FileNotFoundError(f"UNIT G2 checkpoint not found: {checkpoint_g2}")

    output_dir = _resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(fold_csv, index_col="id_slice")
    df = df.iloc[args.start_index : args.start_index + args.num_samples]
    dataset = util_data.ImgDataset(data=df, cfg_data=cfg["data"], step=args.split, do_augmentation=False)

    shared_dim = cfg["model"]["dim"] * (2 ** cfg["model"]["n_downsample"])
    shared_e = ResidualBlock(features=shared_dim).to(device)
    shared_g = ResidualBlock(features=shared_dim).to(device)

    e1 = Encoder(
        in_channels=1,
        dim=cfg["model"]["dim"],
        n_downsample=cfg["model"]["n_downsample"],
        shared_block=shared_e,
    ).to(device).eval()
    g2 = Generator(
        out_channels=1,
        dim=cfg["model"]["dim"],
        n_upsample=cfg["model"]["n_downsample"],
        shared_block=shared_g,
        out_activation="sigmoid",
    ).to(device).eval()

    util_general.load_checkpoint(str(checkpoint_e1), e1, map_location=device)
    util_general.load_checkpoint(str(checkpoint_g2), g2, map_location=device)

    manifest_rows = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            mri, gt_ct, sample_id = dataset[idx]
            mri_b = mri.unsqueeze(0).to(device)
            _, latent = e1(mri_b)
            fake_ct = g2(latent).squeeze(0).cpu().clamp(0, 1)
            mri = mri.cpu().clamp(0, 1)
            gt_ct = gt_ct.cpu().clamp(0, 1)

            row = df.loc[sample_id]
            export_index = idx + args.start_index
            panel = _make_panel(mri, gt_ct, fake_ct)
            panel_name = f"{export_index:03d}_{_sanitize_name(sample_id)}_mri_gtct_synthct.png"
            save_image(panel, output_dir / panel_name)

            manifest_rows.append(
                {
                    "export_index": export_index,
                    "panel_png": panel_name,
                    "id_slice": sample_id,
                    "patient_id": row["patient_id"],
                    "slice_number": row["slice_number"],
                    "mri_path": row["slice_mri_path"],
                    "ground_truth_ct_path": row["slice_ct_path"],
                }
            )

    pd.DataFrame(manifest_rows).to_csv(output_dir / "manifest.csv", index=False)
    print(f"Saved {len(manifest_rows)} flat comparison panels to: {output_dir}")
    print("Each PNG is ordered left-to-right as: MRI | ground-truth CT | synthetic CT")


if __name__ == "__main__":
    main()
