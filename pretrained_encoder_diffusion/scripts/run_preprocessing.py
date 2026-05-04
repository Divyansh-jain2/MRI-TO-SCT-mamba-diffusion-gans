# run_preprocessing.py  — run ONCE from terminal before training
# python run_preprocessing.py

import os, glob, numpy as np, nibabel as nib
from natsort import natsorted
from monai.transforms import Compose, AddChanneld, ResizeWithPadOrCropd

img_size  = (192, 192, 96)
CT_CLIP   = (-1024, 1650)
DATA_ROOT = '/home/teaching/Desktop/Munish_Synthetic_CT/mc_ddpm_data/pelvis'
NPY_ROOT  = '/home/teaching/Desktop/Munish_Synthetic_CT/mc_ddpm_data/pelvis_npy'

pad_crop = Compose([
    AddChanneld(keys=["image", "label"]),
    ResizeWithPadOrCropd(
        keys=["image", "label"],
        spatial_size=img_size,
        constant_values=-1,
    ),
])

for split in ["imagesTr/labelsTr", "imagesVal/labelsVal", "imagesTs/labelsTs"]:
    img_dir, lbl_dir = split.split("/")
    mr_files = natsorted(glob.glob(os.path.join(DATA_ROOT, img_dir, "*.nii.gz")))
    ct_files = natsorted(glob.glob(os.path.join(DATA_ROOT, lbl_dir, "*.nii.gz")))

    out_dir = os.path.join(NPY_ROOT, img_dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nProcessing {img_dir} — {len(mr_files)} volumes...")

    for mr_f, ct_f in zip(mr_files, ct_files):
        pid = os.path.basename(mr_f).replace("_mr.nii.gz", "")

        # Load
        mr_vol = nib.load(mr_f).get_fdata(dtype=np.float32)
        ct_vol = nib.load(ct_f).get_fdata(dtype=np.float32)

        # Normalise MRI independently to [-1, 1]
        vmin, vmax = mr_vol.min(), mr_vol.max()
        mr_vol = ((mr_vol - vmin) / (vmax - vmin + 1e-8)) * 2.0 - 1.0

        # Normalise CT: clip HU then scale to [-1, 1]
        lo, hi = CT_CLIP
        ct_vol = np.clip(ct_vol, lo, hi)
        ct_vol = ((ct_vol - lo) / (hi - lo)) * 2.0 - 1.0

        # Pad/crop to fixed img_size — no spacing, no orientation (already done)
        out = pad_crop({"image": mr_vol, "label": ct_vol})
        mr_out = np.array(out["image"]).squeeze()   # (192, 192, 96)
        ct_out = np.array(out["label"]).squeeze()

        # Save as single .npy file containing both
        save_path = os.path.join(out_dir, f"{pid}.npy")
        np.save(save_path, np.stack([mr_out, ct_out]))  # shape (2, 192, 192, 96)
        print(f"  Saved {pid}.npy  shape={mr_out.shape}")

print("\nPreprocessing complete.")
