"""
Inference script for Hybrid MRI Encoder + Cross-Attention Diffusion Model.
Loads the best checkpoint and runs on the test dataset.

Usage:
    python inference_hybrid.py
    python inference_hybrid.py --checkpoint ./checkpoints_brain_hybrid/checkpoint_epoch_300.pt
    python inference_hybrid.py --mc_runs 5        # more Monte-Carlo runs for smoother output
    python inference_hybrid.py --save_npy          # also save raw numpy arrays
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from monai.inferers import SlidingWindowInferer

# ── Project imports ──────────────────────────────────────────────────────────
from network.hybrid_model import HybridSwinVITModel
from diffusion.HybridGaussianDiffusion import HybridGaussianDiffusion
from diffusion.Create_diffusion import create_gaussian_diffusion

import glob
from torch.utils.data import Dataset
from natsort import natsorted
from monai.transforms import (
    Compose, RandSpatialCropSamplesd, ToTensord,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset (copied from main_hybrid.py to avoid importing the training script)
# ═══════════════════════════════════════════════════════════════════════════════
class CustomDataset(Dataset):
    def __init__(self, imgs_path, labels_path=None, train_flag=True):
        self.train_flag = train_flag
        self.files = natsorted(glob.glob(imgs_path + "*.npy"),
                               key=lambda y: y.lower())
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .npy files found in {imgs_path}")
        print(f"Found {len(self.files)} preprocessed volumes "
              f"[{'train' if train_flag else 'val/test'}]")

        self.patch_size = (64, 64, 4)
        self.patch_num  = 2
        self.patch_transform = Compose([
            RandSpatialCropSamplesd(
                keys=["image", "label"],
                roi_size=self.patch_size,
                num_samples=self.patch_num,
                random_size=False,
            ),
            ToTensord(keys=["image", "label"]),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data   = np.load(self.files[idx])
        mr_vol = data[0]
        ct_vol = data[1]

        data_dict = {
            "image": mr_vol[np.newaxis],
            "label": ct_vol[np.newaxis],
        }

        if not self.train_flag:
            img_tensor   = torch.from_numpy(mr_vol[np.newaxis]).float()
            label_tensor = torch.from_numpy(ct_vol[np.newaxis]).float()
        else:
            out   = self.patch_transform(data_dict)
            img   = np.zeros([self.patch_num, self.patch_size[0], self.patch_size[1], self.patch_size[2]])
            label = np.zeros([self.patch_num, self.patch_size[0], self.patch_size[1], self.patch_size[2]])
            for i, sample in enumerate(out):
                img[i]   = sample["image"].numpy()
                label[i] = sample["label"].numpy()
            img_tensor   = torch.unsqueeze(torch.from_numpy(img.copy()),   1).float()
            label_tensor = torch.unsqueeze(torch.from_numpy(label.copy()), 1).float()

        return img_tensor, label_tensor

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG  (must match training exactly)
# ═══════════════════════════════════════════════════════════════════════════════
DATA_ROOT    = '/home/teaching/Desktop/Munish_Synthetic_CT/mc_ddpm_data/brain_npy'
SAVE_DIR     = './checkpoints_brain_hybrid'
DEFAULT_CKPT = os.path.join(SAVE_DIR, 'best_model.pt')
OUTPUT_DIR   = './inference_results_hybrid'

img_size     = (192, 192, 96)
patch_size   = (64, 64, 4)
CT_CLIP      = (-1024, 1650)

# Model config (must match main_hybrid.py)
ENC_CHANNELS        = (64, 128, 192, 256)
ENCODER_WINDOW      = (4, 4, 4)
ENCODER_NUM_HEADS   = (4, 4, 8, 8)
ENCODER_POOL_KERNEL = (2, 2, 1)
MODEL_CHANNELS      = 64
CHANNEL_MULT        = (1, 2, 4)
NUM_RES_BLOCKS      = [2, 2, 2]
SAMPLE_KERNEL       = ([2,2,2], [2,2,1])
NUM_HEADS           = [4, 4, 8]
ATTENTION_RES       = "32,16,8"

# Diffusion config
DIFFUSION_STEPS    = 1000
TIMESTEP_RESPACING = [50]
NOISE_SCHEDULE     = 'linear'


def parse_args():
    parser = argparse.ArgumentParser(description='Hybrid model inference')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CKPT,
                        help='Path to model checkpoint (default: best_model.pt)')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help='Directory to save results')
    parser.add_argument('--mc_runs', type=int, default=3,
                        help='Number of Monte-Carlo sampling runs (default: 3)')
    parser.add_argument('--save_npy', action='store_true',
                        help='Also save raw numpy arrays')
    parser.add_argument('--split', type=str, default='test',
                        choices=['test', 'val'],
                        help='Which split to run inference on (default: test)')
    return parser.parse_args()


def build_model(device):
    """Build model with same config as training."""
    attention_ds = [int(r) for r in ATTENTION_RES.split(",")]

    model = HybridSwinVITModel(
        image_size=patch_size,
        in_channels=1,
        model_channels=MODEL_CHANNELS,
        out_channels=2,
        dims=3,
        sample_kernel=SAMPLE_KERNEL,
        num_res_blocks=NUM_RES_BLOCKS,
        attention_resolutions=tuple(attention_ds),
        dropout=0,
        channel_mult=CHANNEL_MULT,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=NUM_HEADS,
        window_size=None,
        num_head_channels=64,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_new_attention_order=False,
        enc_channels=ENC_CHANNELS,
        freeze_encoder=False,
        encoder_window_size=ENCODER_WINDOW,
        encoder_num_heads=ENCODER_NUM_HEADS,
        encoder_pool_kernel=ENCODER_POOL_KERNEL,
    ).to(device)

    return model


def build_diffusion():
    """Build diffusion process with same config as training."""
    return create_gaussian_diffusion(
        steps=DIFFUSION_STEPS,
        learn_sigma=True,
        sigma_small=False,
        noise_schedule=NOISE_SCHEDULE,
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=True,
        timestep_respacing=TIMESTEP_RESPACING,
        diffusion_class=HybridGaussianDiffusion,
    )


def denorm_to_hu(img_norm, ct_clip=CT_CLIP):
    """Convert from [-1, 1] normalised back to Hounsfield Units."""
    lo, hi = ct_clip
    return (img_norm + 1.0) / 2.0 * (hi - lo) + lo


def save_visualisation(mri, ct_gt, ct_pred, save_path, sample_idx):
    """Save a side-by-side comparison figure."""
    mid = mri.shape[-1] // 2

    def _np(vol):
        s = vol[0, 0, :, :, mid].float().cpu().numpy()
        return (s - s.min()) / (s.max() - s.min() + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(_np(mri),     cmap='gray'); axes[0].set_title('MRI Input')
    axes[1].imshow(_np(ct_gt),   cmap='gray'); axes[1].set_title('CT Ground Truth')
    axes[2].imshow(_np(ct_pred), cmap='gray'); axes[2].set_title('Predicted CT')
    for a in axes:
        a.axis('off')
    plt.suptitle(f'Sample {sample_idx} — Hybrid Diffusion Inference', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Checkpoint: {args.checkpoint}')
    print(f'MC runs: {args.mc_runs}')
    print(f'Split: {args.split}')

    # ── Output dirs ──
    os.makedirs(args.output_dir, exist_ok=True)
    nifti_dir = os.path.join(args.output_dir, 'nifti')
    vis_dir   = os.path.join(args.output_dir, 'vis')
    os.makedirs(nifti_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    if args.save_npy:
        npy_dir = os.path.join(args.output_dir, 'npy')
        os.makedirs(npy_dir, exist_ok=True)

    # ── Build model & diffusion ──
    model = build_model(device)
    diffusion = build_diffusion()

    # ── Load checkpoint ──
    if not os.path.exists(args.checkpoint):
        print(f'ERROR: Checkpoint not found: {args.checkpoint}')
        sys.exit(1)

    print(f'Loading checkpoint...')
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f'Model loaded successfully.')

    # ── Dataset ──
    if args.split == 'test':
        dataset = CustomDataset(DATA_ROOT + '/imagesTs/', train_flag=False)
    else:
        dataset = CustomDataset(DATA_ROOT + '/imagesVal/', train_flag=False)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=2
    )
    print(f'Samples to process: {len(loader)}')

    # ── Inferer (sliding window) ──
    inferer = SlidingWindowInferer(patch_size, sw_batch_size=2, overlap=0.25, mode='constant')

    def diffusion_sampling(condition, model):
        return diffusion.p_sample_loop(
            model,
            (condition.shape[0], 1,
             condition.shape[2], condition.shape[3], condition.shape[4]),
            condition=condition,
            clip_denoised=True,
        )

    # ── Run inference ──
    all_l1 = []
    total_start = time.time()

    with torch.no_grad():
        for idx, (mri, ct_gt) in enumerate(loader):
            sample_start = time.time()
            mri   = mri.to(device)
            ct_gt = ct_gt.to(device)

            # Monte-Carlo averaged prediction
            mc_preds = []
            for mc in range(args.mc_runs):
                with torch.cuda.amp.autocast():
                    pred = inferer(mri, diffusion_sampling, model)
                mc_preds.append(pred)
            ct_pred = torch.stack(mc_preds).mean(dim=0)

            # ── Metrics ──
            l1 = torch.nn.functional.l1_loss(ct_pred, ct_gt).item()
            all_l1.append(l1)

            sample_time = time.time() - sample_start
            print(f'[{idx+1}/{len(loader)}] L1: {l1:.6f} | Time: {sample_time:.1f}s')

            # ── Save NIfTI ──
            pred_np = ct_pred[0, 0].cpu().numpy()
            pred_hu = denorm_to_hu(pred_np)
            nib.save(
                nib.Nifti1Image(pred_hu, np.eye(4)),
                os.path.join(nifti_dir, f'pred_sample_{idx:03d}.nii.gz')
            )

            gt_np = ct_gt[0, 0].cpu().numpy()
            gt_hu = denorm_to_hu(gt_np)
            nib.save(
                nib.Nifti1Image(gt_hu, np.eye(4)),
                os.path.join(nifti_dir, f'gt_sample_{idx:03d}.nii.gz')
            )

            # ── Save numpy (optional) ──
            if args.save_npy:
                np.save(os.path.join(npy_dir, f'pred_{idx:03d}.npy'), pred_np)
                np.save(os.path.join(npy_dir, f'gt_{idx:03d}.npy'), gt_np)

            # ── Save visualisation ──
            save_visualisation(
                mri.cpu(), ct_gt.cpu(), ct_pred.cpu(),
                os.path.join(vis_dir, f'sample_{idx:03d}.png'), idx
            )

    # ── Summary ──
    total_time = time.time() - total_start
    mean_l1 = np.mean(all_l1)
    std_l1  = np.std(all_l1)

    print(f'\n{"="*60}')
    print(f'INFERENCE COMPLETE')
    print(f'{"="*60}')
    print(f'Samples processed : {len(all_l1)}')
    print(f'Mean L1 loss      : {mean_l1:.6f} ± {std_l1:.6f}')
    print(f'Min  L1 loss      : {np.min(all_l1):.6f}')
    print(f'Max  L1 loss      : {np.max(all_l1):.6f}')
    print(f'Total time        : {total_time/60:.1f} minutes')
    print(f'Results saved to  : {args.output_dir}')
    print(f'{"="*60}')

    # Save metrics to text file
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        f.write(f'Checkpoint: {args.checkpoint}\n')
        f.write(f'Split: {args.split}\n')
        f.write(f'MC runs: {args.mc_runs}\n')
        f.write(f'Samples: {len(all_l1)}\n')
        f.write(f'Mean L1: {mean_l1:.6f} ± {std_l1:.6f}\n')
        f.write(f'Min  L1: {np.min(all_l1):.6f}\n')
        f.write(f'Max  L1: {np.max(all_l1):.6f}\n\n')
        f.write('Per-sample L1:\n')
        for i, l in enumerate(all_l1):
            f.write(f'  Sample {i:03d}: {l:.6f}\n')


if __name__ == '__main__':
    main()
