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
import torch.nn.functional as F
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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
OUTPUT_DIR   = './inference_hybrid_results2'

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
TIMESTEP_RESPACING = [200]
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


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics: PSNR, SSIM, MAE (in HU)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_psnr(pred, gt, data_range=2.0):
    """PSNR in dB. data_range=2.0 for [-1,1] normalised volumes."""
    mse = F.mse_loss(pred, gt)
    if mse == 0:
        return 100.0
    return (10 * torch.log10(torch.tensor(data_range ** 2) / mse)).item()


def _gaussian_kernel_2d(window_size=11, sigma=1.5):
    x = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    gauss /= gauss.sum()
    kernel = gauss.unsqueeze(1) * gauss.unsqueeze(0)
    return kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, ws, ws]


def compute_ssim(pred, gt, window_size=11):
    """Compute SSIM averaged over all axial slices. Returns scalar."""
    pred = pred.float().cpu()
    gt = gt.float().cpu()
    kernel = _gaussian_kernel_2d(window_size)
    pad = window_size // 2
    C1, C2 = 0.01 ** 2, 0.03 ** 2

    # Average SSIM over all axial (last dim) slices
    B, C, H, W, D = pred.shape
    ssim_vals = []
    for d in range(D):
        p = pred[:, :, :, :, d]  # [B, 1, H, W]
        g = gt[:, :, :, :, d]
        mu1 = F.conv2d(p, kernel, padding=pad)
        mu2 = F.conv2d(g, kernel, padding=pad)
        sigma1_sq = F.conv2d(p * p, kernel, padding=pad) - mu1 * mu1
        sigma2_sq = F.conv2d(g * g, kernel, padding=pad) - mu2 * mu2
        sigma12 = F.conv2d(p * g, kernel, padding=pad) - mu1 * mu2
        num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
        den = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_vals.append((num / den).mean().item())
    return np.mean(ssim_vals)


def compute_mae_hu(pred_hu, gt_hu):
    """Mean Absolute Error in Hounsfield Units."""
    return np.mean(np.abs(pred_hu - gt_hu))


# ═══════════════════════════════════════════════════════════════════════════════
# Visualisation: 3 views × 4 columns (MRI, GT CT, Synth CT, Abs Error)
# ═══════════════════════════════════════════════════════════════════════════════
def save_visualisation(mri_hu, gt_hu, pred_hu, save_path, sample_idx, metrics):
    """
    Creates a publication-quality 3-row × 4-column figure:
      Rows:    Axial, Coronal, Sagittal
      Columns: MRI Input, Ground Truth CT, Synthetic CT, Absolute Error
    
    mri_hu:  [H, W, D] numpy (normalised scale)
    gt_hu:   [H, W, D] numpy (HU scale)
    pred_hu: [H, W, D] numpy (HU scale)
    metrics: dict with 'psnr', 'ssim', 'mae', 'l1'
    """
    H, W, D = gt_hu.shape
    error = np.abs(pred_hu - gt_hu)

    # Slice indices (middle of each axis)
    ax_idx = D // 2       # axial:    slice along Z
    cor_idx = W // 2      # coronal:  slice along Y (H axis)
    sag_idx = H // 2      # sagittal: slice along X (W axis)

    def norm01(x):
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-8)

    # Extract slices: [view][column] — each is a 2D array
    views = {
        'Axial': {
            'mri':   norm01(mri_hu[:, :, ax_idx]),
            'gt':    gt_hu[:, :, ax_idx],
            'pred':  pred_hu[:, :, ax_idx],
            'error': error[:, :, ax_idx],
        },
        'Coronal': {
            'mri':   norm01(mri_hu[:, cor_idx, :]),
            'gt':    gt_hu[:, cor_idx, :],
            'pred':  pred_hu[:, cor_idx, :],
            'error': error[:, cor_idx, :],
        },
        'Sagittal': {
            'mri':   norm01(mri_hu[sag_idx, :, :]),
            'gt':    gt_hu[sag_idx, :, :],
            'pred':  pred_hu[sag_idx, :, :],
            'error': error[sag_idx, :, :],
        },
    }

    # CT display range (HU)
    ct_vmin, ct_vmax = -200, 200
    err_vmax = 200

    fig = plt.figure(figsize=(16, 12), facecolor='black')
    gs = GridSpec(3, 5, figure=fig, width_ratios=[1, 1, 1, 1, 0.05],
                  hspace=0.05, wspace=0.05)

    col_titles = ['MRI Input', 'Ground Truth CT', 'Synthetic CT', 'Absolute Error']

    for row_i, (view_name, slices) in enumerate(views.items()):
        for col_i, (key, cmap_name) in enumerate([
            ('mri', 'gray'), ('gt', 'gray'), ('pred', 'gray'), ('error', 'hot')
        ]):
            ax = fig.add_subplot(gs[row_i, col_i])
            img = slices[key]

            if key == 'mri':
                ax.imshow(img, cmap=cmap_name, vmin=0, vmax=1, aspect='auto')
            elif key == 'error':
                im = ax.imshow(img, cmap=cmap_name, vmin=0, vmax=err_vmax, aspect='auto')
            else:
                ax.imshow(img, cmap=cmap_name, vmin=ct_vmin, vmax=ct_vmax, aspect='auto')

            ax.axis('off')

            # Column titles (top row only)
            if row_i == 0:
                ax.set_title(col_titles[col_i], fontsize=13, color='white', pad=8)

            # Row labels (left column only)
            if col_i == 0:
                ax.text(-0.05, 0.5, view_name, transform=ax.transAxes,
                        fontsize=12, color='white', ha='right', va='center',
                        rotation=90)

    # Colorbar for error maps
    cbar_ax = fig.add_subplot(gs[:, 4])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label('Absolute Error (HU)', color='white', fontsize=11)
    cb.ax.yaxis.set_tick_params(color='white')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')

    # Metrics text
    metrics_text = (
        f"MAE: {metrics['mae']:.1f} HU  |  "
        f"PSNR: {metrics['psnr']:.2f} dB  |  "
        f"SSIM: {metrics['ssim']:.4f}  |  "
        f"L1: {metrics['l1']:.6f}"
    )
    fig.suptitle(
        f"Sample {sample_idx}\n{metrics_text}",
        fontsize=14, color='white', y=0.98
    )

    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='black', edgecolor='none')
    plt.close(fig)


def main():
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Checkpoint: {args.checkpoint}')
    print(f'MC runs: {args.mc_runs}')
    print(f'Split: test')

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

    # ── Dataset (test set only) ──
    dataset = CustomDataset(DATA_ROOT + '/imagesTs/', train_flag=False)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=2
    )
    total_samples = len(loader)
    print(f'Samples to process: {total_samples}')

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
    all_l1, all_psnr, all_ssim, all_mae = [], [], [], []
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

            # ── Metrics (normalised domain) ──
            l1   = torch.nn.functional.l1_loss(ct_pred, ct_gt).item()
            psnr_val = compute_psnr(ct_pred, ct_gt, data_range=2.0)
            ssim_val = compute_ssim(ct_pred, ct_gt)

            # ── Convert to HU for MAE ──
            pred_np = ct_pred[0, 0].cpu().numpy()
            gt_np   = ct_gt[0, 0].cpu().numpy()
            pred_hu = denorm_to_hu(pred_np)
            gt_hu   = denorm_to_hu(gt_np)
            mae_hu  = compute_mae_hu(pred_hu, gt_hu)

            all_l1.append(l1)
            all_psnr.append(psnr_val)
            all_ssim.append(ssim_val)
            all_mae.append(mae_hu)

            sample_time = time.time() - sample_start
            elapsed = time.time() - total_start
            avg_per_sample = elapsed / (idx + 1)
            eta_min = avg_per_sample * (total_samples - idx - 1) / 60

            print(
                f'[{idx+1}/{total_samples}] '
                f'L1: {l1:.6f} | PSNR: {psnr_val:.2f}dB | '
                f'SSIM: {ssim_val:.4f} | MAE: {mae_hu:.1f}HU | '
                f'Time: {sample_time:.1f}s | '
                f'Progress: {idx+1}/{total_samples} done | ETA: {eta_min:.1f}min'
            )

            # ── Save NIfTI ──
            nib.save(
                nib.Nifti1Image(pred_hu, np.eye(4)),
                os.path.join(nifti_dir, f'pred_sample_{idx:03d}.nii.gz')
            )
            nib.save(
                nib.Nifti1Image(gt_hu, np.eye(4)),
                os.path.join(nifti_dir, f'gt_sample_{idx:03d}.nii.gz')
            )

            # ── Save numpy (optional) ──
            if args.save_npy:
                np.save(os.path.join(npy_dir, f'pred_{idx:03d}.npy'), pred_np)
                np.save(os.path.join(npy_dir, f'gt_{idx:03d}.npy'), gt_np)

            # ── Save 3-view visualisation with error map ──
            mri_np = mri[0, 0].cpu().numpy()
            sample_metrics = {
                'l1': l1, 'psnr': psnr_val, 'ssim': ssim_val, 'mae': mae_hu
            }
            save_visualisation(
                mri_np, gt_hu, pred_hu,
                os.path.join(vis_dir, f'sample_{idx:03d}.png'),
                idx, sample_metrics
            )

    # ── Summary ──
    total_time = time.time() - total_start

    print(f'\n{"="*70}')
    print(f'INFERENCE COMPLETE')
    print(f'{"="*70}')
    print(f'Samples processed : {len(all_l1)}')
    print(f'Mean L1 loss      : {np.mean(all_l1):.6f} ± {np.std(all_l1):.6f}')
    print(f'Mean PSNR         : {np.mean(all_psnr):.2f} ± {np.std(all_psnr):.2f} dB')
    print(f'Mean SSIM         : {np.mean(all_ssim):.4f} ± {np.std(all_ssim):.4f}')
    print(f'Mean MAE (HU)     : {np.mean(all_mae):.1f} ± {np.std(all_mae):.1f} HU')
    print(f'Total time        : {total_time/60:.1f} minutes')
    print(f'Results saved to  : {args.output_dir}')
    print(f'{"="*70}')

    # ── Save metrics to text file ──
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        f.write(f'Checkpoint: {args.checkpoint}\n')
        f.write(f'Split: test\n')
        f.write(f'MC runs: {args.mc_runs}\n')
        f.write(f'Samples: {len(all_l1)}\n\n')
        f.write(f'=== AGGREGATE METRICS ===\n')
        f.write(f'Mean L1:       {np.mean(all_l1):.6f} ± {np.std(all_l1):.6f}\n')
        f.write(f'Mean PSNR:     {np.mean(all_psnr):.2f} ± {np.std(all_psnr):.2f} dB\n')
        f.write(f'Mean SSIM:     {np.mean(all_ssim):.4f} ± {np.std(all_ssim):.4f}\n')
        f.write(f'Mean MAE (HU): {np.mean(all_mae):.1f} ± {np.std(all_mae):.1f} HU\n\n')
        f.write(f'=== PER-SAMPLE METRICS ===\n')
        f.write(f'{"Sample":>8} {"L1":>10} {"PSNR(dB)":>10} {"SSIM":>10} {"MAE(HU)":>10}\n')
        f.write(f'{"-"*50}\n')
        for i in range(len(all_l1)):
            f.write(
                f'{i:>8d} {all_l1[i]:>10.6f} {all_psnr[i]:>10.2f} '
                f'{all_ssim[i]:>10.4f} {all_mae[i]:>10.1f}\n'
            )

    # ── Save summary bar plots ──
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    sample_ids = np.arange(len(all_l1))

    for ax, data, label, color in zip(axes,
        [all_mae, all_psnr, all_ssim, all_l1],
        ['MAE (HU)', 'PSNR (dB)', 'SSIM', 'L1 Loss'],
        ['#e74c3c', '#2ecc71', '#3498db', '#f39c12']
    ):
        ax.bar(sample_ids, data, color=color, alpha=0.8)
        ax.axhline(np.mean(data), color='white', linestyle='--', linewidth=1.5, label=f'Mean: {np.mean(data):.3f}')
        ax.set_xlabel('Sample', fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.set_facecolor('#1a1a2e')

    fig.patch.set_facecolor('#0f0f23')
    for ax in axes:
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#333')

    plt.suptitle('Per-Sample Metrics Summary', fontsize=15, color='white', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'metrics_summary.png'),
                dpi=150, bbox_inches='tight', facecolor='#0f0f23')
    plt.close(fig)
    print(f'Saved metrics summary plot → {args.output_dir}/metrics_summary.png')


if __name__ == '__main__':
    main()
