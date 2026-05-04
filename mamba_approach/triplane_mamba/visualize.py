"""
Visualization script for Mamba MRI-to-CT Synthesis.
Shows MRI | Predicted CT | Ground Truth CT side by side across axial, sagittal, coronal slices.

Usage:
    python visualize.py \
        --data_dir /DATA/divyansh/mc_ddpm_data/brain_npy \
        --checkpoint ./checkpoints/segmamba/segmamba_best.pth \
        --model segmamba \
        --num_cases 3
"""

import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from dataset import BrainMRICTDataset
from models import get_model


# ─────────────────────────────────────────────
# Sliding window inference (same as evaluate.py)
# ─────────────────────────────────────────────
def sliding_window_inference(model, volume, patch_size=(64, 192, 192),
                              overlap=0.5, device='cuda'):
    model.eval()
    _, _, D, H, W = volume.shape
    pD, pH, pW = patch_size

    if D <= pD and H <= pH and W <= pW:
        with torch.no_grad():
            return model(volume.to(device)).cpu()

    stride_d = max(1, int(pD * (1 - overlap)))
    stride_h = max(1, int(pH * (1 - overlap)))
    stride_w = max(1, int(pW * (1 - overlap)))

    pred_vol  = torch.zeros(1, 1, D, H, W)
    count_vol = torch.zeros(1, 1, D, H, W)

    d_starts = list(range(0, max(D - pD, 0) + 1, stride_d))
    h_starts = list(range(0, max(H - pH, 0) + 1, stride_h))
    w_starts = list(range(0, max(W - pW, 0) + 1, stride_w))

    if d_starts[-1] + pD < D: d_starts.append(D - pD)
    if h_starts[-1] + pH < H: h_starts.append(H - pH)
    if w_starts[-1] + pW < W: w_starts.append(W - pW)

    with torch.no_grad():
        for ds in d_starts:
            for hs in h_starts:
                for ws in w_starts:
                    patch = volume[:, :, ds:ds+pD, hs:hs+pH, ws:ws+pW].to(device)
                    pred_patch = model(patch).cpu()
                    pred_vol[:, :,  ds:ds+pD, hs:hs+pH, ws:ws+pW] += pred_patch
                    count_vol[:, :, ds:ds+pD, hs:hs+pH, ws:ws+pW] += 1

    return pred_vol / count_vol.clamp(min=1)


# ─────────────────────────────────────────────
# Plot one case: 3 planes x 3 columns (MRI | Pred CT | GT CT)
# ─────────────────────────────────────────────
def plot_case(mri, pred_ct, gt_ct, case_name, save_path):
    """
    mri, pred_ct, gt_ct: numpy arrays of shape (D, H, W)
    """
    D, H, W = mri.shape

    # Pick middle slices for each plane
    axial_idx    = D // 2
    coronal_idx  = H // 2
    sagittal_idx = W // 2

    # Slices for each view
    slices = {
        'Axial'    : (mri[axial_idx, :, :],    pred_ct[axial_idx, :, :],    gt_ct[axial_idx, :, :]),
        'Coronal'  : (mri[:, coronal_idx, :],   pred_ct[:, coronal_idx, :],  gt_ct[:, coronal_idx, :]),
        'Sagittal' : (mri[:, :, sagittal_idx],  pred_ct[:, :, sagittal_idx], gt_ct[:, :, sagittal_idx]),
    }

    fig = plt.figure(figsize=(15, 12))
    fig.suptitle(f'Case: {case_name}', fontsize=14, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(3, 4, figure=fig, wspace=0.05, hspace=0.3,
                           width_ratios=[1, 1, 1, 0.05])

    col_titles = ['MRI (Input)', 'Predicted CT', 'Ground Truth CT']
    row_titles = list(slices.keys())

    for row_idx, (plane, (s_mri, s_pred, s_gt)) in enumerate(slices.items()):

        # Compute difference map
        diff = np.abs(s_pred - s_gt)

        for col_idx, (img, title) in enumerate(zip(
            [s_mri, s_pred, s_gt],
            col_titles
        )):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            im = ax.imshow(img, cmap='gray', vmin=-1, vmax=1, aspect='auto')
            ax.axis('off')

            if row_idx == 0:
                ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
            if col_idx == 0:
                ax.set_ylabel(plane, fontsize=10, fontweight='bold', labelpad=8)
                ax.yaxis.set_label_position('left')
                ax.axis('on')
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

        # Colorbar for last column
        cax = fig.add_subplot(gs[row_idx, 3])
        plt.colorbar(im, cax=cax)
        cax.tick_params(labelsize=7)

    # Compute MAE for this case
    mae  = np.mean(np.abs(pred_ct - gt_ct))
    psnr_mse = np.mean((pred_ct - gt_ct) ** 2)
    psnr = 20 * np.log10(2.0) - 10 * np.log10(psnr_mse + 1e-8)

    fig.text(0.5, 0.01, f'MAE: {mae:.4f}  |  PSNR: {psnr:.2f} dB',
             ha='center', fontsize=11, color='dimgray')

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved -> {save_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    # Load test dataset
    test_dataset = BrainMRICTDataset(
        data_dir=os.path.join(args.data_dir, 'imagesTs'),
        patch_size=(64, 192, 192),
        mode='test'
    )

    # Load model
    model = get_model(args.model, base_ch=args.base_ch).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"[Loaded] {args.checkpoint}")

    num_cases = min(args.num_cases, len(test_dataset))
    print(f"[Visualizing] {num_cases} cases\n")

    for i in range(num_cases):
        mri_tensor, ct_tensor, fpath = test_dataset[i]
        case_name = os.path.basename(fpath).replace('.npy', '')
        print(f"[{i+1}/{num_cases}] {case_name}")

        # Run inference
        mri_input = mri_tensor.unsqueeze(0)  # (1, 1, D, H, W)
        pred = sliding_window_inference(
            model, mri_input,
            patch_size=(64, 192, 192),
            overlap=0.5, device=device
        )

        mri_np   = mri_tensor.squeeze().numpy()   # (D, H, W)
        pred_np  = pred.squeeze().numpy()          # (D, H, W)
        gt_np    = ct_tensor.squeeze().numpy()     # (D, H, W)

        save_path = os.path.join(args.out_dir, f'{case_name}_comparison.png')
        plot_case(mri_np, pred_np, gt_np, case_name, save_path)

    print(f"\n[Done] All visualizations saved to: {args.out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',   type=str,
                        default='/DATA/divyansh/mc_ddpm_data/brain_npy')
    parser.add_argument('--checkpoint', type=str,
                        default='./checkpoints/segmamba/segmamba_best.pth')
    parser.add_argument('--model',      type=str, default='segmamba',
                        choices=['segmamba', 'umamba'])
    parser.add_argument('--base_ch',    type=int, default=32)
    parser.add_argument('--num_cases',  type=int, default=3,
                        help='Number of test cases to visualize')
    parser.add_argument('--out_dir',    type=str, default='./visualizations')
    args = parser.parse_args()
    main(args)