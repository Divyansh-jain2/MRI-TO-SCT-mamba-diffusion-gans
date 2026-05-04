"""
Evaluation script for Mamba MRI-to-CT synthesis.
Runs sliding-window inference on test set and computes metrics.

Usage:
    python evaluate.py --data_dir /DATA/divyansh/brain_npy \
                       --checkpoint ./checkpoints/segmamba_best.pth \
                       --model segmamba \
                       --save_preds
"""

import os
import argparse
import numpy as np
import torch
from torch.nn import functional as F
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

from dataset import get_dataloaders
from models import get_model


# ─────────────────────────────────────────────
# Sliding window inference
# ─────────────────────────────────────────────
def sliding_window_inference(model, volume, patch_size=(64, 192, 192),
                              overlap=0.5, device='cuda'):
    """
    volume: tensor (1, 1, D, H, W)
    Returns: tensor (1, 1, D, H, W)
    """
    model.eval()
    _, _, D, H, W = volume.shape
    pD, pH, pW = patch_size

    # If volume fits in patch, just run directly
    if D <= pD and H <= pH and W <= pW:
        with torch.no_grad():
            return model(volume.to(device)).cpu()

    stride_d = max(1, int(pD * (1 - overlap)))
    stride_h = max(1, int(pH * (1 - overlap)))
    stride_w = max(1, int(pW * (1 - overlap)))

    pred_vol   = torch.zeros(1, 1, D, H, W)
    count_vol  = torch.zeros(1, 1, D, H, W)

    d_starts = list(range(0, max(D - pD, 0) + 1, stride_d))
    h_starts = list(range(0, max(H - pH, 0) + 1, stride_h))
    w_starts = list(range(0, max(W - pW, 0) + 1, stride_w))

    # Ensure last patch covers the end
    if d_starts[-1] + pD < D:
        d_starts.append(D - pD)
    if h_starts[-1] + pH < H:
        h_starts.append(H - pH)
    if w_starts[-1] + pW < W:
        w_starts.append(W - pW)

    with torch.no_grad():
        for ds in d_starts:
            for hs in h_starts:
                for ws in w_starts:
                    patch = volume[:, :, ds:ds+pD, hs:hs+pH, ws:ws+pW].to(device)
                    pred_patch = model(patch).cpu()
                    pred_vol[:, :,  ds:ds+pD, hs:hs+pH, ws:ws+pW] += pred_patch
                    count_vol[:, :, ds:ds+pD, hs:hs+pH, ws:ws+pW] += 1

    pred_vol = pred_vol / count_vol.clamp(min=1)
    return pred_vol


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────
def compute_metrics(pred_np, target_np):
    """
    pred_np, target_np: numpy arrays in [-1, 1]
    Returns dict of metrics.
    """
    mae  = np.mean(np.abs(pred_np - target_np))
    psnr = sk_psnr(target_np, pred_np, data_range=2.0)
    ssim = sk_ssim(target_np, pred_np, data_range=2.0,
                   win_size=7, channel_axis=None)
    return {'MAE': mae, 'PSNR': psnr, 'SSIM': ssim}


# ─────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────
def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    _, _, test_loader = get_dataloaders(
        base_dir=args.data_dir,
        patch_size=(64, 192, 192),
        batch_size=1,
        num_workers=2
    )

    model = get_model(args.model, base_ch=args.base_ch).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"[Loaded] {args.checkpoint} (epoch {ckpt.get('epoch','?')})")

    if args.save_preds:
        os.makedirs(args.pred_dir, exist_ok=True)

    all_metrics = []

    for i, (mri, ct, fnames) in enumerate(test_loader):
        fname = os.path.basename(fnames[0])
        print(f"[{i+1}/{len(test_loader)}] Processing: {fname}")

        pred = sliding_window_inference(
            model, mri, patch_size=(64, 192, 192),
            overlap=0.5, device=device
        )

        pred_np   = pred.squeeze().numpy()
        target_np = ct.squeeze().numpy()

        metrics = compute_metrics(pred_np, target_np)
        all_metrics.append(metrics)

        print(f"  MAE:  {metrics['MAE']:.4f}")
        print(f"  PSNR: {metrics['PSNR']:.2f} dB")
        print(f"  SSIM: {metrics['SSIM']:.4f}")

        if args.save_preds:
            save_path = os.path.join(args.pred_dir, fname.replace('.npy', '_pred.npy'))
            np.save(save_path, pred_np)

    # Summary
    print("\n" + "=" * 50)
    print(f"[Results] {args.model} on test set ({len(all_metrics)} cases)")
    print("=" * 50)
    for metric in ['MAE', 'PSNR', 'SSIM']:
        vals = [m[metric] for m in all_metrics]
        print(f"  {metric}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # Save results
    results_path = os.path.join(
        os.path.dirname(args.checkpoint),
        f'{args.model}_test_results.txt'
    )
    with open(results_path, 'w') as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n\n")
        for metric in ['MAE', 'PSNR', 'SSIM']:
            vals = [m[metric] for m in all_metrics]
            f.write(f"{metric}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}\n")
    print(f"\n[Saved] Results -> {results_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mamba MRI-to-CT Evaluation')
    parser.add_argument('--data_dir',   type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--model',      type=str, default='segmamba',
                        choices=['segmamba', 'umamba'])
    parser.add_argument('--base_ch',    type=int, default=32)
    parser.add_argument('--save_preds', action='store_true',
                        help='Save predicted CT volumes as .npy')
    parser.add_argument('--pred_dir',   type=str, default='./predictions')
    args = parser.parse_args()
    evaluate(args)