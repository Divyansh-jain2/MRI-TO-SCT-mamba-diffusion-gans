"""
Evaluation script for TriPlaneMamba-UNet MRI-to-CT synthesis.
Sliding-window inference + optional TTA.

Usage:
    python evaluate.py --data_dir /DATA/divyansh/mc_ddpm_data/brain_npy \
                       --checkpoint ./checkpoints_triplane/triplane_best.pth

    # With test-time augmentation:
    python evaluate.py --data_dir /DATA/divyansh/mc_ddpm_data/brain_npy \
                       --checkpoint ./checkpoints_triplane/triplane_best.pth --tta
"""

import os
import argparse
import numpy as np
import torch
from torch.cuda.amp import autocast
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

from dataset import get_dataloaders
from models import get_model


def sliding_window_inference(model, volume, patch_size=(32, 128, 128),
                              overlap=0.5, device='cuda'):
    """Sliding-window inference on full volume."""
    model.eval()
    _, _, D, H, W = volume.shape
    pD, pH, pW = patch_size

    if D <= pD and H <= pH and W <= pW:
        with torch.no_grad(), autocast(enabled=True):
            out = model(volume.to(device))
            if isinstance(out, tuple):
                out = out[0]
            return out.float().cpu()

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
                    with autocast(enabled=True):
                        p = model(patch)
                        if isinstance(p, tuple):
                            p = p[0]
                    pred_vol[:, :,  ds:ds+pD, hs:hs+pH, ws:ws+pW] += p.float().cpu()
                    count_vol[:, :, ds:ds+pD, hs:hs+pH, ws:ws+pW] += 1

    return pred_vol / count_vol.clamp(min=1)


def tta_inference(model, volume, patch_size=(32, 128, 128),
                  overlap=0.5, device='cuda'):
    """Test-time augmentation: average 4 flipped predictions."""
    preds = [sliding_window_inference(model, volume, patch_size, overlap, device)]
    for flip_dim in [2, 3, 4]:
        flipped = torch.flip(volume, dims=[flip_dim])
        p = sliding_window_inference(model, flipped, patch_size, overlap, device)
        preds.append(torch.flip(p, dims=[flip_dim]))
    return torch.stack(preds).mean(dim=0)


def compute_metrics(pred_np, target_np):
    mae  = np.mean(np.abs(pred_np - target_np))
    rmse = np.sqrt(np.mean((pred_np - target_np) ** 2))
    psnr = sk_psnr(target_np, pred_np, data_range=2.0)
    ssim = sk_ssim(target_np, pred_np, data_range=2.0,
                   win_size=7, channel_axis=None)
    return {'MAE': mae, 'RMSE': rmse, 'PSNR': psnr, 'SSIM': ssim}


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    _, _, test_loader = get_dataloaders(
        base_dir=args.data_dir,
        patch_size=(32, 128, 128),
        batch_size=1,
        num_workers=4
    )

    # Checkpoint was saved with deep_supervision=True, so build the model identically.
    # The aux heads are gated by `self.training`, so they are completely inert during eval.
    model = get_model('triplanemamba', base_ch=args.base_ch,
                      deep_supervision=True, use_checkpoint=False).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"[Loaded] {args.checkpoint} (epoch {ckpt.get('epoch','?')})")
    else:
        model.load_state_dict(ckpt)
        print(f"[Loaded] {args.checkpoint}")

    model.eval()

    if args.save_preds:
        os.makedirs(args.pred_dir, exist_ok=True)

    all_metrics = []
    infer_fn = tta_inference if args.tta else sliding_window_inference

    for i, (mri, ct, fnames) in enumerate(test_loader):
        fname = os.path.basename(fnames[0])
        print(f"[{i+1}/{len(test_loader)}] {fname}", end=" ")

        pred = infer_fn(model, mri, patch_size=(32, 128, 128),
                        overlap=0.5, device=device)
        pred_np   = pred.squeeze().numpy()
        target_np = ct.squeeze().numpy()

        m = compute_metrics(pred_np, target_np)
        all_metrics.append(m)
        print(f"MAE:{m['MAE']:.4f} PSNR:{m['PSNR']:.2f} SSIM:{m['SSIM']:.4f}")

        if args.save_preds:
            np.save(os.path.join(args.pred_dir,
                    fname.replace('.npy', '_pred.npy')), pred_np)

    print("\n" + "=" * 60)
    print(f"[Results] TriPlaneMamba-UNet | TTA={'ON' if args.tta else 'OFF'} | "
          f"{len(all_metrics)} test cases")
    print("=" * 60)
    for metric in ['MAE', 'RMSE', 'PSNR', 'SSIM']:
        vals = [m[metric] for m in all_metrics]
        print(f"  {metric}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    results_path = os.path.join(
        os.path.dirname(args.checkpoint), 'triplane_test_results.txt')
    with open(results_path, 'w') as f:
        f.write(f"Model: TriPlaneMamba-UNet (base_ch={args.base_ch})\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"TTA: {args.tta}\n\n")
        for metric in ['MAE', 'RMSE', 'PSNR', 'SSIM']:
            vals = [m[metric] for m in all_metrics]
            f.write(f"{metric}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}\n")
    print(f"\n[Saved] Results -> {results_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TriMamba MRI-to-CT Eval')
    parser.add_argument('--data_dir',   type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--base_ch',    type=int, default=32)
    parser.add_argument('--save_preds', action='store_true')
    parser.add_argument('--pred_dir',   type=str, default='./predictions_trimamba')
    parser.add_argument('--tta',        action='store_true',
                        help='Test-time augmentation (4 flips)')
    args = parser.parse_args()
    evaluate(args)