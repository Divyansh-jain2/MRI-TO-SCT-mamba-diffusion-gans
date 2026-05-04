"""
Training script for TriMamba-UNet MRI-to-CT synthesis.

Features:
  - Cosine annealing LR with warm restarts (T_0=100)
  - Deep supervision support
  - Mixed precision training (fp16)
  - NaN/OOM guards
  - Full checkpoint resume (model + optimizer + scheduler + scaler + history)
  - Visual dashboards every 5 epochs

Usage:
    python train.py --data_dir /DATA/divyansh/mc_ddpm_data/brain_npy

    # Resume from checkpoint:
    python train.py --data_dir /DATA/divyansh/mc_ddpm_data/brain_npy \
                    --resume ./checkpoints_trimamba/trimamba_epoch150.pth
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.amp import GradScaler, autocast
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import get_dataloaders
from models import get_model
from losses import CompoundLossV2


# ─────────────────────────────────────────────
# Metrics & Visualization
# ─────────────────────────────────────────────
def compute_mae(pred, target):
    return torch.mean(torch.abs(pred.float() - target.float())).item()


def compute_psnr(pred, target):
    mse = torch.mean((pred.float() - target.float()) ** 2).item()
    if mse == 0:
        return float('inf')
    return 20 * np.log10(2.0) - 10 * np.log10(mse)


def save_training_dashboard(history, mri, pred, target, epoch, save_dir):
    """Saves a dashboard: loss/PSNR graphs + image slices."""
    d_mid = mri.shape[2] // 2

    mri_slice = mri[0, 0, d_mid].detach().cpu().float().numpy()
    pred_slice = pred[0, 0, d_mid].detach().cpu().float().numpy()
    target_slice = target[0, 0, d_mid].detach().cpu().float().numpy()
    diff_slice = np.abs(pred_slice - target_slice)

    fig = plt.figure(figsize=(18, 10))
    epochs_range = range(1, len(history['train_loss']) + 1)

    ax1 = plt.subplot2grid((2, 8), (0, 0), colspan=4)
    ax1.plot(epochs_range, history['train_loss'], label='Train Loss',
             color='#e74c3c', linewidth=2, marker='o', markersize=2)
    ax1.set_title('Training Loss', fontsize=14)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle='--', alpha=0.6); ax1.legend()

    ax2 = plt.subplot2grid((2, 8), (0, 4), colspan=4)
    ax2.plot(epochs_range, history['val_psnr'], label='Val PSNR (dB)',
             color='#2ecc71', linewidth=2, marker='s', markersize=2)
    ax2.set_title('Validation PSNR', fontsize=14)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('PSNR (dB)')
    ax2.grid(True, linestyle='--', alpha=0.6); ax2.legend()

    for i, (img, title) in enumerate([
        (mri_slice, "Input MRI"), (pred_slice, f"Predicted CT (ep {epoch})"),
        (target_slice, "Target CT")
    ]):
        ax = plt.subplot2grid((2, 8), (1, i*2), colspan=2)
        ax.imshow(img, cmap='gray'); ax.set_title(title); ax.axis('off')

    ax6 = plt.subplot2grid((2, 8), (1, 6), colspan=2)
    im = ax6.imshow(diff_slice, cmap='hot', vmin=0, vmax=0.3)
    ax6.set_title("Error Map"); ax6.axis('off')
    plt.colorbar(im, ax=ax6, fraction=0.046)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"dashboard_epoch_{epoch:03d}.png"),
                bbox_inches='tight', dpi=150)
    plt.close()


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    os.makedirs(args.save_dir, exist_ok=True)
    visuals_dir = os.path.join(args.save_dir, 'visuals')
    os.makedirs(visuals_dir, exist_ok=True)

    # ── Data ──
    train_loader, val_loader, _ = get_dataloaders(
        base_dir=args.data_dir,
        patch_size=tuple(args.patch_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # ── Model ──
    model = get_model('trimamba', base_ch=args.base_ch,
                      deep_supervision=True).to(device)

    # ── Loss ──
    criterion = CompoundLossV2(
        w_mae=1.0, w_ssim=0.2, w_grad=0.05,
        w_ffl=0.1, w_ds2=0.4, w_ds3=0.2
    )

    # ── Optimizer ──
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # ── LR Schedule ──
    # T_0=100 so the LR restart at epoch 100 does NOT coincide with
    # the loss stage change at epoch 50 (which caused NaN last time)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2)

    # ── AMP (fp16 — bf16 is not supported by torch.fft on all GPUs) ──
    use_amp = torch.cuda.is_available()
    scaler = GradScaler('cuda', enabled=use_amp)

    best_val_psnr = 0.0
    best_val_mae = float('inf')
    start_epoch = 1
    history = {'train_loss': [], 'val_mae': [], 'val_psnr': []}
    train_log = []

    # ── Resume from checkpoint ──
    if args.resume:
        print(f"\n[Resume] Loading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        if 'best_val_psnr' in ckpt:
            best_val_psnr = ckpt['best_val_psnr']
        if 'best_val_mae' in ckpt:
            best_val_mae = ckpt['best_val_mae']
        if 'history' in ckpt:
            history = ckpt['history']
        start_epoch = ckpt['epoch'] + 1
        print(f"[Resume] Continuing from epoch {start_epoch} | "
              f"Best PSNR so far: {best_val_psnr:.2f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        del ckpt
        torch.cuda.empty_cache()

    print(f"\n[Config] epochs={args.epochs} | batch={args.batch_size} | "
          f"lr={args.lr} | patch={args.patch_size} | "
          f"AMP={'fp16' if use_amp else 'OFF'}")
    print("=" * 70)

    for epoch in range(start_epoch, args.epochs + 1):
        # ── Train ──
        model.train()
        epoch_losses = {}
        t0 = time.time()
        valid_steps = 0

        for step, (mri, ct, _) in enumerate(train_loader):
            mri = mri.to(device, non_blocking=True)
            ct  = ct.to(device, non_blocking=True)

            try:
                optimizer.zero_grad(set_to_none=True)

                with autocast('cuda', enabled=use_amp):
                    output = model(mri)

                if isinstance(output, tuple):
                    pred, aux2, aux3 = output
                    pred, aux2, aux3 = pred.float(), aux2.float(), aux3.float()
                    loss, loss_dict = criterion(
                        pred, ct.float(), epoch, aux_preds=(aux2, aux3))
                else:
                    pred = output.float()
                    loss, loss_dict = criterion(pred, ct.float(), epoch)

                # NaN guard
                if not torch.isfinite(loss):
                    print(f"  [WARN] NaN/Inf loss at ep {epoch} step {step}, skipping")
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                valid_steps += 1
                for k, v in loss_dict.items():
                    epoch_losses[k] = epoch_losses.get(k, 0.0) + float(v)

            except RuntimeError as err:
                if 'out of memory' in str(err).lower():
                    print(f"  [WARN] OOM at ep {epoch} step {step}, skipping")
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    continue
                raise

        scheduler.step()

        denom = max(valid_steps, 1)
        avg_losses = {k: v / denom for k, v in epoch_losses.items()}
        elapsed = time.time() - t0

        # ── Validate ──
        model.eval()
        val_mae_list, val_psnr_list = [], []
        snapshot_data = None

        with torch.no_grad():
            for mri, ct, _ in val_loader:
                mri = mri.to(device, non_blocking=True)
                ct  = ct.to(device, non_blocking=True)

                with autocast('cuda', enabled=use_amp):
                    pred = model(mri)
                    if isinstance(pred, tuple):
                        pred = pred[0]

                if torch.isfinite(pred).all():
                    val_mae_list.append(compute_mae(pred, ct))
                    val_psnr_list.append(compute_psnr(pred, ct))

                    if snapshot_data is None:
                        snapshot_data = (mri, pred, ct)

        val_mae = float(np.mean(val_mae_list)) if val_mae_list else float('nan')
        val_psnr = float(np.mean(val_psnr_list)) if val_psnr_list else float('nan')

        history['train_loss'].append(avg_losses.get('total', 0))
        history['val_mae'].append(val_mae)
        history['val_psnr'].append(val_psnr)

        # ── Dashboard ──
        if (epoch % 5 == 0 or epoch <= 5) and snapshot_data is not None:
            save_training_dashboard(history, *snapshot_data, epoch, visuals_dir)

        # ── Log ──
        parts = " ".join([f"{k}:{v:.4f}" for k, v in avg_losses.items()
                          if k != 'total'])
        log_str = (f"Epoch [{epoch:03d}/{args.epochs}] "
                   f"Loss: {avg_losses.get('total',0):.4f} ({parts}) | "
                   f"Val MAE: {val_mae:.4f} PSNR: {val_psnr:.2f} | "
                   f"LR: {optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s")
        print(log_str)
        train_log.append(log_str)

        # ── Full checkpoint dict (reused for best + periodic saves) ──
        def _make_ckpt():
            return {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_val_psnr': best_val_psnr,
                'best_val_mae': best_val_mae,
                'val_mae': val_mae, 'val_psnr': val_psnr,
                'history': history,
            }

        # ── Save best (by PSNR) ──
        if np.isfinite(val_psnr) and val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            best_val_mae = val_mae
            torch.save(_make_ckpt(), os.path.join(args.save_dir, 'trimamba_best.pth'))
            print(f"  --> Saved best (PSNR: {val_psnr:.2f}, MAE: {val_mae:.4f})")

        # ── Periodic checkpoint (every 50 epochs) ──
        if epoch % 50 == 0:
            torch.save(_make_ckpt(),
                       os.path.join(args.save_dir, f'trimamba_epoch{epoch}.pth'))

    # Save log
    with open(os.path.join(args.save_dir, 'training_log.txt'), 'w') as f:
        f.write('\n'.join(train_log))

    print(f"\n[Done] Best PSNR: {best_val_psnr:.2f} | Best MAE: {best_val_mae:.4f}")
    print(f"[Done] Checkpoints in: {args.save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TriMamba MRI-to-CT Training')

    parser.add_argument('--data_dir',    type=str, required=True)
    parser.add_argument('--epochs',      type=int, default=500)
    parser.add_argument('--batch_size',  type=int, default=1)
    parser.add_argument('--lr',          type=float, default=2e-4)
    parser.add_argument('--base_ch',     type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir',    type=str, default='./checkpoints_trimamba')
    parser.add_argument('--patch_size',  type=int, nargs=3, default=[32, 128, 128])
    parser.add_argument('--resume',      type=str, default=None,
                        help='Path to checkpoint to resume from')

    args = parser.parse_args()
    train(args)