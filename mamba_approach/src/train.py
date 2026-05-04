"""
Training script for Mamba-driven MRI-to-CT synthesis.
Usage:
    python train.py --data_dir /DATA/divyansh/brain_npy --model segmamba
    python train.py --data_dir /DATA/divyansh/brain_npy --model umamba
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt  # <-- Added for visualization

from dataset import get_dataloaders
from models import get_model
from losses import CompoundLoss


# ─────────────────────────────────────────────
# Metrics & Visualization
# ─────────────────────────────────────────────
def compute_mae(pred, target):
    return torch.mean(torch.abs(pred - target)).item()

def compute_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float('inf')
    return 20 * np.log10(2.0) - 10 * np.log10(mse)  # data range = 2 ([-1,1])


def save_training_dashboard(history, mri, pred, target, epoch, save_dir):
    """Saves a unified dashboard showing metric graphs and image slices."""
    # Mamba processes 3D patches (Batch, Channel, D, H, W). Extract middle depth slice.
    d_mid = mri.shape[2] // 2
    
    mri_slice = mri[0, 0, d_mid, :, :].detach().cpu().numpy()
    pred_slice = pred[0, 0, d_mid, :, :].detach().cpu().numpy()
    target_slice = target[0, 0, d_mid, :, :].detach().cpu().numpy()
    
    fig = plt.figure(figsize=(15, 10))
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    # --- Top Row: Graphs ---
    # Train Loss Graph
    ax1 = plt.subplot2grid((2, 6), (0, 0), colspan=3)
    ax1.plot(epochs_range, history['train_loss'], label='Total Train Loss', color='#e74c3c', linewidth=2, marker='o', markersize=4)
    ax1.set_title('Training Loss over Time', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # Val PSNR Graph
    ax2 = plt.subplot2grid((2, 6), (0, 3), colspan=3)
    ax2.plot(epochs_range, history['val_psnr'], label='Val PSNR (dB)', color='#2ecc71', linewidth=2, marker='s', markersize=4)
    ax2.set_title('Validation PSNR over Time (Higher is better)', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('PSNR (dB)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    # --- Bottom Row: Images ---
    ax3 = plt.subplot2grid((2, 6), (1, 0), colspan=2)
    ax3.imshow(mri_slice, cmap='gray')
    ax3.set_title("Input MRI")
    ax3.axis('off')
    
    ax4 = plt.subplot2grid((2, 6), (1, 2), colspan=2)
    ax4.imshow(pred_slice, cmap='gray')
    ax4.set_title(f"Generated CT (Epoch {epoch})")
    ax4.axis('off')
    
    ax5 = plt.subplot2grid((2, 6), (1, 4), colspan=2)
    ax5.imshow(target_slice, cmap='gray')
    ax5.set_title("Target CT (Ground Truth)")
    ax5.axis('off')
    
    plt.tight_layout()
    out_path = os.path.join(save_dir, f"dashboard_epoch_{epoch:03d}.png")
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()


# ─────────────────────────────────────────────
# Polynomial LR schedule
# ─────────────────────────────────────────────
def poly_lr_lambda(epoch, max_epochs, power=0.9):
    return (1 - epoch / max_epochs) ** power


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create visuals folder for the dashboards
    visuals_dir = os.path.join(args.save_dir, 'visuals')
    os.makedirs(visuals_dir, exist_ok=True)

    # Data
    train_loader, val_loader, _ = get_dataloaders(
        base_dir=args.data_dir,
        patch_size=tuple(args.patch_size),  
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Model
    model = get_model(args.model, base_ch=args.base_ch).to(device)

    # Loss
    criterion = CompoundLoss(w1=1.0, w2=0.1, w3=0.1, device=device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda ep: poly_lr_lambda(ep, args.epochs)
    )

    best_val_mae = float('inf')
    train_log = []
    
    # Dictionary to track metrics for the graphs
    history = {
        'train_loss': [],
        'val_mae': [],
        'val_psnr': []
    }

    print(f"\n[Training] {args.model} | {args.epochs} epochs | "
          f"batch={args.batch_size} | lr={args.lr} | patch={args.patch_size}")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        model.train()
        epoch_losses = {'total': 0, 'wMAE': 0, 'SSIM': 0, 'AFP': 0}
        t0 = time.time()

        for step, (mri, ct, _) in enumerate(train_loader):
            mri = mri.to(device)
            ct  = ct.to(device)

            optimizer.zero_grad()
            pred = model(mri)
            loss, loss_dict = criterion(pred, ct, epoch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            for k in epoch_losses:
                epoch_losses[k] += loss_dict.get(k, 0)

        scheduler.step()

        n_steps = len(train_loader)
        avg_losses = {k: v / n_steps for k, v in epoch_losses.items()}
        elapsed = time.time() - t0

        # ── Validate ──
        model.eval()
        val_mae_list = []
        val_psnr_list = []
        saved_snapshot = False  # Ensure we only grab one image per epoch

        with torch.no_grad():
            for mri, ct, _ in val_loader:
                mri = mri.to(device)
                ct  = ct.to(device)
                pred = model(mri)
                
                val_mae_list.append(compute_mae(pred, ct))
                val_psnr_list.append(compute_psnr(pred, ct))
                
                # Capture the very first validation batch for the visual dashboard
                if not saved_snapshot:
                    snapshot_mri = mri
                    snapshot_pred = pred
                    snapshot_ct = ct
                    saved_snapshot = True

        val_mae  = np.mean(val_mae_list)
        val_psnr = np.mean(val_psnr_list)
        
        # Append to history for the graphs
        history['train_loss'].append(avg_losses['total'])
        history['val_mae'].append(val_mae)
        history['val_psnr'].append(val_psnr)
        
        # Generate the dashboard PNG
        save_training_dashboard(history, snapshot_mri, snapshot_pred, snapshot_ct, epoch, visuals_dir)

        log_str = (
            f"Epoch [{epoch:03d}/{args.epochs}] "
            f"Loss: {avg_losses['total']:.4f} "
            f"(wMAE:{avg_losses['wMAE']:.4f} "
            f"SSIM:{avg_losses['SSIM']:.4f} "
            f"AFP:{avg_losses['AFP']:.4f}) | "
            f"Val MAE: {val_mae:.4f} | Val PSNR: {val_psnr:.2f} dB | "
            f"Time: {elapsed:.1f}s"
        )
        print(log_str)
        train_log.append(log_str)

        # ── Save best model ──
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': val_mae,
                'val_psnr': val_psnr,
            }, os.path.join(args.save_dir, f'{args.model}_best.pth'))
            print(f"  --> Saved best model (Val MAE: {best_val_mae:.4f})")

        # ── Save checkpoint every 50 epochs ──
        if epoch % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.save_dir, f'{args.model}_epoch{epoch}.pth'))

    # Save training log
    with open(os.path.join(args.save_dir, f'{args.model}_train_log.txt'), 'w') as f:
        f.write('\n'.join(train_log))

    print(f"\n[Done] Best Val MAE: {best_val_mae:.4f}")
    print(f"[Done] Checkpoints saved to: {args.save_dir}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mamba MRI-to-CT Training')

    parser.add_argument('--data_dir',    type=str, required=True,
                        help='Path to brain_npy folder containing imagesTr/imagesVal/imagesTs')
    parser.add_argument('--model',       type=str, default='segmamba',
                        choices=['segmamba', 'umamba'],
                        help='Model architecture')
    parser.add_argument('--epochs',      type=int, default=500)
    parser.add_argument('--batch_size',  type=int, default=1)
    parser.add_argument('--lr',          type=float, default=5e-4)
    parser.add_argument('--base_ch',     type=int, default=32,
                        help='Base channel count (32 recommended for A5000 24GB)')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir',    type=str, default='./checkpoints',
                        help='Directory to save model checkpoints')
    
    # ADDED PATCH SIZE ARGUMENT HERE (Defaulting to the safe 14GB size)
    parser.add_argument('--patch_size', type=int, nargs=3, default=[32, 128, 128],
                        help='Patch size for training (D H W). Example: 32 128 128')

    args = parser.parse_args()
    train(args)