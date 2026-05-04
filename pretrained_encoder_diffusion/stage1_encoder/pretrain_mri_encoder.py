"""
Pretrain MRI Semantic Encoder using dual-task learning.

Dual tasks:
  1. MRI Reconstruction: forces encoder to learn spatially precise features
  2. CT Prediction:      forces encoder to learn CT-discriminative MRI features
                         (bone density, tissue boundaries, HU-relevant structures)

Combined loss:
  L = 0.5 * L1(CT_pred, CT_gt) + 0.3 * L1(MRI_recon, MRI_gt) + 0.2 * SSIM(MRI_recon, MRI_gt)

After 300 epochs, only the encoder weights are saved and loaded into main_hybrid.py.

Usage:
  python pretrain_mri_encoder.py
  # Resumes from checkpoint if one exists
"""

import os
import time
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from natsort import natsorted

from network.mri_encoder import MRIAutoencoder

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
DATA_ROOT  = '/home/teaching/Desktop/Munish_Synthetic_CT/mc_ddpm_data/brain_npy'
SAVE_DIR   = './checkpoints_mri_encoder'
VIS_DIR    = './mri_encoder_vis'
LOG_DIR    = './runs_mri_encoder'
CT_CLIP    = (-1024, 1650)

EPOCHS          = 500
BATCH_SIZE      = 4          # Larger than diffusion training — no 50-step inference here
LR              = 1e-4       # 3e-4 caused NaN via fp16 SSIM overflow — safer value
WEIGHT_DECAY    = 1e-5
PATIENCE        = 80         # Early stopping patience (epochs without SSIM improvement)
VAL_EVERY       = 10         # Validate every N epochs
VIS_EVERY       = 20         # Visualize reconstructions every N epochs

# Model config — must match what main_hybrid.py expects
ENC_CHANNELS     = (64, 128, 192, 256)
GLOBAL_DIM       = 256
WINDOW_SIZE      = (4, 4, 4)
NUM_HEADS        = (4, 4, 8, 8)
POOL_KERNEL      = (2, 2, 1)   # XY↓2, Z stays (patches are thin in Z)
DROPOUT          = 0.1

# Loss weights: CT task gets highest weight (task-specific pretraining)
W_CT   = 0.50
W_MRI  = 0.30
W_SSIM = 0.20

PATCH_SIZE = (64, 64, 4)   # Same patch size as main_hybrid.py

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ═══════════════════════════════════════════════════════════════════════════════
# SSIM Loss (computed on 2D slices to avoid 3D kernel complexity)
# ═══════════════════════════════════════════════════════════════════════════════
def ssim_loss(pred, target, window_size=11, reduction='mean'):
    """
    Structural Similarity loss on each 2D axial slice.
    pred, target: [B, 1, H, W, D] — loop over D slices.
    Always computed in float32 for numerical stability regardless of AMP dtype.
    """
    # Always use float32 — avoids HalfTensor/FloatTensor mismatch under AMP
    pred   = pred.float()
    target = target.float()
    B, C, H, W, D = pred.shape
    losses = []
    for d in range(D):
        p_slice = pred[:, :, :, :, d]    # [B, 1, H, W]
        t_slice = target[:, :, :, :, d]  # [B, 1, H, W]
        losses.append(_ssim2d(p_slice, t_slice, window_size))
    ssim_val = torch.stack(losses).mean()
    return 1.0 - ssim_val   # Loss = 1 - SSIM (minimise)


def _gaussian_kernel(window_size, sigma=1.5):
    x = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    gauss /= gauss.sum()
    kernel2d = gauss.unsqueeze(1) * gauss.unsqueeze(0)
    return kernel2d.unsqueeze(0).unsqueeze(0)   # [1, 1, ws, ws]


def _ssim2d(pred, target, window_size=11):
    """SSIM on [B, 1, H, W] tensors."""
    kernel = _gaussian_kernel(window_size).to(pred.device)  # pred is always float32 here
    pad = window_size // 2

    mu1 = F.conv2d(pred,   kernel, padding=pad)
    mu2 = F.conv2d(target, kernel, padding=pad)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred,     kernel, padding=pad) - mu1_sq
    sigma2_sq = F.conv2d(target * target, kernel, padding=pad) - mu2_sq
    sigma12   = F.conv2d(pred * target,   kernel, padding=pad) - mu1_mu2

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    return (num / den).mean()


# ═══════════════════════════════════════════════════════════════════════════════
# PSNR Metric
# ═══════════════════════════════════════════════════════════════════════════════
def psnr(pred, target, data_range=2.0):
    """PSNR in dB. data_range=2 for [-1,1] normalised volumes."""
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return torch.tensor(100.0)
    return 10 * torch.log10(data_range ** 2 / mse)


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════════
class MRICTDataset(Dataset):
    """
    Loads preprocessed .npy files containing [MRI, CT] volumes.
    During pretraining, returns random 3D patches of size PATCH_SIZE.
    """
    def __init__(self, data_dir, patch_size=PATCH_SIZE, patches_per_vol=4, train=True):
        self.files = natsorted(glob.glob(data_dir + "*.npy"), key=lambda y: y.lower())
        if not self.files:
            raise FileNotFoundError(f"No .npy files found in {data_dir}")
        self.patch_size = patch_size
        self.patches_per_vol = patches_per_vol
        self.train = train
        print(f"  Found {len(self.files)} volumes [{'train' if train else 'val'}]")

    def __len__(self):
        return len(self.files) * self.patches_per_vol

    def __getitem__(self, idx):
        vol_idx  = idx // self.patches_per_vol
        data     = np.load(self.files[vol_idx])   # [2, H, W, D]
        mri_vol  = data[0]   # [-1, 1]
        ct_vol   = data[1]   # [-1, 1]

        pH, pW, pD = self.patch_size

        if self.train:
            # Random crop
            H, W, D = mri_vol.shape
            sh = np.random.randint(0, max(1, H - pH))
            sw = np.random.randint(0, max(1, W - pW))
            sd = np.random.randint(0, max(1, D - pD))
        else:
            # Centre crop for validation
            H, W, D = mri_vol.shape
            sh = max(0, (H - pH) // 2)
            sw = max(0, (W - pW) // 2)
            sd = max(0, (D - pD) // 2)

        mri_patch = mri_vol[sh:sh+pH, sw:sw+pW, sd:sd+pD]
        ct_patch  = ct_vol [sh:sh+pH, sw:sw+pW, sd:sd+pD]

        # Pad if needed
        mri_patch = _pad_to(mri_patch, pH, pW, pD)
        ct_patch  = _pad_to(ct_patch,  pH, pW, pD)

        mri_t = torch.from_numpy(mri_patch[np.newaxis]).float()  # [1, H, W, D]
        ct_t  = torch.from_numpy(ct_patch [np.newaxis]).float()
        return mri_t, ct_t


def _pad_to(vol, H, W, D):
    """Zero-pad to at least target size."""
    h, w, d = vol.shape
    ph = max(0, H - h); pw = max(0, W - w); pd = max(0, D - d)
    if ph or pw or pd:
        vol = np.pad(vol, ((0, ph), (0, pw), (0, pd)), mode='constant')
    return vol[:H, :W, :D]


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD MODEL
# ═══════════════════════════════════════════════════════════════════════════════
model = MRIAutoencoder(
    enc_channels=ENC_CHANNELS,
    global_dim=GLOBAL_DIM,
    window_size=WINDOW_SIZE,
    num_heads=NUM_HEADS,
    pool_kernel=POOL_KERNEL,
    dropout=DROPOUT,
).to(device)

total_params   = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n{'='*60}")
print(f"MRI AUTOENCODER SUMMARY")
print(f"{'='*60}")
print(f"Total parameters:  {total_params:>12,}")
print(f"Trainable:         {trainable_params:>12,}")
print(f"Encoder channels:  {ENC_CHANNELS}")
print(f"Loss weights:      CT={W_CT}, MRI={W_MRI}, SSIM={W_SSIM}")
print(f"{'='*60}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# DATALOADERS
# ═══════════════════════════════════════════════════════════════════════════════
train_ds  = MRICTDataset(DATA_ROOT + '/imagesTr/', patches_per_vol=4, train=True)
val_ds    = MRICTDataset(DATA_ROOT + '/imagesVal/', patches_per_vol=1, train=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                          num_workers=2, pin_memory=True)

print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")


# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMISER + SCHEDULER
# ═══════════════════════════════════════════════════════════════════════════════
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)
scaler    = torch.cuda.amp.GradScaler()

l1_loss = nn.L1Loss()


# ═══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT RESUME
# ═══════════════════════════════════════════════════════════════════════════════
CKPT_PATH = os.path.join(SAVE_DIR, 'pretrain_checkpoint.pt')
BEST_PATH = os.path.join(SAVE_DIR, 'best_mri_encoder.pt')

start_epoch   = 0
best_val_ssim = -1.0
no_improve    = 0

if os.path.exists(CKPT_PATH):
    print(f"Resuming from checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    start_epoch   = ckpt['epoch'] + 1
    best_val_ssim = ckpt.get('best_val_ssim', -1.0)
    no_improve    = ckpt.get('no_improve', 0)
    print(f"Resumed at epoch {start_epoch}, best SSIM: {best_val_ssim:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# TENSORBOARD
# ═══════════════════════════════════════════════════════════════════════════════
run_name = f"mri_encoder_{time.strftime('%Y%m%d_%H%M%S')}"
writer   = SummaryWriter(log_dir=os.path.join(LOG_DIR, run_name))
print(f"TensorBoard: tensorboard --logdir {LOG_DIR} --port 6007")


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALISE RECONSTRUCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
def visualise(epoch, mri_in, mri_recon, ct_gt, ct_pred):
    """Save side-by-side comparison PNG for this epoch."""
    mid = mri_in.shape[-1] // 2   # Middle axial slice

    def to_np(t):
        arr = t[0, 0, :, :, mid].detach().float().cpu().numpy()  # .float() for AMP float16 compat
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].imshow(to_np(mri_in),    cmap='gray'); axes[0, 0].set_title('MRI Input')
    axes[0, 1].imshow(to_np(mri_recon), cmap='gray'); axes[0, 1].set_title('MRI Reconstructed')
    axes[1, 0].imshow(to_np(ct_gt),     cmap='gray'); axes[1, 0].set_title('CT Ground Truth')
    axes[1, 1].imshow(to_np(ct_pred),   cmap='gray'); axes[1, 1].set_title('CT Predicted')

    for a in axes.flat:
        a.axis('off')

    plt.suptitle(f'Epoch {epoch+1} — MRI Reconstruction + CT Prediction', fontsize=14)
    plt.tight_layout()
    out_path = os.path.join(VIS_DIR, f'epoch_{epoch+1:04d}.png')
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved visualisation → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# TRAIN ONE EPOCH
# ═══════════════════════════════════════════════════════════════════════════════
def train_epoch(epoch):
    model.train()
    ct_losses, mri_losses, ssim_losses, total_losses = [], [], [], []
    nan_batches = 0
    t0 = time.time()

    for mri, ct in train_loader:
        mri = mri.to(device)
        ct  = ct.to(device)

        optimizer.zero_grad()

        # Forward pass in AMP for speed
        with torch.cuda.amp.autocast():
            mri_recon_fp16, ct_pred_fp16, _ = model(mri)

        # Cast to float32 BEFORE loss — SSIM Gaussian conv overflows in fp16 → NaN
        mri_recon = mri_recon_fp16.float()
        ct_pred   = ct_pred_fp16.float()
        mri_f32   = mri.float()
        ct_f32    = ct.float()

        l_ct   = l1_loss(ct_pred,   ct_f32)
        l_mri  = l1_loss(mri_recon, mri_f32)
        l_ssim = ssim_loss(mri_recon, mri_f32)   # already float32 — safe
        loss   = W_CT * l_ct + W_MRI * l_mri + W_SSIM * l_ssim

        # Skip NaN batches cleanly (don't corrupt GradScaler state)
        if not torch.isfinite(loss):
            nan_batches += 1
            scaler.update()   # keep GradScaler state valid
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                       error_if_nonfinite=False)
        scaler.step(optimizer)
        scaler.update()

        ct_losses.append(l_ct.item())
        mri_losses.append(l_mri.item())
        ssim_losses.append(l_ssim.item())
        total_losses.append(loss.item())

    elapsed = time.time() - t0
    if nan_batches:
        print(f"  [WARNING] {nan_batches} NaN batches skipped this epoch")
    return {
        'total':   np.mean(total_losses) if total_losses else float('nan'),
        'ct':      np.mean(ct_losses)    if ct_losses    else float('nan'),
        'mri':     np.mean(mri_losses)   if mri_losses   else float('nan'),
        'ssim':    np.mean(ssim_losses)  if ssim_losses  else float('nan'),
        'time':    elapsed,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATE
# ═══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def validate(epoch):
    model.eval()
    psnr_mri_list, ssim_mri_list = [], []
    psnr_ct_list,  ssim_ct_list  = [], []

    mri_sample = ct_sample = mri_recon_sample = ct_pred_sample = None

    for i, (mri, ct) in enumerate(val_loader):
        mri = mri.to(device)
        ct  = ct.to(device)

        with torch.cuda.amp.autocast():
            mri_recon, ct_pred, _ = model(mri)

        psnr_mri_list.append(psnr(mri_recon, mri).item())
        psnr_ct_list.append( psnr(ct_pred,   ct ).item())
        ssim_mri_list.append(1.0 - ssim_loss(mri_recon, mri).item())
        ssim_ct_list.append( 1.0 - ssim_loss(ct_pred,   ct ).item())

        if i == 0:
            mri_sample       = mri
            ct_sample        = ct
            mri_recon_sample = mri_recon
            ct_pred_sample   = ct_pred

    metrics = {
        'psnr_mri':  np.mean(psnr_mri_list),
        'psnr_ct':   np.mean(psnr_ct_list),
        'ssim_mri':  np.mean(ssim_mri_list),
        'ssim_ct':   np.mean(ssim_ct_list),
        'samples':   (mri_sample, ct_sample, mri_recon_sample, ct_pred_sample),
    }
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"PRETRAINING MRI ENCODER — {EPOCHS} epochs")
print(f"CT task weight: {W_CT} | MRI task weight: {W_MRI} | SSIM weight: {W_SSIM}")
print(f"{'='*60}\n")

global_step = 0
total_start = time.time()

for epoch in range(start_epoch, EPOCHS):

    # ── Train ─────────────────────────────────────────────────────────────────
    stats = train_epoch(epoch)
    scheduler.step()
    lr = scheduler.get_last_lr()[0]

    writer.add_scalar('Train/Total_Loss', stats['total'], epoch)
    writer.add_scalar('Train/CT_Loss',    stats['ct'],    epoch)
    writer.add_scalar('Train/MRI_Loss',   stats['mri'],   epoch)
    writer.add_scalar('Train/SSIM_Loss',  stats['ssim'],  epoch)
    writer.add_scalar('Train/LR',         lr,             epoch)

    elapsed_total = (time.time() - total_start) / 60
    eta_min = elapsed_total / (epoch - start_epoch + 1) * (EPOCHS - epoch - 1)

    print(
        f"[{epoch+1:03d}/{EPOCHS}] "
        f"Loss: {stats['total']:.4f}  "
        f"(CT={stats['ct']:.4f} MRI={stats['mri']:.4f} SSIM={stats['ssim']:.4f})  "
        f"LR={lr:.1e}  "
        f"Time: {stats['time']:.1f}s  "
        f"ETA: {eta_min:.0f}min"
    )

    # ── Validate ──────────────────────────────────────────────────────────────
    if (epoch + 1) % VAL_EVERY == 0 or epoch == start_epoch:
        metrics = validate(epoch)
        mri_s, ct_s, mr_r, ct_p = metrics['samples']

        writer.add_scalar('Val/PSNR_MRI', metrics['psnr_mri'], epoch)
        writer.add_scalar('Val/PSNR_CT',  metrics['psnr_ct'],  epoch)
        writer.add_scalar('Val/SSIM_MRI', metrics['ssim_mri'], epoch)
        writer.add_scalar('Val/SSIM_CT',  metrics['ssim_ct'],  epoch)

        print(
            f"  ── VAL ── "
            f"PSNR_MRI: {metrics['psnr_mri']:.2f}dB  "
            f"SSIM_MRI: {metrics['ssim_mri']:.4f}  "
            f"PSNR_CT: {metrics['psnr_ct']:.2f}dB  "
            f"SSIM_CT: {metrics['ssim_ct']:.4f}"
        )

        # Combined metric for model selection: weight CT SSIM higher
        combined_ssim = 0.6 * metrics['ssim_ct'] + 0.4 * metrics['ssim_mri']

        if combined_ssim > best_val_ssim:
            best_val_ssim = combined_ssim
            no_improve    = 0
            # Save encoder weights only (not decoder — not needed for hybrid model)
            torch.save(model.encoder.state_dict(), BEST_PATH)
            print(f"  ★ New best! Combined SSIM={combined_ssim:.4f} → saved encoder to {BEST_PATH}")
        else:
            no_improve += VAL_EVERY
            print(f"  No improvement for {no_improve} epochs (patience={PATIENCE})")

        # Log validation images to TensorBoard
        def norm01(t):
            mid = t.shape[-1] // 2
            s = t[0, 0, :, :, mid].detach().cpu()
            return ((s - s.min()) / (s.max() - s.min() + 1e-8)).unsqueeze(0)

        writer.add_image('Val/MRI_input',    norm01(mri_s), epoch)
        writer.add_image('Val/MRI_recon',    norm01(mr_r),  epoch)
        writer.add_image('Val/CT_gt',        norm01(ct_s),  epoch)
        writer.add_image('Val/CT_predicted', norm01(ct_p),  epoch)

    # ── Visualise ─────────────────────────────────────────────────────────────
    if (epoch + 1) % VIS_EVERY == 0 or epoch == start_epoch:
        with torch.no_grad():
            mri_v, ct_v = next(iter(val_loader))
            mri_v, ct_v = mri_v.to(device), ct_v.to(device)
            with torch.cuda.amp.autocast():
                mr_r_v, ct_p_v, _ = model(mri_v)
        visualise(epoch, mri_v, mr_r_v, ct_v, ct_p_v)

    # ── Save training checkpoint (resume support) ─────────────────────────────
    torch.save({
        'epoch':         epoch,
        'model':         model.state_dict(),
        'optimizer':     optimizer.state_dict(),
        'scheduler':     scheduler.state_dict(),
        'best_val_ssim': best_val_ssim,
        'no_improve':    no_improve,
    }, CKPT_PATH)

    # ── Early stopping ────────────────────────────────────────────────────────
    if no_improve >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch+1}: no SSIM improvement for {PATIENCE} epochs.")
        break

# ─── End ─────────────────────────────────────────────────────────────────────
writer.close()
total_min = (time.time() - total_start) / 60
print(f"\n{'='*60}")
print(f"PRETRAINING COMPLETE!")
print(f"Total time: {total_min:.1f} minutes")
print(f"Best combined SSIM: {best_val_ssim:.4f}")
print(f"Encoder weights saved: {BEST_PATH}")
print(f"{'='*60}")
print(f"\nNext step: run 'python main_hybrid.py'")
print(f"It will automatically load the pretrained encoder from:\n  {BEST_PATH}")
print(f"and freeze it for diffusion model training.")
