"""
Training script for Hybrid MRI Encoder + Cross-Attention Diffusion Model.
Optimized for lab computers with strong GPUs.

Key changes from main.py:
  - Uses HybridSwinVITModel (MRI encoder + cross-attention UNet)
  - Uses HybridGaussianDiffusion (no concatenation, separate MRI conditioning)
  - MRI passed as separate condition, not concatenated with noisy CT
"""

import PIL
import time
import torch
import torchvision
import torch.nn.functional as F
from torch import nn
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader
import glob
import scipy.io
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from random import randint
import random
import time
import re
import itertools
from timm.models.layers import DropPath
from scipy import ndimage
from skimage import io
from skimage import transform
from natsort import natsorted
from skimage.transform import rotate, AffineTransform
from timm.models.layers import DropPath, to_3tuple, trunc_normal_
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    RandAffined,
    RandCropByLabelClassesd,
    SpatialPadd,
    RandAdjustContrastd,
    RandShiftIntensityd,
    ScaleIntensityd,
    NormalizeIntensityd,
    RandScaleIntensityd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    ScaleIntensityRangePercentilesd,
    Resized,
    Transposed,
    RandSpatialCropd,
    RandSpatialCropSamplesd,
    ResizeWithPadOrCropd
)
from monai.transforms import (CastToTyped,
                              Compose, CropForegroundd, EnsureChannelFirstd, LoadImaged,
                              NormalizeIntensity, RandCropByPosNegLabeld,
                              RandFlipd, RandGaussianNoised,
                              RandGaussianSmoothd, RandScaleIntensityd,
                              RandZoomd, SpatialCrop, SpatialPadd, EnsureTyped)
from monai.transforms.compose import MapTransform
from monai.config import print_config
from monai.metrics import DiceMetric
from skimage.transform import resize
import scipy.io
import matplotlib.pyplot as plt
from monai.inferers import SlidingWindowInferer
import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F
import copy
from diffusion.Create_diffusion import *
from diffusion.resampler import *
import nibabel as nib
from torch.utils.tensorboard import SummaryWriter

from network.hybrid_model import HybridSwinVITModel
from diffusion.HybridGaussianDiffusion import HybridGaussianDiffusion

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — Optimized for ~10GB VRAM
# ═══════════════════════════════════════════════════════════════════════════════

BATCH_SIZE_TRAIN = 8  # Increased batch size
img_size         = (192, 192, 96)
patch_size       = (64, 64, 4)
spacing          = (1, 1, 1)
patch_num        = 2
channels         = 1
metric           = torch.nn.L1Loss()

DATA_ROOT    = '/home/teaching/Desktop/Munish_Synthetic_CT/mc_ddpm_data/brain_npy'
SAVE_DIR     = './checkpoints_brain_hybrid'
LOG_DIR      = './runs'
CT_CLIP      = (-1024, 1650)
LR           = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS       = 500
VIS_EVERY    = 10

# Path to pretrained MRI encoder (from pretrain_mri_encoder.py)
# If this file exists: encoder is loaded + frozen  (best quality)
# If empty string '': falls back to joint training (freeze_encoder=False)
MRI_ENCODER_PRETRAIN_PATH = './checkpoints_mri_encoder/best_mri_encoder.pt'

os.makedirs(SAVE_DIR, exist_ok=True)
print(f'img_size  : {img_size}')
print(f'patch_size: {patch_size}')
print(f'spacing   : {spacing}')
print(f'data root : {DATA_ROOT}')

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL CONFIG — Full quality for ~10GB VRAM
# ═══════════════════════════════════════════════════════════════════════════════

# MRI Encoder config (must match pretrain_mri_encoder.py EXACTLY)
ENC_CHANNELS        = (64, 128, 192, 256)
ENCODER_WINDOW      = (4, 4, 4)
ENCODER_NUM_HEADS   = (4, 4, 8, 8)
ENCODER_POOL_KERNEL = (2, 2, 1)  # Match pretraining

# Denoiser UNet config (full)
MODEL_CHANNELS     = 64
CHANNEL_MULT       = (1, 2, 4)  # 3 levels
NUM_RES_BLOCKS     = [2, 2, 2]
SAMPLE_KERNEL     = ([2,2,2],[2,2,1])
NUM_HEADS          = [4, 4, 8]
ATTENTION_RES      = "32,16,8"

# Diffusion config
DIFFUSION_STEPS    = 1000
TIMESTEP_RESPACING = [50]
NOISE_SCHEDULE     = 'linear'

# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
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

        self.patch_transform = Compose([
            RandSpatialCropSamplesd(
                keys=["image", "label"],
                roi_size=patch_size,
                num_samples=patch_num,
                random_size=False,
            ),
            ToTensord(keys=["image", "label"]),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data   = np.load(self.files[idx])
        mr_vol = data[0]   # MRI  [-1, 1]
        ct_vol = data[1]   # CT   [-1, 1]

        data_dict = {
            "image": mr_vol[np.newaxis],
            "label": ct_vol[np.newaxis],
        }

        if not self.train_flag:
            img_tensor   = torch.from_numpy(mr_vol[np.newaxis]).float()
            label_tensor = torch.from_numpy(ct_vol[np.newaxis]).float()
        else:
            out   = self.patch_transform(data_dict)
            img   = np.zeros([patch_num, patch_size[0], patch_size[1], patch_size[2]])
            label = np.zeros([patch_num, patch_size[0], patch_size[1], patch_size[2]])
            for i, sample in enumerate(out):
                img[i]   = sample["image"].numpy()
                label[i] = sample["label"].numpy()
            img_tensor   = torch.unsqueeze(torch.from_numpy(img.copy()),   1).float()
            label_tensor = torch.unsqueeze(torch.from_numpy(label.copy()), 1).float()

        return img_tensor, label_tensor


# ═══════════════════════════════════════════════════════════════════════════════
# Build Diffusion Process
# ═══════════════════════════════════════════════════════════════════════════════
diffusion_steps        = DIFFUSION_STEPS
learn_sigma            = True
timestep_respacing     = TIMESTEP_RESPACING
sigma_small            = False
class_cond             = False
noise_schedule         = NOISE_SCHEDULE
use_kl                 = False
predict_xstart         = False
rescale_timesteps      = True
rescale_learned_sigmas = True
use_checkpoint         = False

# Use HybridGaussianDiffusion instead of standard GaussianDiffusion
diffusion = create_gaussian_diffusion(
    steps=diffusion_steps,
    learn_sigma=learn_sigma,
    sigma_small=sigma_small,
    noise_schedule=noise_schedule,
    use_kl=use_kl,
    predict_xstart=predict_xstart,
    rescale_timesteps=rescale_timesteps,
    rescale_learned_sigmas=rescale_learned_sigmas,
    timestep_respacing=timestep_respacing,
    diffusion_class=HybridGaussianDiffusion,  # Key: use hybrid diffusion
)
schedule_sampler = UniformSampler(diffusion)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ═══════════════════════════════════════════════════════════════════════════════
# Build Hybrid Model
# ═══════════════════════════════════════════════════════════════════════════════
attention_ds = []
for res in ATTENTION_RES.split(","):
    attention_ds.append(int(res))

use_scale_shift_norm = True
resblock_updown      = False
dropout              = 0

# Create Hybrid Model (matching main.py configs)
model = HybridSwinVITModel(
    image_size=patch_size,
    in_channels=1,  # Now 1 (noisy CT only, MRI is separate)
    model_channels=MODEL_CHANNELS,
    out_channels=2,
    dims=3,
    sample_kernel=SAMPLE_KERNEL,
    num_res_blocks=NUM_RES_BLOCKS,
    attention_resolutions=tuple(attention_ds),
    dropout=dropout,
    channel_mult=CHANNEL_MULT,
    num_classes=None,
    use_checkpoint=use_checkpoint,
    use_fp16=False,
    num_heads=NUM_HEADS,
    window_size=None,
    num_head_channels=64,
    num_heads_upsample=-1,
    use_scale_shift_norm=use_scale_shift_norm,
    resblock_updown=resblock_updown,
    use_new_attention_order=False,
    # Hybrid-specific params
    enc_channels=ENC_CHANNELS,
    freeze_encoder=False,  # Will be overridden below if pretrained weights exist
    encoder_window_size=ENCODER_WINDOW,
    encoder_num_heads=ENCODER_NUM_HEADS,
    encoder_pool_kernel=ENCODER_POOL_KERNEL,
).to(device)

# ── Load pretrained MRI encoder weights if available ─────────────────────────
_pretrain_loaded = False
if MRI_ENCODER_PRETRAIN_PATH and os.path.exists(MRI_ENCODER_PRETRAIN_PATH):
    print(f"\n[PRETRAINED ENCODER] Loading weights from: {MRI_ENCODER_PRETRAIN_PATH}")
    enc_weights = torch.load(MRI_ENCODER_PRETRAIN_PATH, map_location=device)
    missing, unexpected = model.mri_encoder.load_state_dict(enc_weights, strict=True)
    if missing:
        print(f"  WARNING: missing keys: {missing}")
    # Freeze encoder — it has been pretrained, denoiser trains alone
    model.mri_encoder._freeze_encoder = True
    model.mri_encoder._freeze()
    _pretrain_loaded = True
    print(f"  ✓ Encoder loaded and frozen. Denoiser UNet will train independently.")
else:
    print(f"\n[JOINT TRAINING] No pretrained encoder found at: {MRI_ENCODER_PRETRAIN_PATH}")
    print(f"  Falling back to joint encoder+denoiser training (freeze_encoder=False).")
    print(f"  Tip: run 'python pretrain_mri_encoder.py' first for better results.")

print(f"\n{'='*60}")
print("HYBRID MODEL SUMMARY")
print(f"{'='*60}")
pytorch_total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params = pytorch_total_params - trainable_params
print(f"Total parameters:      {pytorch_total_params:>12,}")
print(f"Trainable (UNet):     {trainable_params:>12,}")
print(f"Frozen (MRI Encoder): {frozen_params:>12,}")
print(f"Encoder pretrained:   {'YES ✓' if _pretrain_loaded else 'NO (joint training)':>12}")
print(f"{'='*60}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Optimizer
# ═══════════════════════════════════════════════════════════════════════════════
torch.backends.cudnn.benchmark = True
optimizer = torch.optim.AdamW(
    model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
)
scaler    = torch.cuda.amp.GradScaler()
# Cosine annealing: smoothly decays LR to 1e-6 over 500 epochs
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-6
)


# ═══════════════════════════════════════════════════════════════════════════════
# Inferer
# ═══════════════════════════════════════════════════════════════════════════════
img_num  = 2  # Reduced for speed, increase if you have more VRAM
overlap  = 0.25
inferer  = SlidingWindowInferer(patch_size, img_num, overlap=overlap, mode='constant')


def diffusion_sampling(condition, model):
    """Sample from diffusion model with MRI as separate condition."""
    sampled_images = diffusion.p_sample_loop(
        model,
        (condition.shape[0], 1,
         condition.shape[2], condition.shape[3], condition.shape[4]),
        condition=condition,  # This is the MRI volume
        clip_denoised=True,
    )
    return sampled_images


# ═══════════════════════════════════════════════════════════════════════════════
# Train Function
# ═══════════════════════════════════════════════════════════════════════════════
def train(model, optimizer, data_loader1, loss_history, epoch, writer, global_step):
    model.train()
    total_samples   = len(data_loader1.dataset)
    A_to_B_loss_sum = []
    total_time      = 0

    for i, (x1, y1) in enumerate(data_loader1):
        # x1 = MRI (condition), y1 = CT (target)
        traintarget    = y1.view(-1, 1, patch_size[0], patch_size[1], patch_size[2]).to(device)
        mri_condition  = x1.view(-1, 1, patch_size[0], patch_size[1], patch_size[2]).to(device)
        t, weights     = schedule_sampler.sample(mri_condition.shape[0], device)
        aa             = time.time()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            # KEY CHANGE: Pass MRI as separate condition, not concatenated
            all_loss = diffusion.training_losses(
                model, traintarget, condition_start=mri_condition, t=t
            )
            A_to_B_loss = (all_loss["loss"] * weights).mean()
            A_to_B_loss_sum.append(all_loss["loss"].mean().detach().cpu().numpy())
        scaler.scale(A_to_B_loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        writer.add_scalar("Loss/train_step",     A_to_B_loss.item(), global_step)
        writer.add_scalar("GradNorm/train_step", grad_norm,          global_step)
        global_step += 1

        total_time += time.time() - aa
        if i % 30 == 0:
            print('optimization time: ' + str(time.time()-aa))
            print('[' + '{:5}'.format(i * BATCH_SIZE_TRAIN) + '/' + '{:5}'.format(total_samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_loader1)) + '%)]  A_to_B_Loss: ' +
                  '{:6.7f}'.format(np.nanmean(A_to_B_loss_sum)))

    average_loss = np.nanmean(A_to_B_loss_sum)
    loss_history.append(average_loss)
    writer.add_scalar("Loss/train_epoch", average_loss, epoch)
    print("Total time per sample is: " + str(total_time))
    print('Averaged loss is: ' + str(average_loss))
    return average_loss, global_step


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluate Function
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate(model, epoch, path, data_loader1, best_loss, writer):
    model.eval()
    aa = time.time()

    loss_all = []
    mc_images = []

    with torch.no_grad():
        for i, (x1, y1) in enumerate(data_loader1):
            if i >= 5:  # Evaluate 5 volumes for robust metrics
                break

            target     = y1.to(device)
            mri_cond   = x1.to(device)

            # 3 MC runs for robust uncertainty-averaged prediction
            mc_runs = []
            for _ in range(3):
                with torch.cuda.amp.autocast():
                    sampled_images = inferer(mri_cond, diffusion_sampling, model)
                mc_runs.append(sampled_images)
            sampled_images = torch.stack(mc_runs).mean(dim=0)

            loss = metric(sampled_images, target)
            loss_all.append(loss.item())
            print('L1 loss: ' + str(loss.item()))

            if i == 0:
                lo, hi = CT_CLIP
                sct_np = sampled_images[0, 0].cpu().numpy()
                sct_hu = (sct_np + 1.0) / 2.0 * (hi - lo) + lo
                nib.save(
                    nib.Nifti1Image(sct_hu, np.eye(4)),
                    path + 'sct_epoch' + str(epoch) + '_sample0.nii.gz'
                )

                mid = sct_np.shape[-1] // 2
                x1_log = x1
                y1_log = y1
                sct_log = sampled_images

    avg_loss = np.mean(loss_all)
    print('Average L1 loss: ' + str(avg_loss))
    print('Eval time: ' + str(time.time()-aa))

    if avg_loss < best_loss:
        lo, hi = CT_CLIP
        sct_np = sct_log[0, 0].cpu().numpy()
        sct_hu = (sct_np + 1.0) / 2.0 * (hi - lo) + lo
        nib.save(
            nib.Nifti1Image(sct_hu, np.eye(4)),
            path + 'best_sct_sample0.nii.gz'
        )

    writer.add_scalar("Loss/val_epoch", avg_loss, epoch)

    def norm01(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)
    mid = sct_log[0, 0].shape[-1] // 2
    writer.add_image("Val/MRI_input",      norm01(x1_log[0, 0, :, :, mid].numpy())[None],               epoch)
    writer.add_image("Val/CT_groundtruth", norm01(y1_log[0, 0, :, :, mid].numpy())[None],               epoch)
    writer.add_image("Val/CT_synthetic",   norm01(sct_log[0, 0, :, :, mid].cpu().numpy())[None],        epoch)

    return avg_loss


# ═══════════════════════════════════════════════════════════════════════════════
# Dataloaders
# ═══════════════════════════════════════════════════════════════════════════════
training_set1 = CustomDataset(DATA_ROOT + '/imagesTr/', train_flag=True)
testing_set1  = CustomDataset(DATA_ROOT + '/imagesTs/', train_flag=False)
val_set1      = CustomDataset(DATA_ROOT + '/imagesVal/', train_flag=False)

train_params = {'batch_size': BATCH_SIZE_TRAIN, 'shuffle': True,  'pin_memory': True, 'drop_last': False, 'num_workers': 4}
test_params  = {'batch_size': 1,                'shuffle': False, 'pin_memory': True, 'drop_last': False, 'num_workers': 2}

train_loader1 = torch.utils.data.DataLoader(training_set1, **train_params)
test_loader1  = torch.utils.data.DataLoader(testing_set1,  **test_params)
val_loader1   = torch.utils.data.DataLoader(val_set1,      **test_params)

print(f'Train batches : {len(train_loader1)}')
print(f'Val batches   : {len(val_loader1)}')
print(f'Test batches  : {len(test_loader1)}')


# ═══════════════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════════════
path        = './results_brain_hybrid/'
A_to_B_PATH = os.path.join(SAVE_DIR, 'best_model.pt')
if not os.path.exists(path):
    os.makedirs(path)


# ═══════════════════════════════════════════════════════════════════════════════
# TensorBoard
# ═══════════════════════════════════════════════════════════════════════════════
run_name = f"hybrid_brain_{time.strftime('%Y%m%d_%H%M%S')}"
writer   = SummaryWriter(log_dir=os.path.join(LOG_DIR, run_name))
print(f'TensorBoard run : {run_name}')
print(f'Monitor with    : tensorboard --logdir ./runs --port 6006')


# ═══════════════════════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════════════════════
N_EPOCHS           = EPOCHS
best_loss          = 0.0545  # Best loss from epoch 380 in logs
start_epoch        = 300     # Resuming from checkpoint at epoch 300
global_step        = start_epoch * len(train_loader1)
train_loss_history = []
test_loss_history  = []
total_start_time   = time.time()
VAL_EVERY          = 10   # Validate every 10 epochs
VIS_SAVE_EVERY     = 20   # Save visualisation PNG every 20 epochs

# Resume from checkpoint
checkpoint_path = os.path.join(SAVE_DIR, f'checkpoint_epoch_{start_epoch}.pt')
if os.path.exists(checkpoint_path):
    print(f"\n[RESUME] Loading checkpoint: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=True)
    
    # Fast-forward scheduler to the correct epoch to maintain LR decay
    print(f"[RESUME] Fast-forwarding scheduler to epoch {start_epoch}...")
    for _ in range(start_epoch):
        scheduler.step()
    print(f"[RESUME] Scheduler resumed at LR: {scheduler.get_last_lr()[0]:.2e}")
else:
    print(f"\n[WARNING] Checkpoint not found at {checkpoint_path}")
    print("Starting from scratch or best model if uncommented below.")
    # model.load_state_dict(torch.load(A_to_B_PATH), strict=False)

print(f'\n{"="*60}')
print(f'RESUMING TRAINING: {start_epoch+1} to {N_EPOCHS} epochs')
print(f'{"="*60}\n')

for epoch in range(start_epoch, N_EPOCHS):
    epoch_start = time.time()
    
    # ── Training ──
    print(f'[{epoch+1}/{N_EPOCHS}] TRAINING...', flush=True)
    avg_train, global_step = train(model, optimizer, train_loader1,
                                   train_loss_history, epoch, writer, global_step)
    train_time = time.time() - epoch_start
    
    # ── LR Schedule step ──
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    
    # ── ETA Calculation ──
    avg_time = (time.time() - total_start_time) / (epoch + 1)
    remaining = avg_time * (N_EPOCHS - epoch - 1)
    eta_mins = remaining / 60
    
    print(f'[{epoch+1}/{N_EPOCHS}] TRAIN | Loss: {avg_train:.4f} | LR: {current_lr:.2e} | Time: {train_time:.1f}s | ETA: {eta_mins:.1f} min')
    
    # ── Validation (every 10 epochs) ──────────────────────────────────────
    if (epoch + 1) % VAL_EVERY == 0 or epoch == 0:
        print(f'[{epoch+1}/{N_EPOCHS}] VALIDATING...', flush=True)
        val_start = time.time()
        average_loss = evaluate(model, epoch, path, val_loader1, best_loss, writer)
        val_time = time.time() - val_start
        
        if average_loss < best_loss:
            print(f'[{epoch+1}/{N_EPOCHS}] ★ SAVED BEST MODEL (loss: {average_loss:.4f})')
            torch.save(model.state_dict(), A_to_B_PATH)
            best_loss = average_loss
        else:
            print(f'[{epoch+1}/{N_EPOCHS}] VAL | Loss: {average_loss:.4f} | Time: {val_time:.1f}s')

    # ── Save visualisation PNG every 20 epochs ──────────────────────────────
    if (epoch + 1) % VIS_SAVE_EVERY == 0 or epoch == 0:
        model.eval()
        vis_dir = os.path.join(path, 'vis')
        os.makedirs(vis_dir, exist_ok=True)
        with torch.no_grad():
            try:
                mri_v, ct_v = next(iter(val_loader1))
                mri_v = mri_v.to(device)
                ct_v  = ct_v.to(device)
                with torch.cuda.amp.autocast():
                    sct_v = inferer(mri_v, diffusion_sampling, model)

                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                mid = sct_v.shape[-1] // 2

                def _np(t, idx=0):
                    s = t[0, idx, :, :, mid].float().detach().cpu().numpy()
                    return (s - s.min()) / (s.max() - s.min() + 1e-8)

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(_np(mri_v), cmap='gray'); axes[0].set_title('MRI Input')
                axes[1].imshow(_np(ct_v),  cmap='gray'); axes[1].set_title('CT Ground Truth')
                axes[2].imshow(_np(sct_v), cmap='gray'); axes[2].set_title(f'Generated CT (ep {epoch+1})')
                for a in axes: a.axis('off')
                plt.suptitle(f'Epoch {epoch+1} — Hybrid Diffusion Model', fontsize=13)
                plt.tight_layout()
                vis_path = os.path.join(vis_dir, f'epoch_{epoch+1:04d}.png')
                plt.savefig(vis_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                print(f'[{epoch+1}/{N_EPOCHS}] Saved visualisation → {vis_path}')
            except Exception as e:
                print(f'[{epoch+1}/{N_EPOCHS}] Visualisation skipped: {e}')
        model.train()

    # ── Periodic checkpoint ───────────────────────────────────────────────
    if (epoch + 1) % 100 == 0:
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f'checkpoint_epoch_{epoch+1}.pt'))
        print(f'[{epoch+1}/{N_EPOCHS}] CHECKPOINT saved')

writer.close()
total_time = (time.time() - total_start_time) / 60
print(f'\n{"="*60}')
print(f'TRAINING COMPLETE! Total time: {total_time:.1f} minutes')
print(f'Best val loss: {best_loss:.4f}')
print(f'Model saved: {A_to_B_PATH}')
print(f'{"="*60}')
