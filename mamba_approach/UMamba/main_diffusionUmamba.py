"""
main_diffusionUmamba.py
=======================
Complete training script for MRI -> synthetic CT using DiffusionUMamba.

This file contains only training/data/evaluation logic.
All model architecture is in diffusion_mamba_models.py.

Usage
-----
  Train from scratch:
      python main_diffusionUmamba.py

  Resume from last best checkpoint:
      python main_diffusionUmamba.py --resume

What changed vs the baseline train.py
--------------------------------------
  Model       : DiffusionUMamba with cfg_dropout_p=0.10, deep_supervision=True
  Losses      : diffusion + deep_supervision + tissue_weighted_l1 + freq_loss
  Scheduler   : linear warmup (10 ep) + cosine annealing (490 ep)
  Evaluation  : guided_sample() with CFG weight w=3.0
  Checkpoints : scheduler_state saved/loaded alongside model/optimizer/scaler
  TensorBoard : each loss term logged separately for diagnosis
"""

# ===========================================================================
# Standard library
# ===========================================================================
import os
import glob
import time
import random
import argparse

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ===========================================================================
# Third-party
# ===========================================================================
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from natsort import natsorted
from monai.inferers import SlidingWindowInferer
from monai.metrics import PSNRMetric, SSIMMetric
from monai.transforms import (
    Compose,
    RandSpatialCropSamplesd,
    ToTensord,
)
from torch.utils.tensorboard import SummaryWriter

# ===========================================================================
# Project imports — diffusion framework
# ===========================================================================
from diffusion.Create_diffusion import create_gaussian_diffusion
from diffusion.resampler import UniformSampler

# ===========================================================================
# Project imports — model and loss utilities (all architecture lives here)
# ===========================================================================
from diffusion_mamba_models import (
    DiffusionUMamba,
    tissue_weighted_l1,
    freq_loss,
    guided_sample,
)

# ===========================================================================
# CLI arguments
# ===========================================================================
parser = argparse.ArgumentParser(description="DiffusionUMamba MRI->sCT training")
parser.add_argument(
    '--resume', action='store_true', default=False,
    help='Resume training from the best saved checkpoint.',
)
args   = parser.parse_args()
RESUME = args.resume


# ===========================================================================
# Config
# ===========================================================================

# ── Data ─────────────────────────────────────────────────────────────────────

DATA_ROOT  = '/DATA/Munish_Synthetic_CT/mc_ddpm_data/brain_npy'
img_size   = (192, 192, 96)
patch_size = (64, 64, 4)
spacing    = (1, 1, 1)
patch_num  = 2
CT_CLIP    = (-1024, 1650)

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE_TRAIN = 4
LR               = 3e-5
WEIGHT_DECAY     = 1e-5
EPOCHS           = 500
WARMUP_EPOCHS    = 10

# ── Loss weights ──────────────────────────────────────────────────────────────
# Each term is logged separately to TensorBoard so you can monitor the
# relative contribution of each component and tune the weights independently.
DEEP_SUP_W   = 0.40   # weighted sum of dec3 + dec2 auxiliary head losses
TW_LOSS_W    = 1.00   # tissue-weighted L1 (bone x5, air x2) on pred_xstart
FREQ_LOSS_W  = 0.10   # FFT magnitude L1 on pred_xstart

# ── Inference (evaluation) ────────────────────────────────────────────────────
CFG_GUIDANCE_W = 3.0  # guidance weight w for classifier-free guidance
EVAL_MC_RUNS   = 3    # number of MC diffusion trials to average per eval volume
EVAL_MAX_VOLS  = 5    # cap on validation volumes evaluated per epoch

# ── Paths ─────────────────────────────────────────────────────────────────────
SAVE_DIR    = './checkpoints_brain_baseline'
LOG_DIR     = './runs'
RESULTS_DIR = './results_brain'
CKPT_PATH   = os.path.join(SAVE_DIR, 'best_model.pt')

os.makedirs(SAVE_DIR,    exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f'img_size  : {img_size}')
print(f'patch_size: {patch_size}')
print(f'spacing   : {spacing}')
print(f'data root : {DATA_ROOT}')


# ===========================================================================
# Dataset
# ===========================================================================

class CustomDataset(Dataset):
    """
    Loads preprocessed brain .npy volumes from disk.

    Each .npy file has shape (2, D, H, W):
        channel 0 : MRI  in [-1, 1]
        channel 1 : CT   in [-1, 1]

    Training mode  : returns random 3-D patches via RandSpatialCropSamplesd.
                     Returned tensors shape: (patch_num, 1, *patch_size)
    Validation mode: returns the full volume.
                     Returned tensors shape: (1, D, H, W)
    """

    def __init__(self, imgs_path: str, train_flag: bool = True):
        self.train_flag = train_flag
        self.files = natsorted(
            glob.glob(imgs_path + "*.npy"),
            key=lambda y: y.lower(),
        )
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .npy files found in {imgs_path}")
        print(
            f"Found {len(self.files)} preprocessed volumes "
            f"[{'train' if train_flag else 'val/test'}]"
        )

        # Patch sampling transform (training only)
        self.patch_transform = Compose([
            RandSpatialCropSamplesd(
                keys=["image", "label"],
                roi_size=patch_size,
                num_samples=patch_num,
                random_size=False,
            ),
            ToTensord(keys=["image", "label"]),
        ])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        data   = np.load(self.files[idx])
        mr_vol = data[0]   # MRI  [-1, 1]
        ct_vol = data[1]   # CT   [-1, 1]

        if not self.train_flag:
            # Full volume for sliding-window inference
            img_tensor   = torch.from_numpy(mr_vol[np.newaxis]).float()
            label_tensor = torch.from_numpy(ct_vol[np.newaxis]).float()
            return img_tensor, label_tensor

        # Training: random patch extraction
        data_dict = {
            "image": mr_vol[np.newaxis],
            "label": ct_vol[np.newaxis],
        }
        out   = self.patch_transform(data_dict)
        img   = np.zeros([patch_num, patch_size[0], patch_size[1], patch_size[2]])
        label = np.zeros([patch_num, patch_size[0], patch_size[1], patch_size[2]])
        for i, sample in enumerate(out):
            img[i]   = sample["image"].numpy()
            label[i] = sample["label"].numpy()

        img_tensor   = torch.unsqueeze(torch.from_numpy(img.copy()),   1).float()
        label_tensor = torch.unsqueeze(torch.from_numpy(label.copy()), 1).float()
        return img_tensor, label_tensor


# ===========================================================================
# Diffusion process
# ===========================================================================

diffusion = create_gaussian_diffusion(
    steps=1000,
    learn_sigma=True,
    sigma_small=False,
    noise_schedule='cosine',
    use_kl=False,
    predict_xstart=True,
    rescale_timesteps=True,
    rescale_learned_sigmas=True,
    timestep_respacing=[50],
)
schedule_sampler = UniformSampler(diffusion)


# ===========================================================================
# Device
# ===========================================================================

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ===========================================================================
# Model
# ===========================================================================

A_to_B_model = DiffusionUMamba(
    in_ch=2,               # noisy CT (ch 0) + MRI condition (ch 1)
    out_ch=2,              # mean + variance (learn_sigma=True)
    base_ch=64,
    time_emb_dim=256,
    d_state=16,
    cfg_dropout_p=0.10,    # 10 % condition dropout for CFG training
    deep_supervision=True, # auxiliary heads at dec3 and dec2
).to(device)

pytorch_total_params = sum(p.numel() for p in A_to_B_model.parameters())
print(f"Total parameters: {pytorch_total_params:,}")
torch.backends.cudnn.benchmark = True


# ===========================================================================
# Optimizer and LR scheduler
# ===========================================================================

optimizer = torch.optim.AdamW(
    A_to_B_model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)

# Warm-up: start at 10 % of LR, ramp up linearly over WARMUP_EPOCHS epochs
scheduler_warmup = LinearLR(
    optimizer,
    start_factor=0.1,
    total_iters=WARMUP_EPOCHS,
)
# Cosine decay: smoothly reduce LR from its peak down to 1e-6
scheduler_cosine = CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS - WARMUP_EPOCHS,
    eta_min=1e-6,
)
# Chain warm-up -> cosine
scheduler = SequentialLR(
    optimizer,
    schedulers=[scheduler_warmup, scheduler_cosine],
    milestones=[WARMUP_EPOCHS],
)

scaler = torch.cuda.amp.GradScaler()


# ===========================================================================
# Sliding-window inferer (used during evaluation)
# ===========================================================================

inferer = SlidingWindowInferer(
    roi_size=patch_size,
    sw_batch_size=1,
    overlap=0.25,
    mode='constant',
)

val_metric = torch.nn.L1Loss()
# Initialize PSNR and SSIM
psnr_metric = PSNRMetric(max_val=2.0) 
ssim_metric = SSIMMetric(data_range=2.0, spatial_dims=3)



# ===========================================================================
# Diffusion sampling wrappers
# ===========================================================================

def diffusion_sampling(condition: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """
    Plain (unconditional) diffusion sampling — used as a fallback.
    Matches the SlidingWindowInferer callable signature: fn(input, network).
    """
    return diffusion.p_sample_loop(
        model,
        shape=(
            condition.shape[0], 1,
            condition.shape[2], condition.shape[3], condition.shape[4],
        ),
        condition=condition,
        clip_denoised=True,
    )


def cfg_sampling(condition: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """
    CFG-guided diffusion sampling.
    Matches the SlidingWindowInferer callable signature: fn(input, network).
    """
    return guided_sample(
        diffusion=diffusion,
        model=model,
        condition=condition,
        out_shape=(
            condition.shape[0], 1,
            condition.shape[2], condition.shape[3], condition.shape[4],
        ),
        w=CFG_GUIDANCE_W,
        clip=True,
    )


# ===========================================================================
# Training step
# ===========================================================================

def train(
    model:       nn.Module,
    optimizer:   torch.optim.Optimizer,
    data_loader: DataLoader,
    loss_history: list,
    epoch:       int,
    writer:      SummaryWriter,
    global_step: int,
) -> tuple:
    """
    One training epoch.

    Total loss = diffusion_loss
               + DEEP_SUP_W  * deep_sup_loss
               + TW_LOSS_W   * tissue_weighted_l1   (on pred_xstart)
               + FREQ_LOSS_W * freq_loss             (on pred_xstart)

    Each term is logged independently to TensorBoard so you can diagnose
    which component is driving changes in behaviour.

    Returns:
        (average_diffusion_loss_for_epoch, updated_global_step)
    """
    model.train()
    total_samples   = len(data_loader.dataset)
    diff_loss_log   = []
    total_time      = 0

    for i, (x1, y1) in enumerate(data_loader):
        traintarget    = y1.view(
            -1, 1, patch_size[0], patch_size[1], patch_size[2]
        ).to(device)
        traincondition = x1.view(
            -1, 1, patch_size[0], patch_size[1], patch_size[2]
        ).to(device)

        t, weights = schedule_sampler.sample(traincondition.shape[0], device)

        tick = time.time()
        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            # ── 1. Diffusion loss (calls model.forward() internally) ──────────
            all_loss  = diffusion.training_losses(
                model, traintarget, traincondition, t
            )
            diff_loss = (all_loss["loss"] * weights).mean()

            # ── 2. Deep supervision auxiliary loss ────────────────────────────
            # model.aux3 and model.aux2 were populated by the forward() above
            ds_loss = model.deep_sup_loss(traintarget) * DEEP_SUP_W

            # ── 3. Tissue-weighted L1 on x_start prediction ───────────────────
            # predict_xstart=True ensures "pred_xstart" key is present
            if "pred_xstart" in all_loss:
                tw_loss = tissue_weighted_l1(
                    pred=all_loss["pred_xstart"].float(),
                    target=traincondition.float(),       # NOTE: MRI is condition
                    condition=traincondition.float(),
                ) * TW_LOSS_W
            else:
                tw_loss = torch.tensor(0.0, device=device)

            # ── 4. Frequency-domain loss on x_start prediction ───────────────
            if "pred_xstart" in all_loss:
                fq_loss = freq_loss(
                    pred=all_loss["pred_xstart"].float(),
                    target=traintarget.float(),
                ) * FREQ_LOSS_W
            else:
                fq_loss = torch.tensor(0.0, device=device)

            # ── Total loss ────────────────────────────────────────────────────
            total_loss = diff_loss + ds_loss + tw_loss + fq_loss
            diff_loss_log.append(all_loss["loss"].mean().detach().cpu().numpy())

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # ── TensorBoard: per-step scalars ─────────────────────────────────────
        writer.add_scalar("Loss/total_step",     total_loss.item(), global_step)
        writer.add_scalar("Loss/diff_step",      diff_loss.item(),  global_step)
        writer.add_scalar("Loss/deep_sup_step",  ds_loss.item(),    global_step)
        writer.add_scalar("Loss/tissue_w_step",  tw_loss.item(),    global_step)
        writer.add_scalar("Loss/freq_step",      fq_loss.item(),    global_step)
        writer.add_scalar("GradNorm/step",       grad_norm,         global_step)
        writer.add_scalar("LR/step",
                          optimizer.param_groups[0]['lr'],           global_step)
        global_step += 1

        total_time += time.time() - tick
        if i % 30 == 0:
            print(
                f'  [{i * BATCH_SIZE_TRAIN:5d}/{total_samples:5d} '
                f'({100 * i / len(data_loader):3.0f}%)]  '
                f'Diff loss: {np.nanmean(diff_loss_log):8.7f}  '
                f'DS: {ds_loss.item():.5f}  '
                f'TW: {tw_loss.item():.5f}  '
                f'Freq: {fq_loss.item():.5f}'
            )

    avg_loss = float(np.nanmean(diff_loss_log))
    loss_history.append(avg_loss)
    writer.add_scalar("Loss/train_epoch", avg_loss, epoch)
    print(f'  Epoch avg diffusion loss : {avg_loss:.7f}')
    print(f'  Total iteration time     : {total_time:.1f} s')
    return avg_loss, global_step


# ===========================================================================
# Evaluation step
# ===========================================================================

def evaluate(
    model:       nn.Module,
    epoch:       int,
    data_loader: DataLoader,
    best_loss:   float,
    writer:      SummaryWriter,
) -> float:
    """
    Evaluation on the validation set.

    Uses CFG-guided sliding-window inference with EVAL_MC_RUNS Monte Carlo
    trials averaged per volume. Logs MRI / ground-truth CT / synthetic CT
    slices to TensorBoard for visual inspection.

    Returns:
        Mean L1 loss over evaluated volumes.
    """
    model.eval()
    t_start  = time.time()
    loss_all = []

    # Variables for TensorBoard image logging (set on first volume)
    x1_log  = None
    y1_log  = None
    sct_log = None

    with torch.no_grad():
        for i, (x1, y1) in enumerate(data_loader):
            if i >= EVAL_MAX_VOLS:
                break

            target    = y1.to(device)    # (1, 1, D, H, W)
            condition = x1.to(device)    # (1, 1, D, H, W)

            # ── Monte Carlo averaging over EVAL_MC_RUNS diffusion trials ───────
            mc_runs = []
            for _ in range(EVAL_MC_RUNS):
                with torch.cuda.amp.autocast():
                    samp = inferer(condition, cfg_sampling, model)
                mc_runs.append(samp)
            sampled_images = torch.stack(mc_runs).mean(dim=0)

            loss = val_metric(sampled_images, target)
            loss_all.append(loss.item())
            psnr_metric(y_pred=sampled_images, y=target)
            ssim_metric(y_pred=sampled_images, y=target)
            print(f'  Vol {i}  L1 loss: {loss.item():.6f}')

            if i == 0:
                # Save NIfTI of first validation volume
                lo, hi   = CT_CLIP
                sct_np   = sampled_images[0, 0].cpu().numpy()
                sct_hu   = (sct_np + 1.0) / 2.0 * (hi - lo) + lo
                nib.save(
                    nib.Nifti1Image(sct_hu, np.eye(4)),
                    os.path.join(RESULTS_DIR, f'sct_epoch{epoch}_sample0.nii.gz'),
                )
                x1_log  = x1
                y1_log  = y1
                sct_log = sampled_images

    avg_loss = float(np.mean(loss_all))
    avg_psnr = psnr_metric.aggregate().item()
    avg_ssim = ssim_metric.aggregate().item()

    psnr_metric.reset()
    ssim_metric.reset()

    print(f'  Average val L1 loss : {avg_loss:.6f}')
    print(f'  Average val PSNR    : {avg_psnr:.4f} dB')
    print(f'  Average val SSIM    : {avg_ssim:.4f}')
    print(f'  Eval time           : {time.time() - t_start:.1f} s')

    # ── Save best synthetic CT NIfTI ─────────────────────────────────────────
    if avg_loss < best_loss and sct_log is not None:
        lo, hi  = CT_CLIP
        sct_np  = sct_log[0, 0].cpu().numpy()
        sct_hu  = (sct_np + 1.0) / 2.0 * (hi - lo) + lo
        nib.save(
            nib.Nifti1Image(sct_hu, np.eye(4)),
            os.path.join(RESULTS_DIR, 'best_sct_sample0.nii.gz'),
        )

    # ── TensorBoard: scalar + images ─────────────────────────────────────────
    writer.add_scalar("Loss/val_epoch", avg_loss, epoch)
    writer.add_scalar("Metrics/val_PSNR", avg_psnr, epoch)
    writer.add_scalar("Metrics/val_SSIM", avg_ssim, epoch)

    if sct_log is not None:
        def norm01(t: torch.Tensor) -> np.ndarray:
            t = t.float()
            return ((t - t.min()) / (t.max() - t.min() + 1e-8)).cpu().numpy()

        mid = sct_log[0, 0].shape[-1] // 2
        writer.add_image(
            "Val/MRI_input",
            norm01(x1_log[0, 0, :, :, mid])[None],
            epoch,
        )
        writer.add_image(
            "Val/CT_groundtruth",
            norm01(y1_log[0, 0, :, :, mid])[None],
            epoch,
        )
        writer.add_image(
            "Val/CT_synthetic",
            norm01(sct_log[0, 0, :, :, mid])[None],
            epoch,
        )

    return avg_loss


# ===========================================================================
# Dataloaders
# ===========================================================================

training_set = CustomDataset(DATA_ROOT + '/imagesTr/',  train_flag=True)
val_set      = CustomDataset(DATA_ROOT + '/imagesVal/', train_flag=False)
testing_set  = CustomDataset(DATA_ROOT + '/imagesTs/',  train_flag=False)

train_loader = DataLoader(
    training_set,
    batch_size=BATCH_SIZE_TRAIN,
    shuffle=True,
    pin_memory=True,
    drop_last=False,
    num_workers=4,
)
val_loader = DataLoader(
    val_set,
    batch_size=1,
    shuffle=False,
    pin_memory=True,
    drop_last=False,
    num_workers=2,
)
test_loader = DataLoader(
    testing_set,
    batch_size=1,
    shuffle=False,
    pin_memory=True,
    drop_last=False,
    num_workers=2,
)

print(f'Train batches : {len(train_loader)}')
print(f'Val   batches : {len(val_loader)}')
print(f'Test  batches : {len(test_loader)}')


# ===========================================================================
# TensorBoard writer
# ===========================================================================

run_name = f"brain_{time.strftime('%Y%m%d_%H%M%S')}"
writer   = SummaryWriter(log_dir=os.path.join(LOG_DIR, run_name))
print(f'TensorBoard run : {run_name}')
print(f'Monitor with    : tensorboard --logdir {LOG_DIR} --port 6006')


# ===========================================================================
# Training loop
# ===========================================================================

best_loss          = 1.0
global_step        = 0
start_epoch        = 0
train_loss_history = []

if RESUME:
    checkpoint = torch.load(CKPT_PATH, map_location=device)
    A_to_B_model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    scaler.load_state_dict(checkpoint['scaler_state'])
    scheduler.load_state_dict(checkpoint['scheduler_state'])
    global_step = checkpoint['global_step']
    best_loss   = checkpoint['best_loss']
    start_epoch = checkpoint['epoch'] + 1
    print(f'Resumed from epoch {checkpoint["epoch"]}  '
          f'(best_loss = {best_loss:.6f})')
else:
    print('Starting training from scratch.')

for epoch in range(start_epoch, EPOCHS):
    print(f'\nEpoch {epoch}/{EPOCHS - 1}')
    epoch_start = time.time()

    # ── Training ──────────────────────────────────────────────────────────────
    avg_train, global_step = train(
        model=A_to_B_model,
        optimizer=optimizer,
        data_loader=train_loader,
        loss_history=train_loss_history,
        epoch=epoch,
        writer=writer,
        global_step=global_step,
    )
    scheduler.step()
    print(f'Epoch time: {time.time() - epoch_start:.2f} s  '
          f'LR: {optimizer.param_groups[0]["lr"]:.2e}')

    # ── Validation (every 10 epochs) ──────────────────────────────────────────
    if epoch % 10 == 0:
        avg_val = evaluate(
            model=A_to_B_model,
            epoch=epoch,
            data_loader=val_loader,
            best_loss=best_loss,
            writer=writer,
        )

        # ── Save checkpoint if validation improved ────────────────────────────
        if avg_val < best_loss:
            print(f'  Val improved {best_loss:.6f} -> {avg_val:.6f}. Saving checkpoint.')
            torch.save(
                {
                    'epoch':           epoch,
                    'model_state':     A_to_B_model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scaler_state':    scaler.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'global_step':     global_step,
                    'best_loss':       avg_val,          # save the new best, not old
                },
                CKPT_PATH,
            )
            best_loss = avg_val

writer.close()
print('\nTraining complete.')