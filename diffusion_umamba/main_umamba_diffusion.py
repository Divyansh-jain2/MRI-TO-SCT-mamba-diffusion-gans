import PIL
import time
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader
import glob
import scipy.io
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# MONAI Seed Patch for NumPy >= 1.24
import monai
if hasattr(monai, 'transforms') and hasattr(monai.transforms, 'transform'):
    monai.transforms.transform.MAX_SEED = (1 << 32) - 1
if hasattr(monai, 'transforms') and hasattr(monai.transforms, 'compose'):
    monai.transforms.compose.MAX_SEED = (1 << 32) - 1
if hasattr(monai, 'utils') and hasattr(monai.utils, 'misc'):
    monai.utils.misc.MAX_SEED = (1 << 32) - 1

import numpy as np
from random import randint
import random
import time
import re
import itertools
from timm.models.layers import DropPath
from einops import rearrange
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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from network.Diffusion_model_transformer import *
from models import UMamba


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--resume', action='store_true', default=False)
args = parser.parse_args()
RESUME = args.resume

# ── Config: Brain / SynthRAD2023 ─────────────────────────────────────────────
BATCH_SIZE_TRAIN = 8           # Optimal gradients!
img_size         = (192, 192, 96)
patch_size       = (64, 64, 4) # Matches Swin Baseline!
spacing          = (1, 1, 1)
patch_num        = 2
channels         = 1
metric           = torch.nn.L1Loss()

DATA_ROOT    = '/DATA/Munish_Synthetic_CT/mc_ddpm_data/brain_npy'
SAVE_DIR     = './checkpoints_brain_umamba_diffusion'
LOG_DIR      = './runs_umamba_diffusion'
CT_CLIP      = (-1024, 1650)
LR           = 3e-5
WEIGHT_DECAY = 1e-5
EPOCHS       = 500
VIS_EVERY    = 10

os.makedirs(SAVE_DIR, exist_ok=True)
print(f'img_size  : {img_size}')
print(f'patch_size: {patch_size}')
print(f'spacing   : {spacing}')
print(f'data root : {DATA_ROOT}')


# ── Dataset ───────────────────────────────────────────────────────────────────
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


# ── Build the MC-IDDPM process ────────────────────────────────────────────────
diffusion_steps        = 1000
learn_sigma            = True
timestep_respacing     = [50]
sigma_small            = False
class_cond             = False
noise_schedule         = 'linear'
use_kl                 = False
predict_xstart         = True
rescale_timesteps      = True
rescale_learned_sigmas = True
use_checkpoint         = False

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
)
schedule_sampler = UniformSampler(diffusion)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ── Build the MC-IDDPM network ────────────────────────────────────────────────
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

A_to_B_model = UMamba(
    in_ch=2,
    out_ch=2,
    base_ch=64,
    is_diffusion=True,
    strides=((2,2,2), (2,2,1), (2,2,1))
).to(device)


# ── Optimizer ─────────────────────────────────────────────────────────────────
pytorch_total_params = sum(p.numel() for p in A_to_B_model.parameters())
print('parameter number is ' + str(pytorch_total_params))
torch.backends.cudnn.benchmark = True
optimizer = torch.optim.AdamW(A_to_B_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scaler    = torch.cuda.amp.GradScaler()


# ── Inferer ───────────────────────────────────────────────────────────────────
img_num  = 16  # Increased from 2 to 16 to utilize GPU efficiently and massively speed up validation
overlap  = 0.25  
inferer  = SlidingWindowInferer(patch_size, img_num, overlap=overlap, mode='constant')

def diffusion_sampling(condition, model):
    sampled_images = diffusion.p_sample_loop(
        model,
        (condition.shape[0], 1,
         condition.shape[2], condition.shape[3], condition.shape[4]),
        condition=condition, clip_denoised=True,
    )
    return sampled_images


# ── Train function ────────────────────────────────────────────────────────────
def train(model, optimizer, data_loader1, loss_history, epoch, global_step):
    model.train()
    total_samples   = len(data_loader1.dataset)
    A_to_B_loss_sum = []
    total_time      = 0

    for i, (x1, y1) in enumerate(data_loader1):
        traintarget    = y1.view(-1, 1, patch_size[0], patch_size[1], patch_size[2]).to(device)
        traincondition = x1.view(-1, 1, patch_size[0], patch_size[1], patch_size[2]).to(device)
        t, weights     = schedule_sampler.sample(traincondition.shape[0], device)
        aa             = time.time()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            all_loss    = diffusion.training_losses(A_to_B_model, traintarget, traincondition, t)
            A_to_B_loss = (all_loss["loss"] * weights).mean()
            A_to_B_loss_sum.append(all_loss["loss"].mean().detach().cpu().numpy())
        scaler.scale(A_to_B_loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        global_step += 1

        total_time += time.time() - aa
        if i % 30 == 0:
            print('optimization time: ' + str(time.time()-aa))
            print('[' + '{:5}'.format(i * BATCH_SIZE_TRAIN) + '/' + '{:5}'.format(total_samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_loader1)) + '%)]  A_to_B_Loss: ' +
                  '{:6.7f}'.format(np.nanmean(A_to_B_loss_sum)))

    average_loss = np.nanmean(A_to_B_loss_sum)
    loss_history.append(average_loss)
    print("Total time per sample is: " + str(total_time))
    print('Averaged loss is: ' + str(average_loss))
    return average_loss, global_step


# ── Evaluate function ─────────────────────────────────────────────────────────
def evaluate(model, epoch, path, data_loader1, best_loss):
    model.eval()
    aa = time.time()

    loss_all = []
    psnr_all = []
    mc_images = []

    with torch.no_grad():
        for i, (x1, y1) in enumerate(data_loader1):
            if i >= 5:
                break

            target    = y1.to(device)
            condition = x1.to(device)

            # ── Fast Evaluation (1 MC trial instead of 3 for speed) ───────
            mc_runs = []
            for _ in range(1):
                with torch.cuda.amp.autocast():
                    sampled_images = inferer(condition, diffusion_sampling, model)
                mc_runs.append(sampled_images)
            sampled_images = torch.stack(mc_runs).mean(dim=0)

            loss = metric(sampled_images, target)
            loss_all.append(loss.item())
            
            # --- PSNR Calculation ---
            mse = torch.mean((sampled_images - target) ** 2)
            psnr = 10 * torch.log10((2.0 ** 2) / mse).item() if mse > 0 else float('inf')
            psnr_all.append(psnr)
            print(f'L1 loss: {loss.item():.4f}  |  PSNR: {psnr:.2f} dB')

            # Save first sample only
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
    avg_psnr = np.mean(psnr_all)
    print(f'Average Validation L1 loss: {avg_loss:.4f}  |  Average PSNR: {avg_psnr:.2f} dB')
    print('Eval time: ' + str(time.time()-aa))

    if avg_loss < best_loss:
        lo, hi = CT_CLIP
        sct_np = sct_log[0, 0].cpu().numpy()
        sct_hu = (sct_np + 1.0) / 2.0 * (hi - lo) + lo
        nib.save(
            nib.Nifti1Image(sct_hu, np.eye(4)),
            path + 'best_sct_sample0.nii.gz'
        )

    # ── Save Graphical Outputs as PNGs ────────────────────────────
    vis_dir = './visualizations_umamba_diffusion'
    os.makedirs(vis_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('black')
    mid = sct_log[0, 0].shape[-1] // 2
    
    mri_img = x1_log[0, 0, :, :, mid].cpu().numpy()
    target_img = y1_log[0, 0, :, :, mid].cpu().numpy()
    pred_img = sct_log[0, 0, :, :, mid].cpu().numpy()
    
    axes[0].imshow(np.rot90(mri_img), cmap='gray')
    axes[0].set_title('MRI Input', color='white')
    axes[0].axis('off')
    
    axes[1].imshow(np.rot90(target_img), cmap='gray', vmin=-1, vmax=1)
    axes[1].set_title('CT Ground Truth', color='white')
    axes[1].axis('off')
    
    axes[2].imshow(np.rot90(pred_img), cmap='gray', vmin=-1, vmax=1)
    axes[2].set_title(f'Synthetic CT\nPSNR: {avg_psnr:.2f} dB', color='white')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f'epoch_{epoch}_comparison.png'), facecolor='black')
    plt.close()

    return avg_loss, avg_psnr


# ── Dataloaders ───────────────────────────────────────────────────────────────
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


# ── Paths ─────────────────────────────────────────────────────────────────────
path        = './results_brain_umamba_diffusion/'
A_to_B_PATH = os.path.join(SAVE_DIR, 'best_model.pt')
LATEST_PATH = os.path.join(SAVE_DIR, 'latest_model.pt')
if not os.path.exists(path):
    os.makedirs(path)


# ── Initialize Tracking ───────────────────────────────────────────────────────
N_EPOCHS           = 500
best_loss          = 1
global_step        = 0
train_loss_history = []
val_loss_history   = []
val_psnr_history   = []
val_epochs         = []
start_epoch        = 0

os.makedirs('./visualizations_umamba_diffusion', exist_ok=True)

# ── Resilient Resume Logic ────────────────────────────────────────────────────
if os.path.exists(LATEST_PATH):
    print("Found 'latest_model.pt'! Automatically resuming sequence seamlessly...")
    checkpoint  = torch.load(LATEST_PATH)
    A_to_B_model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    scaler.load_state_dict(checkpoint['scaler_state'])
    global_step = checkpoint['global_step']
    best_loss   = checkpoint['best_loss']
    start_epoch = checkpoint['epoch'] + 1
    
    if 'train_loss_history' in checkpoint:
        train_loss_history = checkpoint['train_loss_history']
        val_loss_history   = checkpoint['val_loss_history']
        val_psnr_history   = checkpoint['val_psnr_history']
        val_epochs         = checkpoint['val_epochs']
        
    print(f'Resumed cleanly from exactly epoch {checkpoint["epoch"]}')
elif RESUME and os.path.exists(A_to_B_PATH):
    print("Falling back to explicitly resuming from 'best_model.pt'")
    checkpoint  = torch.load(A_to_B_PATH)
    A_to_B_model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    scaler.load_state_dict(checkpoint['scaler_state'])
    global_step = checkpoint['global_step']
    best_loss   = checkpoint['best_loss']
    start_epoch = checkpoint['epoch'] + 1
    print(f'Resumed cleanly from best baseline epoch {checkpoint["epoch"]}')
else:
    print('Starting fresh training entirely from scratch')

for epoch in range(start_epoch, N_EPOCHS):
    print('Epoch:', epoch)
    start_time = time.time()
    avg_train, global_step = train(A_to_B_model, optimizer, train_loader1,
                                   train_loss_history, epoch, global_step)
    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
    
    if epoch > 0 and epoch % 10 == 0:
        average_loss, average_psnr = evaluate(A_to_B_model, epoch, path, val_loader1, best_loss)
        val_loss_history.append(average_loss)
        val_psnr_history.append(average_psnr)
        val_epochs.append(epoch)
        
        # ── Update Metrics Graph ──────────────────────────────────────────────
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        ax[0].plot(train_loss_history, label='Train Loss', alpha=0.6)
        if len(val_epochs) > 0:
            ax[0].plot(val_epochs, val_loss_history, label='Val Loss', marker='o', color='red')
        ax[0].set_title('Loss History')
        ax[0].set_xlabel('Epoch')
        ax[0].legend()
        
        if len(val_epochs) > 0:
            ax[1].plot(val_epochs, val_psnr_history, color='orange', marker='s')
        ax[1].set_title('Validation PSNR')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('PSNR (dB)')
        
        plt.tight_layout()
        plt.savefig('./visualizations_umamba_diffusion/training_metrics.png')
        plt.close()

        if average_loss < best_loss:
            print('Save the latest best model')
            torch.save({
                'epoch':              epoch,
                'model_state':        A_to_B_model.state_dict(),
                'optimizer_state':    optimizer.state_dict(),
                'scaler_state':       scaler.state_dict(),
                'global_step':        global_step,
                'best_loss':          best_loss,
                'train_loss_history': train_loss_history,
                'val_loss_history':   val_loss_history,
                'val_psnr_history':   val_psnr_history,
                'val_epochs':         val_epochs,
            }, A_to_B_PATH)
            best_loss = average_loss

    # ── ALWAYS Save the Latest Checkpoint to protect against hard crashes ──
    torch.save({
        'epoch':              epoch,
        'model_state':        A_to_B_model.state_dict(),
        'optimizer_state':    optimizer.state_dict(),
        'scaler_state':       scaler.state_dict(),
        'global_step':        global_step,
        'best_loss':          best_loss,
        'train_loss_history': train_loss_history,
        'val_loss_history':   val_loss_history,
        'val_psnr_history':   val_psnr_history,
        'val_epochs':         val_epochs,
    }, LATEST_PATH)

print('Training complete')
