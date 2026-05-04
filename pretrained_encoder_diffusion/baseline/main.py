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
from torch.utils.tensorboard import SummaryWriter

from network.Diffusion_model_transformer import *

# ── Config: Brain / SynthRAD2023 ─────────────────────────────────────────────
BATCH_SIZE_TRAIN = 4
img_size         = (192, 192, 96)
patch_size       = (64, 64, 4)
spacing          = (1, 1, 1)
patch_num        = 2
channels         = 1
metric           = torch.nn.L1Loss()

DATA_ROOT    = '/home/teaching/Desktop/Munish_Synthetic_CT/mc_ddpm_data/brain_npy'
SAVE_DIR     = './checkpoints_brain_baseline'
LOG_DIR      = './runs'
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

num_channels          = 64
attention_resolutions = "32,16,8"
channel_mult          = (1, 2, 3, 4)
num_heads             = [4, 4, 8, 16]
window_size           = [[4,4,4],[4,4,4],[4,4,2],[4,4,2]]
num_res_blocks        = [2, 2, 2, 2]
sample_kernel         = ([2,2,2],[2,2,1],[2,2,1],[2,2,1]),

attention_ds = []
for res in attention_resolutions.split(","):
    attention_ds.append(int(res))
class_cond           = False
use_scale_shift_norm = True
resblock_updown      = False
dropout              = 0

A_to_B_model = SwinVITModel(
    image_size=patch_size,
    in_channels=2,
    model_channels=num_channels,
    out_channels=2,
    dims=3,
    sample_kernel=sample_kernel,
    num_res_blocks=num_res_blocks,
    attention_resolutions=tuple(attention_ds),
    dropout=dropout,
    channel_mult=channel_mult,
    num_classes=None,
    use_checkpoint=False,
    use_fp16=False,
    num_heads=num_heads,
    window_size=window_size,
    num_head_channels=64,
    num_heads_upsample=-1,
    use_scale_shift_norm=use_scale_shift_norm,
    resblock_updown=resblock_updown,
    use_new_attention_order=False,
).to(device)


# ── Optimizer ─────────────────────────────────────────────────────────────────
pytorch_total_params = sum(p.numel() for p in A_to_B_model.parameters())
print('parameter number is ' + str(pytorch_total_params))
torch.backends.cudnn.benchmark = True
optimizer = torch.optim.AdamW(A_to_B_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scaler    = torch.cuda.amp.GradScaler()


# ── Inferer ───────────────────────────────────────────────────────────────────
# Much faster: reduce sliding windows for validation speed
img_num  = 2  # Reduced from 12 to 2
overlap  = 0.25  # Reduced from 0.5 to 0.25
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
def train(model, optimizer, data_loader1, loss_history, epoch, writer, global_step):
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


# ── Evaluate function ─────────────────────────────────────────────────────────
def evaluate(model, epoch, path, data_loader1, best_loss, writer):
    model.eval()
    aa = time.time()

    loss_all = []
    mc_images = []

    with torch.no_grad():
        for i, (x1, y1) in enumerate(data_loader1):
            if i >= 5:
                break

            target    = y1.to(device)
            condition = x1.to(device)

            # ── 3 MC trials averaged ──────────────────────────────────────
            mc_runs = []
            for _ in range(3):
                with torch.cuda.amp.autocast():
                    sampled_images = inferer(condition, diffusion_sampling, model)
                mc_runs.append(sampled_images)
            sampled_images = torch.stack(mc_runs).mean(dim=0)

            loss = metric(sampled_images, target)
            loss_all.append(loss.item())
            print('L1 loss: ' + str(loss.item()))

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
path        = './results_brain/'
A_to_B_PATH = os.path.join(SAVE_DIR, 'best_model.pt')
if not os.path.exists(path):
    os.makedirs(path)


# ── TensorBoard ───────────────────────────────────────────────────────────────
run_name = f"brain_{time.strftime('%Y%m%d_%H%M%S')}"
writer   = SummaryWriter(log_dir=os.path.join(LOG_DIR, run_name))
print(f'TensorBoard run : {run_name}')
print(f'Monitor with    : tensorboard --logdir ./runs --port 6006')


# ── Training loop ─────────────────────────────────────────────────────────────
N_EPOCHS           = 500
best_loss          = 1
global_step        = 0
train_loss_history = []
test_loss_history  = []

# Uncomment to resume from checkpoint
# A_to_B_model.load_state_dict(torch.load(A_to_B_PATH), strict=False)

for epoch in range(0, N_EPOCHS):
    print('Epoch:', epoch)
    start_time = time.time()
    avg_train, global_step = train(A_to_B_model, optimizer, train_loader1,
                                   train_loss_history, epoch, writer, global_step)
    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
    if epoch % 10 == 0:
        average_loss = evaluate(A_to_B_model, epoch, path, val_loader1, best_loss, writer)
        if average_loss < best_loss:
            print('Save the latest best model')
            torch.save(A_to_B_model.state_dict(), A_to_B_PATH)
            best_loss = average_loss

writer.close()
print('Training complete')
