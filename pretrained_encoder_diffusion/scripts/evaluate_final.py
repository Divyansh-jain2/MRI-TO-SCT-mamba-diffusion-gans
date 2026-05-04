import torch
import numpy as np
import nibabel as nib
import glob
import os
import time
import csv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from natsort import natsorted
from monai.inferers import SlidingWindowInferer
from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from diffusion.Create_diffusion import *
from diffusion.resampler import *
from network.Diffusion_model_transformer import *

# ── Config ────────────────────────────────────────────────────────────────────
patch_size   = (64, 64, 4)
CT_CLIP      = (-1024, 1650)
SAVE_DIR     = './checkpoints_brain_baseline'
RESULTS_DIR  = './results_final/'
LOG_DIR      = './runs'
DATA_ROOT    = '/home/teaching/Desktop/Munish_Synthetic_CT/mc_ddpm_data/brain_npy'
device       = torch.device("cuda:0")
metric       = torch.nn.L1Loss()
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── TensorBoard ───────────────────────────────────────────────────────────────
run_name = f"final_eval_{time.strftime('%Y%m%d_%H%M%S')}"
writer   = SummaryWriter(log_dir=os.path.join(LOG_DIR, run_name))
print(f'TensorBoard run : {run_name}')
print(f'Monitor with    : tensorboard --logdir ./runs --port 6006')

# ── Load model ────────────────────────────────────────────────────────────────
num_channels          = 64
attention_resolutions = "32,16,8"
channel_mult          = (1, 2, 3, 4)
num_heads             = [4, 4, 8, 16]
window_size           = [[4,4,4],[4,4,4],[4,4,2],[4,4,2]]
num_res_blocks        = [2, 2, 2, 2]
sample_kernel         = ([2,2,2],[2,2,1],[2,2,1],[2,2,1]),
attention_ds          = [int(r) for r in attention_resolutions.split(",")]

A_to_B_model = SwinVITModel(
    image_size=patch_size, in_channels=2, model_channels=num_channels,
    out_channels=2, dims=3, sample_kernel=sample_kernel,
    num_res_blocks=num_res_blocks, attention_resolutions=tuple(attention_ds),
    dropout=0, channel_mult=channel_mult, num_classes=None,
    use_checkpoint=False, use_fp16=False, num_heads=num_heads,
    window_size=window_size, num_head_channels=64, num_heads_upsample=-1,
    use_scale_shift_norm=True, resblock_updown=False,
    use_new_attention_order=False,
).to(device)

A_to_B_model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'best_model.pt')))
A_to_B_model.eval()
print('Model loaded')

# ── Diffusion (full 50 steps) ─────────────────────────────────────────────────
diffusion = create_gaussian_diffusion(
    steps=1000, learn_sigma=True, sigma_small=False,
    noise_schedule='linear', use_kl=False, predict_xstart=False,
    rescale_timesteps=True, rescale_learned_sigmas=True,
    timestep_respacing=[50],
)

def diffusion_sampling(condition, model):
    return diffusion.p_sample_loop(
        model,
        (condition.shape[0], 1,
         condition.shape[2], condition.shape[3], condition.shape[4]),
        condition=condition, clip_denoised=True,
    )

inferer = SlidingWindowInferer(patch_size, 12, overlap=0.5, mode='gaussian')

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(gt_hu, sct_hu):
    mae      = np.mean(np.abs(gt_hu - sct_hu))
    rmse     = np.sqrt(np.mean((gt_hu - sct_hu) ** 2))
    data_range = CT_CLIP[1] - CT_CLIP[0]
    psnr_val = psnr(gt_hu, sct_hu, data_range=data_range)
    ssim_vals = [ssim(gt_hu[:, :, z], sct_hu[:, :, z], data_range=data_range)
                 for z in range(gt_hu.shape[2])]
    ssim_val = np.mean(ssim_vals)
    return mae, rmse, psnr_val, ssim_val

def norm01(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

def save_figure(mr_np, ct_hu, sct_hu, err_np, sample_idx, mae, rmse, psnr_val, ssim_val):
    ax_mid  = mr_np.shape[2] // 2
    cor_mid = mr_np.shape[1] // 2
    sag_mid = mr_np.shape[0] // 2

    slices = {
        'Axial':    (mr_np[:, :, ax_mid],   ct_hu[:, :, ax_mid],   sct_hu[:, :, ax_mid],   err_np[:, :, ax_mid]),
        'Coronal':  (mr_np[:, cor_mid, :],  ct_hu[:, cor_mid, :],  sct_hu[:, cor_mid, :],  err_np[:, cor_mid, :]),
        'Sagittal': (mr_np[sag_mid, :, :],  ct_hu[sag_mid, :, :],  sct_hu[sag_mid, :, :],  err_np[sag_mid, :, :]),
    }

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('black')
    fig.suptitle(
        f'Sample {sample_idx+1}  |  MAE={mae:.2f} HU   RMSE={rmse:.2f} HU   PSNR={psnr_val:.2f} dB   SSIM={ssim_val:.4f}',
        fontsize=14, fontweight='bold', color='white'
    )
    gs = gridspec.GridSpec(3, 4, hspace=0.05, wspace=0.05)
    col_titles = ['MRI Input', 'Ground Truth CT', 'Synthetic CT', 'Absolute Error']

    for row, (plane, (mr_s, ct_s, sct_s, err_s)) in enumerate(slices.items()):
        for col, (img, cmap, vmin, vmax, title) in enumerate([
            (np.rot90(mr_s),  'gray', mr_s.min(), mr_s.max(), col_titles[0]),
            (np.rot90(ct_s),  'gray', -200,        1000,       col_titles[1]),
            (np.rot90(sct_s), 'gray', -200,        1000,       col_titles[2]),
            (np.rot90(err_s), 'hot',  0,           200,        col_titles[3]),
        ]):
            ax = fig.add_subplot(gs[row, col])
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
            ax.axis('off')
            if row == 0:
                ax.set_title(title, fontsize=12, pad=8, color='white')
            if col == 0:
                ax.set_ylabel(plane, fontsize=11, rotation=90, labelpad=10, color='white')
                ax.yaxis.set_label_position('left')
                ax.axis('on')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines[:].set_visible(False)
            if col == 3:
                cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cb.ax.yaxis.set_tick_params(color='white')
                plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')

    out_path = RESULTS_DIR + f'result_sample{sample_idx:03d}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f'  Figure saved : {out_path}')

# ── Run on test set ───────────────────────────────────────────────────────────
test_files = natsorted(glob.glob(DATA_ROOT + '/imagesTs/*.npy'))
print(f'Found {len(test_files)} test volumes')

all_l1, all_mae, all_rmse, all_psnr, all_ssim = [], [], [], [], []

for i, f in enumerate(test_files):
    print(f'\nSample {i+1}/{len(test_files)}')
    data   = np.load(f)
    mr_vol = torch.from_numpy(data[0][np.newaxis][np.newaxis]).float().to(device)
    ct_vol = torch.from_numpy(data[1][np.newaxis][np.newaxis]).float().to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            sct = inferer(mr_vol, diffusion_sampling, A_to_B_model)

    l1_loss = metric(sct, ct_vol).item()
    all_l1.append(l1_loss)

    lo, hi  = CT_CLIP
    ct_hu   = (data[1] + 1.0) / 2.0 * (hi - lo) + lo
    sct_hu  = (sct[0, 0].cpu().numpy() + 1.0) / 2.0 * (hi - lo) + lo
    err_np  = np.abs(sct_hu - ct_hu)

    mae, rmse, psnr_val, ssim_val = compute_metrics(ct_hu, sct_hu)
    all_mae.append(mae); all_rmse.append(rmse)
    all_psnr.append(psnr_val); all_ssim.append(ssim_val)

    print(f'  L1={l1_loss:.6f}  MAE={mae:.2f} HU  RMSE={rmse:.2f} HU  PSNR={psnr_val:.2f} dB  SSIM={ssim_val:.4f}')

    # TensorBoard scalars
    writer.add_scalar("Test/L1_per_sample",   l1_loss,  i)
    writer.add_scalar("Test/MAE_per_sample",  mae,      i)
    writer.add_scalar("Test/RMSE_per_sample", rmse,     i)
    writer.add_scalar("Test/PSNR_per_sample", psnr_val, i)
    writer.add_scalar("Test/SSIM_per_sample", ssim_val, i)

    # TensorBoard images
    mid = sct_hu.shape[-1] // 2
    writer.add_image(f"Test/MRI_input_{i:03d}",      norm01(data[0][:, :, mid])[None], 0)
    writer.add_image(f"Test/CT_groundtruth_{i:03d}", norm01(ct_hu[:, :, mid])[None],   0)
    writer.add_image(f"Test/CT_synthetic_{i:03d}",   norm01(sct_hu[:, :, mid])[None],  0)

    # Save .nii.gz
    nib.save(nib.Nifti1Image(sct_hu, np.eye(4)),
             RESULTS_DIR + f'sct_test_sample{i:03d}.nii.gz')

    # Save PNG figure
    save_figure(data[0], ct_hu, sct_hu, err_np, i, mae, rmse, psnr_val, ssim_val)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f'\n{"="*55}')
print(f'  FINAL RESULTS ON {len(test_files)} TEST SAMPLES')
print(f'{"="*55}')
print(f'  MAE  : {np.mean(all_mae):.2f} ± {np.std(all_mae):.2f} HU')
print(f'  RMSE : {np.mean(all_rmse):.2f} ± {np.std(all_rmse):.2f} HU')
print(f'  PSNR : {np.mean(all_psnr):.2f} ± {np.std(all_psnr):.2f} dB')
print(f'  SSIM : {np.mean(all_ssim):.4f} ± {np.std(all_ssim):.4f}')
print(f'{"="*55}')

writer.add_scalar("Test/Mean_MAE",  np.mean(all_mae),  0)
writer.add_scalar("Test/Mean_RMSE", np.mean(all_rmse), 0)
writer.add_scalar("Test/Mean_PSNR", np.mean(all_psnr), 0)
writer.add_scalar("Test/Mean_SSIM", np.mean(all_ssim), 0)
writer.add_scalar("Test/Mean_L1",   np.mean(all_l1),   0)
writer.close()

# ── Save metrics CSV ──────────────────────────────────────────────────────────
csv_path = RESULTS_DIR + 'metrics.csv'
with open(csv_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['sample', 'L1', 'MAE_HU', 'RMSE_HU', 'PSNR_dB', 'SSIM'])
    for i in range(len(test_files)):
        w.writerow([i, f'{all_l1[i]:.6f}', f'{all_mae[i]:.4f}',
                    f'{all_rmse[i]:.4f}', f'{all_psnr[i]:.4f}', f'{all_ssim[i]:.4f}'])
    w.writerow(['mean', f'{np.mean(all_l1):.6f}', f'{np.mean(all_mae):.4f}',
                f'{np.mean(all_rmse):.4f}', f'{np.mean(all_psnr):.4f}', f'{np.mean(all_ssim):.4f}'])
    w.writerow(['std', '', f'{np.std(all_mae):.4f}',
                f'{np.std(all_rmse):.4f}', f'{np.std(all_psnr):.4f}', f'{np.std(all_ssim):.4f}'])

print(f'  Metrics CSV : {csv_path}')
print(f'  Results     : {RESULTS_DIR}')
