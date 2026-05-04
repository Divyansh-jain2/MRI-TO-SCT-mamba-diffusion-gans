import torch
import numpy as np
import nibabel as nib
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from natsort import natsorted
from monai.inferers import SlidingWindowInferer
from diffusion.Create_diffusion import *
from diffusion.resampler import *
from network.Diffusion_model_transformer import *

# ── Config ────────────────────────────────────────────────────────────────────
patch_size   = (64, 64, 4)
CT_CLIP      = (-1024, 1650)
SAVE_DIR     = './checkpoints_brain'
RESULTS_DIR  = './results_final/'
DATA_ROOT    = '/home/teaching/Desktop/Munish_Synthetic_CT/mc_ddpm_data/brain_npy'
device       = torch.device("cuda:0")
metric       = torch.nn.L1Loss()
NUM_SAMPLES  = 2       # ← only run 2 samples to save time
os.makedirs(RESULTS_DIR, exist_ok=True)

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

# ── Diffusion ─────────────────────────────────────────────────────────────────
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

# ── Run inference ─────────────────────────────────────────────────────────────
test_files = natsorted(glob.glob(DATA_ROOT + '/imagesTs/*.npy'))[:NUM_SAMPLES]
print(f'Running inference on {len(test_files)} samples...')

for i, f in enumerate(test_files):
    print(f'Sample {i+1}/{len(test_files)}...')
    data   = np.load(f)
    mr_vol = torch.from_numpy(data[0][np.newaxis][np.newaxis]).float().to(device)
    ct_vol = torch.from_numpy(data[1][np.newaxis][np.newaxis]).float().to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            sct = inferer(mr_vol, diffusion_sampling, A_to_B_model)

    loss = metric(sct, ct_vol)
    print(f'  L1 loss: {loss.item():.6f}')

    # De-normalise CT to HU for display
    lo, hi   = CT_CLIP
    mr_np    = data[0]
    ct_np    = (data[1] + 1.0) / 2.0 * (hi - lo) + lo
    sct_np   = (sct[0, 0].cpu().numpy() + 1.0) / 2.0 * (hi - lo) + lo
    err_np   = np.abs(sct_np - ct_np)

    # Save .nii.gz
    nib.save(nib.Nifti1Image(sct_np, np.eye(4)),
             RESULTS_DIR + f'sct_sample{i:03d}.nii.gz')

    # ── Figure: axial, coronal, sagittal slices ───────────────────────────
    ax_mid  = mr_np.shape[2] // 2   # axial
    cor_mid = mr_np.shape[1] // 2   # coronal
    sag_mid = mr_np.shape[0] // 2   # sagittal

    slices = {
        'Axial':    (mr_np[:, :, ax_mid],   ct_np[:, :, ax_mid],   sct_np[:, :, ax_mid],   err_np[:, :, ax_mid]),
        'Coronal':  (mr_np[:, cor_mid, :],  ct_np[:, cor_mid, :],  sct_np[:, cor_mid, :],  err_np[:, cor_mid, :]),
        'Sagittal': (mr_np[sag_mid, :, :],  ct_np[sag_mid, :, :],  sct_np[sag_mid, :, :],  err_np[sag_mid, :, :]),
    }

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'Sample {i+1}  |  L1 = {loss.item():.4f}', fontsize=16, fontweight='bold')
    gs  = gridspec.GridSpec(3, 4, hspace=0.05, wspace=0.05)

    col_titles = ['MRI Input', 'Ground Truth CT', 'Synthetic CT', 'Absolute Error']

    for row, (plane, (mr_s, ct_s, sct_s, err_s)) in enumerate(slices.items()):
        for col, (img, cmap, vmin, vmax, title) in enumerate([
            (np.rot90(mr_s),  'gray',    mr_s.min(),   mr_s.max(),   col_titles[0]),
            (np.rot90(ct_s),  'gray',    -200,         1000,         col_titles[1]),
            (np.rot90(sct_s), 'gray',    -200,         1000,         col_titles[2]),
            (np.rot90(err_s), 'hot',     0,            200,          col_titles[3]),
        ]):
            ax = fig.add_subplot(gs[row, col])
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
            ax.axis('off')
            if row == 0:
                ax.set_title(title, fontsize=12, pad=8)
            if col == 0:
                ax.set_ylabel(plane, fontsize=11, rotation=90, labelpad=10)
                ax.yaxis.set_label_position('left')
                ax.axis('on')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines[:].set_visible(False)
            if col == 3:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    out_path = RESULTS_DIR + f'result_sample{i:03d}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f'  Figure saved: {out_path}')

print(f'\nDone. Results in {RESULTS_DIR}')