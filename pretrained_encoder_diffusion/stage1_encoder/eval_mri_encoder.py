import os
import glob
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from natsort import natsorted
from network.mri_encoder import MRIAutoencoder
from skimage.metrics import structural_similarity as ssim
from monai.inferers import SlidingWindowInferer

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG (Ensure these match your pretraining setup)
# ═══════════════════════════════════════════════════════════════════════════════
DATA_ROOT   = '/home/teaching/Desktop/Munish_Synthetic_CT/mc_ddpm_data/brain_npy/imagesTs/' # Change to imagesVal if Ts not present
CKPT_PATH   = './checkpoints_mri_encoder/pretrain_checkpoint.pt'
SAVE_DIR    = './eval_pretrain_results'
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Architecture Parameters
ENC_CHANNELS = (64, 128, 192, 256)
GLOBAL_DIM   = 256
WINDOW_SIZE  = (4, 4, 4)
NUM_HEADS    = (4, 4, 8, 8)
POOL_KERNEL  = (2, 2, 1)

os.makedirs(SAVE_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════
def calculate_psnr(pred, target, data_range=2.0):
    mse = np.mean((pred - target) ** 2)
    if mse == 0: return 100.0
    return 20 * np.log10(data_range / np.sqrt(mse))

# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE SCRIPT
# ═══════════════════════════════════════════════════════════════════════════════
def run_evaluation():
    # 1. Load Model
    model = MRIAutoencoder(
        enc_channels=ENC_CHANNELS,
        global_dim=GLOBAL_DIM,
        window_size=WINDOW_SIZE,
        num_heads=NUM_HEADS,
        pool_kernel=POOL_KERNEL
    ).to(DEVICE)

    if not os.path.exists(CKPT_PATH):
        print(f"Error: Checkpoint not found at {CKPT_PATH}")
        return

    print(f"Loading checkpoint: {CKPT_PATH}")
    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # 2. Find Files
    files = natsorted(glob.glob(os.path.join(DATA_ROOT, "*.npy")))
    if not files:
        DATA_ROOT_ALT = DATA_ROOT.replace('imagesTs', 'imagesVal')
        files = natsorted(glob.glob(os.path.join(DATA_ROOT_ALT, "*.npy")))
        if files:
            print(f"No test files in imagesTs, using validation files from {DATA_ROOT_ALT}")
        else:
            print(f"Error: No .npy files found in {DATA_ROOT}")
            return

    print(f"Found {len(files)} files for evaluation.")

    # Inferer setup
    def model_wrapper(x):
        with torch.cuda.amp.autocast():
            _, ct_pred, _ = model(x)
        return ct_pred

    inferer = SlidingWindowInferer(
        roi_size=(64, 64, 4), 
        sw_batch_size=4, 
        overlap=0.25, 
        mode='gaussian'
    )

    l1_losses, psnr_values, ssim_values = [], [], []

    for i, fpath in enumerate(files):
        fname = os.path.basename(fpath).replace('.npy', '')
        data = np.load(fpath) # [2, H, W, D]
        
        mri_vol = torch.from_numpy(data[0][None, None]).float().to(DEVICE)
        ct_gt   = data[1]

        # 3. Predict via Sliding Window
        with torch.no_grad():
            print(f"  [{i+1}/{len(files)}] Processing {fname}...")
            ct_pred_tensor = inferer(mri_vol, model_wrapper)
            ct_pred = ct_pred_tensor.squeeze().cpu().numpy()

        # 4. Calculate Metrics
        l1 = np.mean(np.abs(ct_pred - ct_gt))
        ps = calculate_psnr(ct_pred, ct_gt)
        mid = ct_gt.shape[-1] // 2
        ss = ssim(ct_pred[..., mid], ct_gt[..., mid], data_range=2.0)

        l1_losses.append(l1)
        psnr_values.append(ps)
        ssim_values.append(ss)

        print(f"      L1: {l1:.4f} | PSNR: {ps:.2f} | SSIM: {ss:.4f}")

        # 5. Visualization (Axial Middle Slice)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(data[0][..., mid], cmap='gray'); axes[0].set_title("MRI Input")
        axes[1].imshow(ct_gt[..., mid],   cmap='gray'); axes[1].set_title("CT Ground Truth")
        axes[2].imshow(ct_pred[..., mid], cmap='gray'); axes[2].set_title("Predicted CT (Pretrain)")
        for ax in axes: ax.axis('off')
        
        plt.savefig(os.path.join(SAVE_DIR, f"{fname}_result.png"))
        plt.close()

    # 6. Final Summary
    print("\n" + "="*40)
    print("FINAL PRETRAIN EVALUATION SUMMARY")
    print("="*40)
    print(f"Avg L1 Loss (normalized): {np.mean(l1_losses):.4f}")
    print(f"Avg PSNR:                {np.mean(psnr_values):.2f} dB")
    print(f"Avg SSIM (middle slice): {np.mean(ssim_values):.4f}")
    print("="*40)
    print(f"Visual results saved to: {SAVE_DIR}")

if __name__ == "__main__":
    run_evaluation()
