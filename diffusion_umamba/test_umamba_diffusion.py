import os
import glob
import torch
import numpy as np
import nibabel as nib
from natsort import natsorted
from torch.utils.data import Dataset, DataLoader
from monai.inferers import SlidingWindowInferer
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
import time
from tqdm import tqdm

# Required imports for model and diffusion
from models import UMamba
from diffusion.Create_diffusion import *
from diffusion.resampler import *

DATA_ROOT    = '/DATA/Munish_Synthetic_CT/mc_ddpm_data/brain_npy'
SAVE_DIR     = './checkpoints_brain_umamba_diffusion'
RESULTS_DIR  = './results_test_brain_umamba_diffusion'
CT_CLIP      = (-1024, 1650)
patch_size   = (64, 64, 4)

os.makedirs(RESULTS_DIR, exist_ok=True)

class CustomTestDataset(Dataset):
    def __init__(self, imgs_path):
        self.files = natsorted(glob.glob(imgs_path + "*.npy"), key=lambda y: y.lower())
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .npy files found in {imgs_path}")
        print(f"Found {len(self.files)} preprocessed test volumes in {imgs_path}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data   = np.load(self.files[idx])
        mr_vol = data[0]   # MRI  [-1, 1]
        ct_vol = data[1]   # CT   [-1, 1]

        img_tensor   = torch.from_numpy(mr_vol[np.newaxis]).float()
        label_tensor = torch.from_numpy(ct_vol[np.newaxis]).float()

        return img_tensor, label_tensor, self.files[idx]

def main():
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

    print("Initializing Diffusion model...")
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Initializing UMamba model...")
    A_to_B_model = UMamba(
        in_ch=2,
        out_ch=2,
        base_ch=64,
        is_diffusion=True,
        strides=((2,2,2), (2,2,1), (2,2,1))
    ).to(device)

    # Note: Using best_model.pt or latest_model.pt
    model_path = os.path.join(SAVE_DIR, 'best_model.pt')
    if not os.path.exists(model_path):
        print(f"Warning: {model_path} not found. Trying latest_model.pt...")
        model_path = os.path.join(SAVE_DIR, 'latest_model.pt')
        if not os.path.exists(model_path):
            print(f"Error: Could not find any model checkpoint in {SAVE_DIR}.")
            return

    print(f"Loading weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    A_to_B_model.load_state_dict(checkpoint['model_state'])
    A_to_B_model.eval()

    img_num  = 16
    overlap  = 0.25  
    inferer  = SlidingWindowInferer(patch_size, img_num, overlap=overlap, mode='constant')

    def diffusion_sampling(condition, model):
        sampled_images = diffusion.p_sample_loop(
            model,
            (condition.shape[0], 1, condition.shape[2], condition.shape[3], condition.shape[4]),
            condition=condition, clip_denoised=True,
        )
        return sampled_images

    testing_set  = CustomTestDataset(DATA_ROOT + '/imagesTs/')
    test_loader  = DataLoader(testing_set, batch_size=1, shuffle=False, num_workers=2)

    print(f"Starting testing on {len(test_loader)} samples...")
    all_psnr = []
    all_ssim = []

    with torch.no_grad():
        for i, (x1, y1, file_path) in enumerate(tqdm(test_loader, desc="Testing")):
            target    = y1.to(device)
            condition = x1.to(device)
            file_name = os.path.basename(file_path[0])

            # Use 3 MC runs for robust generation (can change to 1 for faster inference)
            mc_runs = []
            for _ in range(3):
                with torch.cuda.amp.autocast():
                    sampled_images = inferer(condition, diffusion_sampling, A_to_B_model)
                mc_runs.append(sampled_images)
            
            sampled_images = torch.stack(mc_runs).mean(dim=0)
            
            # Denormalize
            lo, hi = CT_CLIP
            sct_np = sampled_images[0, 0].cpu().numpy()
            sct_hu = (sct_np + 1.0) / 2.0 * (hi - lo) + lo
            
            gt_np = target[0, 0].cpu().numpy()
            gt_hu = (gt_np + 1.0) / 2.0 * (hi - lo) + lo
            
            # Save prediction
            out_nii_path = os.path.join(RESULTS_DIR, f'pred_{file_name.replace(".npy", ".nii.gz")}')
            nib.save(nib.Nifti1Image(sct_hu, np.eye(4)), out_nii_path)
            
            # Save GT for reference
            gt_nii_path = os.path.join(RESULTS_DIR, f'gt_{file_name.replace(".npy", ".nii.gz")}')
            nib.save(nib.Nifti1Image(gt_hu, np.eye(4)), gt_nii_path)

            dr = float(hi - lo)
            psnr_val = psnr_metric(gt_hu, sct_hu, data_range=dr)
            ssim_val = np.mean([ssim_metric(gt_hu[:,:,z], sct_hu[:,:,z], data_range=dr) for z in range(gt_hu.shape[2])])
            
            all_psnr.append(psnr_val)
            all_ssim.append(ssim_val)
            
            print(f"Sample {i+1} ({file_name}): PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")

    avg_psnr = np.mean(all_psnr)
    avg_ssim = np.mean(all_ssim)

    print("\n" + "="*50)
    print(" 🏁 TESTING COMPLETE")
    print("="*50)
    print(f"Average Test PSNR: {avg_psnr:.2f} dB")
    print(f"Average Test SSIM: {avg_ssim:.4f}")
    print(f"Predictions saved to: {RESULTS_DIR}")

if __name__ == '__main__':
    main()
