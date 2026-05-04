import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random


class BrainMRICTDataset(Dataset):
    """
    Dataset for brain MRI-to-CT synthesis.
    Each .npy file has shape (2, 192, 192, 96):
        channel 0 -> MRI  (already normalized to [-1, 1])
        channel 1 -> CT   (already normalized to [-1, 1])
    """

    def __init__(self, data_dir, patch_size=(64, 192, 192), mode='train'):
        """
        Args:
            data_dir: path to imagesTr / imagesVal / imagesTs folder
            patch_size: (D, H, W) patch to extract during training
            mode: 'train', 'val', or 'test'
        """
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.mode = mode

        self.files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith('.npy')
        ])
        print(f"[{mode}] Found {len(self.files)} files in {data_dir}")

    def __len__(self):
        return len(self.files)

    def _extract_patch(self, mri, ct):
        """
        Randomly extract a 3D patch.
        mri, ct: numpy arrays of shape (192, 192, 96) i.e. (H, W, D)
        Returns patch of shape patch_size = (D, H, W)
        """
        H, W, D = mri.shape
        pH, pW, pD = self.patch_size[1], self.patch_size[2], self.patch_size[0]

        # Random crop
        h_start = random.randint(0, max(H - pH, 0))
        w_start = random.randint(0, max(W - pW, 0))
        d_start = random.randint(0, max(D - pD, 0))

        mri_patch = mri[h_start:h_start+pH, w_start:w_start+pW, d_start:d_start+pD]
        ct_patch  = ct[h_start:h_start+pH,  w_start:w_start+pW,  d_start:d_start+pD]

        # Rearrange to (D, H, W)
        mri_patch = np.transpose(mri_patch, (2, 0, 1))
        ct_patch  = np.transpose(ct_patch,  (2, 0, 1))

        return mri_patch, ct_patch

    def __getitem__(self, idx):
        data = np.load(self.files[idx])  # (2, 192, 192, 96)

        mri = data[0]  # (192, 192, 96)
        ct  = data[1]  # (192, 192, 96)

        if self.mode == 'train':
            mri_patch, ct_patch = self._extract_patch(mri, ct)
        else:
            # For val/test use full volume, rearrange to (D, H, W)
            mri_patch = np.transpose(mri, (2, 0, 1))  # (96, 192, 192)
            ct_patch  = np.transpose(ct,  (2, 0, 1))

        # Add channel dim -> (1, D, H, W)
        mri_tensor = torch.from_numpy(mri_patch).unsqueeze(0).float()
        ct_tensor  = torch.from_numpy(ct_patch).unsqueeze(0).float()

        return mri_tensor, ct_tensor, self.files[idx]


def get_dataloaders(base_dir, patch_size=(64, 192, 192), batch_size=2, num_workers=4):
    """
    base_dir should contain: imagesTr/, imagesVal/, imagesTs/
    """
    train_dataset = BrainMRICTDataset(
        os.path.join(base_dir, 'imagesTr'), patch_size=patch_size, mode='train'
    )
    val_dataset = BrainMRICTDataset(
        os.path.join(base_dir, 'imagesVal'), patch_size=patch_size, mode='val'
    )
    test_dataset = BrainMRICTDataset(
        os.path.join(base_dir, 'imagesTs'), patch_size=patch_size, mode='test'
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader