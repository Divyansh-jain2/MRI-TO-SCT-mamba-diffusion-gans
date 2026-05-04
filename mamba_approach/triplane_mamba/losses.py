"""
Improved loss functions for TriMamba-UNet MRI-to-CT synthesis.

Changes from original:
  - REMOVED useless AFP loss (frozen random encoder = noise)
  - ADDED GradientLoss (3D edge preservation for bone boundaries)
  - ADDED FocalFrequencyLoss (high-frequency detail recovery)
  - UPDATED CompoundLoss with better staging + deep supervision support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 1. Weighted HU-aware MAE
# ─────────────────────────────────────────────
class WeightedMAELoss(nn.Module):
    """
    Weighted MAE: bone (HU>300) ×3, soft tissue ×1.5, air ×0.5.
    Data normalized to [-1, 1] using clip range [-1024, 1500].
    """
    def __init__(self):
        super().__init__()
        self.bone_thresh = self._hu_to_norm(300)
        self.air_thresh  = self._hu_to_norm(-700)

    def _hu_to_norm(self, hu):
        return (hu + 1024) / (1500 + 1024) * 2 - 1

    def forward(self, pred, target):
        weight = torch.ones_like(target)
        weight[target > self.bone_thresh] = 3.0
        weight[(target > self.air_thresh) & (target <= self.bone_thresh)] = 1.5
        weight[target <= self.air_thresh] = 0.5

        abs_err = torch.abs(pred - target)
        weighted_err = weight * abs_err
        return weighted_err.sum() / (weight.sum() + 1e-5)


# ─────────────────────────────────────────────
# 2. SSIM Loss (3D)
# ─────────────────────────────────────────────
class SSIMLoss(nn.Module):
    """3D SSIM loss computed with gaussian kernel."""
    def __init__(self, window_size=7, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def _gaussian_kernel_3d(self, size, sigma, device):
        coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel = g[:, None, None] * g[None, :, None] * g[None, None, :]
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        return kernel

    def forward(self, pred, target):
        device = pred.device
        kernel = self._gaussian_kernel_3d(self.window_size, self.sigma, device)
        kernel = kernel.to(dtype=pred.dtype)
        pad = self.window_size // 2

        mu1 = F.conv3d(pred, kernel, padding=pad)
        mu2 = F.conv3d(target, kernel, padding=pad)
        mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

        sigma1_sq = F.conv3d(pred * pred, kernel, padding=pad) - mu1_sq
        sigma2_sq = F.conv3d(target * target, kernel, padding=pad) - mu2_sq
        sigma12   = F.conv3d(pred * target, kernel, padding=pad) - mu1_mu2

        # Clamp to avoid negative variance from numerical issues
        sigma1_sq = sigma1_sq.clamp(min=0)
        sigma2_sq = sigma2_sq.clamp(min=0)

        ssim_map = (
            (2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)
        ) / (
            (mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2)
        )
        return 1.0 - ssim_map.mean()


# ─────────────────────────────────────────────
# 3. Gradient Loss (3D edge preservation)
# ─────────────────────────────────────────────
class GradientLoss(nn.Module):
    """
    Penalizes difference in spatial gradients between pred and target.
    Preserves sharp edges — critical for bone boundaries in CT.
    """
    def forward(self, pred, target):
        # Gradients along D, H, W
        dx_p = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
        dx_t = target[:, :, 1:, :, :] - target[:, :, :-1, :, :]

        dy_p = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
        dy_t = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]

        dz_p = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
        dz_t = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]

        loss_d = torch.abs(dx_p - dx_t).mean()
        loss_h = torch.abs(dy_p - dy_t).mean()
        loss_w = torch.abs(dz_p - dz_t).mean()

        return (loss_d + loss_h + loss_w) / 3.0


# ─────────────────────────────────────────────
# 4. Focal Frequency Loss
# ─────────────────────────────────────────────
class FocalFrequencyLoss(nn.Module):
    """
    Focal Frequency Loss: adaptively weights hard-to-synthesize frequencies.
    Forces the model to recover high-frequency details that L1/L2 losses
    tend to smooth out (bone edges, fine textures).
    
    Reference: Jiang et al., "Focal Frequency Loss for Image Reconstruction 
    and Synthesis", ICCV 2021.
    """
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, pred, target):
        # 3D FFT (cast to float32 as torch.fft does not support bfloat16)
        pred_fp32 = pred.float()
        target_fp32 = target.float()
        
        pred_fft = torch.fft.rfftn(pred_fp32, dim=(-3, -2, -1))
        target_fft = torch.fft.rfftn(target_fp32, dim=(-3, -2, -1))

        # Magnitude difference
        diff = torch.abs(pred_fft - target_fft)

        # Focal weight: emphasize frequencies where error is large
        weight = diff.detach() ** self.alpha
        weight = weight / (weight.mean() + 1e-5)  # normalize

        return (weight * diff).mean().to(pred.dtype)


# ─────────────────────────────────────────────
# 5. Compound Staged Loss (V2) — NaN-safe with gradual ramp-up
# ─────────────────────────────────────────────
class CompoundLossV2(nn.Module):
    """
    Improved staged loss for TriMamba-UNet with GRADUAL ramp-up.
    
    The old version hard-switched losses at epoch 50 which, combined with
    CosineAnnealing LR restart, caused a gradient shock → NaN at epoch 52.
    
    Fix: All new losses are linearly ramped up over 30 epochs.
    
    Staging (with ramp):
        epoch < 50:      L = wMAE only (warmup)
        epoch 50-80:     L = wMAE + ramp(SSIM) + ramp(Grad) + ramp(DS)
        epoch 80-200:    L = wMAE + SSIM + Grad + DS (full weight)
        epoch 200-230:   L = above + ramp(FocalFreqLoss)
        epoch >= 230:    L = all losses at full weight
    """
    RAMP_EPOCHS = 30  # gradually introduce each loss over this many epochs

    def __init__(self, w_mae=1.0, w_ssim=0.2, w_grad=0.05,
                 w_ffl=0.1, w_ds2=0.4, w_ds3=0.2):
        super().__init__()
        self.w_mae  = w_mae
        self.w_ssim = w_ssim
        self.w_grad = w_grad
        self.w_ffl  = w_ffl
        self.w_ds2  = w_ds2
        self.w_ds3  = w_ds3

        self.wmae = WeightedMAELoss()
        self.ssim = SSIMLoss()
        self.grad = GradientLoss()
        self.ffl  = FocalFrequencyLoss(alpha=1.0)

    @staticmethod
    def _ramp(epoch, start_epoch, ramp_len=30):
        """Linear ramp from 0→1 over ramp_len epochs starting at start_epoch."""
        if epoch < start_epoch:
            return 0.0
        return min(1.0, (epoch - start_epoch) / ramp_len)

    def _single_scale_loss(self, pred, target, epoch, ramp_ssim=1.0, ramp_ffl=0.0):
        """Compute loss at a single scale with ramp factors."""
        loss_wmae = self.wmae(pred, target)
        loss_dict = {'wMAE': loss_wmae.item()}
        total = self.w_mae * loss_wmae

        if ramp_ssim > 0:
            loss_ssim = self.ssim(pred, target)
            loss_grad = self.grad(pred, target)
            total = total + ramp_ssim * (self.w_ssim * loss_ssim + self.w_grad * loss_grad)
            loss_dict['SSIM'] = loss_ssim.item()
            loss_dict['Grad'] = loss_grad.item()

        if ramp_ffl > 0:
            loss_ffl = self.ffl(pred, target)
            total = total + ramp_ffl * self.w_ffl * loss_ffl
            loss_dict['FFL'] = loss_ffl.item()

        loss_dict['total'] = total.item()
        return total, loss_dict

    def forward(self, pred, target, epoch, aux_preds=None):
        # Compute ramp factors (gradual introduction of losses)
        ramp_ssim = self._ramp(epoch, 50, self.RAMP_EPOCHS)   # 0→1 over ep 50-80
        ramp_ffl  = self._ramp(epoch, 200, self.RAMP_EPOCHS)  # 0→1 over ep 200-230
        ramp_ds   = self._ramp(epoch, 50, self.RAMP_EPOCHS)   # 0→1 over ep 50-80

        # Main loss
        main_loss, loss_dict = self._single_scale_loss(
            pred, target, epoch, ramp_ssim=ramp_ssim, ramp_ffl=ramp_ffl)
        total = main_loss

        # Deep supervision (also ramped)
        if aux_preds is not None and ramp_ds > 0:
            aux2, aux3 = aux_preds

            target_ds2 = F.interpolate(target, size=aux2.shape[2:],
                                       mode='trilinear', align_corners=False)
            target_ds3 = F.interpolate(target, size=aux3.shape[2:],
                                       mode='trilinear', align_corners=False)

            aux2_loss, _ = self._single_scale_loss(
                aux2, target_ds2, epoch, ramp_ssim=ramp_ssim, ramp_ffl=ramp_ffl)
            aux3_loss, _ = self._single_scale_loss(
                aux3, target_ds3, epoch, ramp_ssim=ramp_ssim, ramp_ffl=ramp_ffl)

            total = total + ramp_ds * (self.w_ds2 * aux2_loss + self.w_ds3 * aux3_loss)
            loss_dict['DS2'] = aux2_loss.item()
            loss_dict['DS3'] = aux3_loss.item()
            loss_dict['total'] = total.item()

        return total, loss_dict


# ─────────────────────────────────────────────
# Legacy CompoundLoss (kept for segmamba/umamba)
# ─────────────────────────────────────────────
class CompoundLoss(nn.Module):
    """Legacy compound loss for backward compatibility."""
    def __init__(self, w1=1.0, w2=0.1, w3=0.1, device='cuda'):
        super().__init__()
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.wmae = WeightedMAELoss()
        self.ssim = SSIMLoss()
        self.grad = GradientLoss()

    def forward(self, pred, target, epoch):
        loss_wmae = self.wmae(pred, target)
        if epoch < 100:
            return loss_wmae, {
                'wMAE': loss_wmae.item(), 'SSIM': 0.0, 'Grad': 0.0,
                'total': loss_wmae.item()
            }
        else:
            loss_ssim = self.ssim(pred, target)
            loss_grad = self.grad(pred, target)
            total = self.w1 * loss_wmae + self.w2 * loss_grad + self.w3 * loss_ssim
            return total, {
                'wMAE': loss_wmae.item(), 'SSIM': loss_ssim.item(),
                'Grad': loss_grad.item(), 'total': total.item()
            }