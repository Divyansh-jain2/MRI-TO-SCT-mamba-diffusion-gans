import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 1. Weighted HU-aware MAE
# ─────────────────────────────────────────────
class WeightedMAELoss(nn.Module):
    """
    Weighted MAE that emphasizes bone (HU > 300), soft tissue, and
    de-emphasizes air/background (HU < -700).

    Since data is normalized to [-1, 1], we map HU thresholds:
        HU 300  -> normalized ~  0.545  (using clip range [-1024, 1500])
        HU -700 -> normalized ~ -0.318
    """

    def __init__(self):
        super().__init__()
        # Thresholds in normalized [-1, 1] space
        # Original HU range clipped to [-1024, 1500], then normalized to [-1,1]
        # norm = (HU - (-1024)) / (1500 - (-1024)) * 2 - 1
        self.bone_thresh = self._hu_to_norm(300)    #  ~0.545
        self.air_thresh  = self._hu_to_norm(-700)   # ~-0.318

    def _hu_to_norm(self, hu):
        return (hu + 1024) / (1500 + 1024) * 2 - 1

    def forward(self, pred, target):
        # Build weight map based on target intensities
        weight = torch.ones_like(target)
        weight[target > self.bone_thresh] = 3.0   # bone
        weight[(target > self.air_thresh) & (target <= self.bone_thresh)] = 1.5  # soft tissue
        weight[target <= self.air_thresh] = 0.5   # air/background

        abs_err = torch.abs(pred - target)
        weighted_err = weight * abs_err
        loss = weighted_err.sum() / (weight.sum() + 1e-8)
        return loss


# ─────────────────────────────────────────────
# 2. SSIM Loss
# ─────────────────────────────────────────────
class SSIMLoss(nn.Module):
    """3D SSIM loss computed on patches."""

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
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
        return kernel

    def forward(self, pred, target):
        # pred, target: (B, 1, D, H, W)
        device = pred.device
        kernel = self._gaussian_kernel_3d(self.window_size, self.sigma, device)

        pad = self.window_size // 2

        mu1 = F.conv3d(pred,   kernel, padding=pad)
        mu2 = F.conv3d(target, kernel, padding=pad)

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv3d(pred * pred,     kernel, padding=pad) - mu1_sq
        sigma2_sq = F.conv3d(target * target, kernel, padding=pad) - mu2_sq
        sigma12   = F.conv3d(pred * target,   kernel, padding=pad) - mu1_mu2

        ssim_map = (
            (2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)
        ) / (
            (mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2)
        )

        return 1.0 - ssim_map.mean()


# ─────────────────────────────────────────────
# 3. Simplified AFP Loss (Feature-space loss)
# ─────────────────────────────────────────────
class AFPLoss(nn.Module):
    """
    Anatomical Feature-Prioritized Loss.
    Uses a lightweight multi-scale feature extractor to compare
    predicted vs target in feature space (perceptual-style loss).
    Since TotalSegmentator is not easily embedded, we use a
    trainable multi-scale encoder as a proxy.
    """

    def __init__(self):
        super().__init__()
        # Simple multi-scale feature extractor (frozen after init)
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1),
            nn.InstanceNorm3d(16),
            nn.LeakyReLU(0.2),
            nn.Conv3d(16, 32, 3, stride=2, padding=1),
            nn.InstanceNorm3d(32),
            nn.LeakyReLU(0.2),
            nn.Conv3d(32, 64, 3, stride=2, padding=1),
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(0.2),
        )
        # Freeze encoder weights
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, pred, target):
        feat_pred   = self.encoder(pred)
        feat_target = self.encoder(target)
        return F.l1_loss(feat_pred, feat_target)


# ─────────────────────────────────────────────
# 4. Compound Staged Loss
# ─────────────────────────────────────────────
class CompoundLoss(nn.Module):
    """
    Staged loss from the paper:
        epoch < 100  : L = wMAE only
        epoch >= 100 : L = w1*wMAE + w2*AFP + w3*SSIM
    """

    def __init__(self, w1=1.0, w2=0.1, w3=0.1, device='cuda'):
        super().__init__()
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

        self.wmae = WeightedMAELoss()
        self.ssim = SSIMLoss()
        self.afp  = AFPLoss().to(device)

    def forward(self, pred, target, epoch):
        loss_wmae = self.wmae(pred, target)

        if epoch < 100:
            return loss_wmae, {
                'wMAE': loss_wmae.item(),
                'SSIM': 0.0,
                'AFP':  0.0,
                'total': loss_wmae.item()
            }
        else:
            loss_ssim = self.ssim(pred, target)
            loss_afp  = self.afp(pred, target)
            total = self.w1 * loss_wmae + self.w2 * loss_afp + self.w3 * loss_ssim
            return total, {
                'wMAE':  loss_wmae.item(),
                'SSIM':  loss_ssim.item(),
                'AFP':   loss_afp.item(),
                'total': total.item()
            }