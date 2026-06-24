"""Augmentation pipelines for ALIGNet training.

Three strategies used in the ablation study:
  - no-augmentation  : images are normalized but not augmented (cfg has neither flag set)
  - traditional      : standard photometric augmentations (cfg: traditional_augmentation: true)
  - sensor-like      : physics-inspired multi-modal simulation, the novel contribution
                       proposed in the paper (cfg: sensor_like_augmentation: true)
"""

import random
import numpy as np
import cv2
from PIL import Image

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.transforms.functional import rgb_to_grayscale
import albumentations as A


# ---------------------------------------------------------------------------
# SAR building blocks
# ---------------------------------------------------------------------------

class SpeckleNoise(torch.nn.Module):
    def __init__(self, variance_range=(0.05, 0.3)):
        super().__init__()
        self.variance_range = variance_range

    def forward(self, x):
        var = torch.empty(1).uniform_(*self.variance_range).item()
        noise = torch.randn_like(x) * var + 1.0
        return x * noise


class LogCompression(torch.nn.Module):
    def __init__(self, a_range=(5.0, 50.0)):
        super().__init__()
        self.a_range = a_range

    def forward(self, x):
        a = torch.empty(1).uniform_(*self.a_range).item()
        return torch.log1p(a * x) / torch.log1p(torch.tensor(a))


class RadiometricGradient(torch.nn.Module):
    def __init__(self, strength=0.3):
        super().__init__()
        self.strength = strength

    def forward(self, x):
        _, h, w = x.shape
        grad = torch.linspace(1 - self.strength, 1 + self.strength, w)
        return x * grad.view(1, 1, w).to(x.device)


class Banding(torch.nn.Module):
    def __init__(self, strength=0.05):
        super().__init__()
        self.strength = strength

    def forward(self, x):
        _, h, _ = x.shape
        bands = torch.sin(torch.linspace(0, 20, h)).view(1, h, 1)
        return x + self.strength * bands.to(x.device)


class SARResolutionDegradation(torch.nn.Module):
    def __init__(self, scale_range=(2, 8), blur_sigma_range=(0.8, 2.5)):
        super().__init__()
        self.scale_range = scale_range
        self.blur_sigma_range = blur_sigma_range

    def forward(self, x):
        _, h, w = x.shape
        s = int(torch.randint(*self.scale_range, (1,)).item())
        sigma = torch.empty(1).uniform_(*self.blur_sigma_range).item()
        x = F.gaussian_blur(x, kernel_size=7, sigma=sigma)
        x = F.resize(x, (h // s, w // s), interpolation=F.InterpolationMode.BILINEAR, antialias=True)
        x = F.resize(x, (h, w), interpolation=F.InterpolationMode.BILINEAR, antialias=True)
        return x


class SARLikeTransform:
    """Gradient-based SAR edge simulation (more complex than the simple SAR pipeline)."""

    def __init__(
        self,
        speckle_shape_range=(0.8, 1.8),
        speckle_scale_range=(0.6, 1.4),
        gradient_weight_range=(0.4, 1.0),
        smoothing_prob=0.7,
        log_compression=True,
    ):
        self.speckle_shape_range = speckle_shape_range
        self.speckle_scale_range = speckle_scale_range
        self.gradient_weight_range = gradient_weight_range
        self.smoothing_prob = smoothing_prob
        self.log_compression = log_compression

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            if img.shape[0] == 3:
                img = rgb_to_grayscale(img)
            img = img.squeeze().cpu().numpy()
        else:
            img = np.array(img)
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)

        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx**2 + gy**2)

        weight = random.uniform(*self.gradient_weight_range)
        sar = weight * grad + (1 - weight) * img

        if random.random() < 0.5:
            sar = cv2.dilate(sar, np.ones((3, 3), np.uint8))

        shape = random.uniform(*self.speckle_shape_range)
        scale = random.uniform(*self.speckle_scale_range)
        noise = np.random.gamma(shape, scale, sar.shape)
        sar = sar * np.clip(noise, 0, 3)

        if self.log_compression:
            sar = np.log1p(sar)
            sar = sar / (sar.mean() + 1e-6)

        if random.random() < self.smoothing_prob:
            sar = cv2.bilateralFilter(sar.astype(np.float32), 5, 50, 50)

        p1, p99 = np.percentile(sar, (1, 99))
        sar = np.clip(sar, p1, p99)
        sar = (sar - p1) / (p99 - p1 + 1e-6)

        return torch.from_numpy(sar).unsqueeze(0).float()


# ---------------------------------------------------------------------------
# DEM / DSM building blocks
# ---------------------------------------------------------------------------

class RGBToHeightSeed(torch.nn.Module):
    def forward(self, x):
        if not torch.is_tensor(x):
            x = T.ToTensor()(x)
        r, g, b = x[0], x[1], x[2]
        h = 0.299 * r + 0.587 * g + 0.114 * b
        return h.unsqueeze(0)


class TerrainSmoothing(torch.nn.Module):
    def __init__(self, kernel_range=(15, 51)):
        super().__init__()
        self.kernel_range = kernel_range

    def forward(self, x):
        k = int(torch.randint(*self.kernel_range, (1,)))
        if k % 2 == 0:
            k += 1
        return F.gaussian_blur(x, kernel_size=k, sigma=k / 6)


class HeightScaling(torch.nn.Module):
    def __init__(self, scale_range=(0.5, 3.0)):
        super().__init__()
        self.scale_range = scale_range

    def forward(self, x):
        x = (x - x.mean()) / (x.std() + 1e-6)
        s = torch.empty(1).uniform_(*self.scale_range).item()
        return x * s


class SurfaceObjects(torch.nn.Module):
    def __init__(self, density=0.002, height_range=(0.2, 1.0)):
        super().__init__()
        self.density = density
        self.height_range = height_range

    def forward(self, x):
        _, h, w = x.shape
        mask = torch.rand(h, w) < self.density
        heights = torch.empty(h, w).uniform_(*self.height_range)
        return x + mask.float().unsqueeze(0) * heights.unsqueeze(0)


class DirectionalSlope(torch.nn.Module):
    def __init__(self, strength=0.3):
        super().__init__()
        self.strength = strength

    def forward(self, x):
        dx = torch.gradient(x, dim=2)[0]
        dy = torch.gradient(x, dim=1)[0]
        slope = torch.sqrt(dx**2 + dy**2)
        return x + self.strength * slope


# ---------------------------------------------------------------------------
# Low-resolution building blocks
# ---------------------------------------------------------------------------

class ResolutionDegradation(torch.nn.Module):
    def __init__(self, scale_range=(2, 8), mode="bilinear"):
        super().__init__()
        self.scale_range = scale_range

    def forward(self, x):
        _, h, w = x.shape
        s = int(torch.randint(*self.scale_range, (1,)))
        x = F.resize(x, (h // s, w // s), interpolation=F.InterpolationMode.BILINEAR)
        x = F.resize(x, (h, w), interpolation=F.InterpolationMode.BILINEAR)
        return x


class SensorBlur(torch.nn.Module):
    def __init__(self, sigma_range=(0.5, 2.5)):
        super().__init__()
        self.sigma_range = sigma_range

    def forward(self, x):
        sigma = torch.empty(1).uniform_(*self.sigma_range).item()
        return F.gaussian_blur(x, kernel_size=7, sigma=sigma)


class Quantization(torch.nn.Module):
    def __init__(self, levels_range=(64, 1024)):
        super().__init__()
        self.levels_range = levels_range

    def forward(self, x):
        levels = int(torch.randint(*self.levels_range, (1,)))
        x_min, x_max = x.min(), x.max()
        x = torch.round((x - x_min) / (x_max - x_min + 1e-6) * levels) / levels
        return x * (x_max - x_min) + x_min


# ---------------------------------------------------------------------------
# Thermal building block
# ---------------------------------------------------------------------------

class RGBThermalAug:
    """
    Pseudo-thermal augmentation. Accepts PIL Image or torch.Tensor. Returns (1, H, W).

    Adapted from:
    Author: OnderT (Onder Tutun)
    Source: https://github.com/OnderT/XoFTR
    File: src/utils/augment.py
    
    Modifications: 
    - Added support for PIL Images and torch.Tensors in `_to_numpy_rgb`.
    - Refactored `__call__` to process a single image directly.
    - Changed output from a 3-channel RGB numpy array to a 1-channel torch.Tensor.
    """

    def __init__(self):
        self.blur = A.Blur(p=0.7, blur_limit=(2, 4))
        self.hsv = A.HueSaturationValue(
            p=0.9,
            val_shift_limit=(-30, 30),
            hue_shift_limit=(-90, 90),
            sat_shift_limit=(-30, 30),
        )
        self.w_0 = np.pi * 2 / 3
        self.w_r = np.pi / 2
        self.theta_r = np.pi / 2

    def _to_numpy_rgb(self, x):
        if isinstance(x, torch.Tensor):
            x = x.permute(1, 2, 0).cpu().numpy() if x.ndim == 3 else x.cpu().numpy()
        if isinstance(x, Image.Image):
            x = np.array(x)
        if x.ndim == 2:
            x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
        if x.dtype != np.uint8:
            x = (x * 255).astype(np.uint8) if x.max() <= 1 else x.astype(np.uint8)
        return x

    def __call__(self, x):
        image = self._to_numpy_rgb(x)
        image = self.hsv(image=image)["image"]
        image = self.blur(image=image)["image"]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        image = image / 255.0 - 0.5
        phase = np.pi / 2 + np.random.randn() * self.theta_r
        w = self.w_0 + abs(np.random.randn()) * self.w_r
        image = np.cos(image * w + phase)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        return torch.from_numpy(image.astype(np.float32)).unsqueeze(0)


# ---------------------------------------------------------------------------
# RandomModalityTransform: combines all sensor-like transforms
# ---------------------------------------------------------------------------

class RandomModalityTransform(torch.nn.Module):
    def __init__(self, transforms, probs=None):
        super().__init__()
        self.transforms = transforms
        if probs is None:
            self.probs = torch.ones(len(transforms)) / len(transforms)
        else:
            self.probs = torch.tensor(probs, dtype=torch.float)

    def forward(self, x):
        idx = torch.multinomial(self.probs, 1).item()
        return self.transforms[idx](x)


# ---------------------------------------------------------------------------
# Individual sensor-like pipelines
# ---------------------------------------------------------------------------

_safe_norm = lambda t: (t - t.min()) / (t.max() - t.min() + 1e-8)

_identity_transform = T.Compose([
    T.Grayscale(num_output_channels=1),
    _safe_norm,
])

_sar_transform = T.Compose([
    T.Grayscale(num_output_channels=1),
    T.ColorJitter(contrast=(1.5, 3.0)),
    LogCompression(a_range=(10.0, 40.0)),
    SpeckleNoise(variance_range=(0.1, 0.4)),
    T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.3, 1.0))], p=0.4),
    T.RandomApply([RadiometricGradient(strength=0.25)], p=0.5),
    T.RandomApply([SARResolutionDegradation(scale_range=(2, 8))], p=0.6),
    T.RandomApply([Banding(0.05)], p=0.3),
    T.Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
    _safe_norm,
])

_sar_edges_transform = T.Compose([SARLikeTransform()])

_thermal_transform = T.Compose([RGBThermalAug()])

_dem_transform = T.Compose([
    RGBToHeightSeed(),
    TerrainSmoothing(kernel_range=(31, 71)),
    HeightScaling(scale_range=(0.5, 2.0)),
    T.Lambda(lambda x: torch.clamp(x, -3, 3)),
    _safe_norm,
])

_dsm_transform = T.Compose([
    RGBToHeightSeed(),
    TerrainSmoothing(kernel_range=(21, 51)),
    SurfaceObjects(density=0.003, height_range=(0.3, 1.5)),
    DirectionalSlope(strength=0.2),
    HeightScaling(scale_range=(1.0, 3.0)),
    T.Lambda(lambda x: torch.clamp(x, -4, 4)),
    _safe_norm,
])

_resolution_transform = T.Compose([
    T.Grayscale(num_output_channels=1),
    T.RandomApply([SensorBlur()], p=0.7),
    ResolutionDegradation(scale_range=(2, 10)),
    T.RandomApply([Quantization(levels_range=(128, 2048))], p=0.5),
    _safe_norm,
])

_easy_transform = T.Compose([
    T.Grayscale(num_output_channels=1),
    T.RandomApply([T.ColorJitter(brightness=0.3, contrast=0.3)], p=0.7),
    T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
    _safe_norm,
])


# ---------------------------------------------------------------------------
# Public API: sensor-like augmentation (the novel contribution)
# ---------------------------------------------------------------------------

sensor_like_augmentation = RandomModalityTransform(
    transforms=[
        _identity_transform,   # identity (15%)
        _sar_transform,        # SAR intensity simulation (10%)
        _sar_edges_transform,  # SAR edge simulation (10%)
        _thermal_transform,    # pseudo-thermal (30%)
        _dem_transform,        # pseudo-DEM (5%)
        _dsm_transform,        # pseudo-DSM (5%)
        _resolution_transform, # low-resolution degradation (20%)
        _easy_transform,       # subtle photometric change (5%)
    ],
    probs=[0.15, 0.10, 0.10, 0.30, 0.05, 0.05, 0.20, 0.05],
)


# ---------------------------------------------------------------------------
# Public API: traditional augmentation (baseline comparison)
# ---------------------------------------------------------------------------

def build_traditional_augmentation(norm_type, mean=None, std=None):
    """Return a Compose pipeline for traditional augmentation (ablation baseline).

    The final normalization step is included so the output is always in the same
    numeric range regardless of the preceding augmentations.

    Args:
        norm_type: "min-max" or "mean-std".
        mean, std: required when norm_type == "mean-std".
    """
    base_aug = [
        T.RandomApply([T.ColorJitter(brightness=0.3, contrast=0.3)], p=0.7),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
        T.RandomErasing(p=0.3, scale=(0.02, 0.08)),
    ]

    if norm_type == "min-max":
        base_aug.append(T.Lambda(lambda t: (t - t.min()) / (t.max() - t.min() + 1e-8)))
    elif norm_type == "mean-std":
        assert mean is not None and std is not None, \
            "[ERROR] 'mean' and 'std' must be provided for mean-std traditional augmentation."
        base_aug.append(T.Normalize(mean=mean, std=std))

    return T.Compose(base_aug)
