import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

import random
import numpy as np
import cv2
from PIL import Image
import albumentations as A
from torchvision.transforms.functional import rgb_to_grayscale


# SAR

class SpeckleNoise(torch.nn.Module):
    def __init__(self, variance_range=(0.05, 0.3)):
        super().__init__()
        self.variance_range = variance_range

    def forward(self, x):
        # x: (C, H, W), expected in [0, 1]
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
        grad = grad.view(1, 1, w).to(x.device)
        return x * grad

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
        # x: (1, H, W)
        _, h, w = x.shape

        # Random degradation factor
        s = int(torch.randint(*self.scale_range, (1,)).item())

        # --- Step 1: sensor blur (PSF / multilook effect)
        sigma = torch.empty(1).uniform_(*self.blur_sigma_range).item()
        x = F.gaussian_blur(x, kernel_size=7, sigma=sigma)

        # --- Step 2: downsample
        h_low, w_low = h // s, w // s
        x = F.resize(
            x,
            size=(h_low, w_low),
            interpolation=F.InterpolationMode.BILINEAR,
            antialias=True,
        )

        # --- Step 3: upsample back
        x = F.resize(
            x,
            size=(h, w),
            interpolation=F.InterpolationMode.BILINEAR,
            antialias=True,
        )

        return x


# DEM / DSM

class RGBToHeightSeed(torch.nn.Module):
    def forward(self, x):
        # x: PIL or Tensor
        if not torch.is_tensor(x):
            x = T.ToTensor()(x)
        # luminance proxy
        r, g, b = x[0], x[1], x[2]
        h = 0.299 * r + 0.587 * g + 0.114 * b
        return h.unsqueeze(0)  # (1, H, W)

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
        x = x - x.mean()
        x = x / (x.std() + 1e-6)
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

# SAR edges

class SARLikeTransform:
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
        """
        img: PIL image or tensor [C,H,W] in range [0,1] or [0,255]
        returns: tensor [1,H,W] SAR-like
        """

        # --- to numpy grayscale ---
        if isinstance(img, torch.Tensor):
            if img.shape[0] == 3:
                img = rgb_to_grayscale(img)
            img = img.squeeze().cpu().numpy()
        else:
            img = np.array(img)
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        img = img.astype(np.float32)

        # normalize
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)

        # --- gradient magnitude (structure response) ---
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx**2 + gy**2)

        weight = random.uniform(*self.gradient_weight_range)
        sar = weight * grad + (1 - weight) * img

        # --- dilation → double bounce effect ---
        if random.random() < 0.5:
            sar = cv2.dilate(sar, np.ones((3,3), np.uint8))

        # --- multiplicative speckle noise ---
        shape = random.uniform(*self.speckle_shape_range)
        scale = random.uniform(*self.speckle_scale_range)
        noise = np.random.gamma(shape, scale, sar.shape)
        noise = np.clip(noise, 0, 3)
        sar = sar * noise

        # --- log compression ---
        if self.log_compression:
            sar = np.log1p(sar)
            sar = sar / (sar.mean() + 1e-6)

        # --- optional smoothing (radar diffusion look) ---
        if random.random() < self.smoothing_prob:
            sar = sar.astype(np.float32)
            sar = cv2.bilateralFilter(sar, 5, 50, 50)

        # normalize output
        # sar = (sar - sar.min()) / (sar.max() - sar.min() + 1e-6)
        p1, p99 = np.percentile(sar, (1, 99))
        sar = np.clip(sar, p1, p99)
        sar = (sar - p1) / (p99 - p1 + 1e-6) 

        return torch.from_numpy(sar).unsqueeze(0).float()


# LOW-RESOLUTION

class ResolutionDegradation(torch.nn.Module):
    def __init__(self, scale_range=(2, 8), mode="bilinear"):
        super().__init__()
        self.scale_range = scale_range
        self.mode = mode

    def forward(self, x):
        _, h, w = x.shape
        s = int(torch.randint(*self.scale_range, (1,)))
        h_low, w_low = h // s, w // s

        x = F.resize(x, (h_low, w_low), interpolation=F.InterpolationMode.BILINEAR)
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
        x = (x - x_min) / (x_max - x_min + 1e-6)
        x = torch.round(x * levels) / levels
        return x * (x_max - x_min) + x_min


# THERMAL AUGMENTATION

class RGBThermalAug:
    """
    Pseudo-thermal augmentation.
    Accepts PIL Image or torch.Tensor.
    Returns torch tensor (1, H, W).
    """

    def __init__(self):
        self.blur = A.Blur(p=0.7, blur_limit=(2, 4))
        self.hsv = A.HueSaturationValue(
            p=0.9,
            val_shift_limit=(-30, 30),
            hue_shift_limit=(-90, 90),
            sat_shift_limit=(-30, 30),
        )

        # cosine transform params
        self.w_0 = np.pi * 2 / 3
        self.w_r = np.pi / 2
        self.theta_r = np.pi / 2

    # ---------- utilities ----------

    def _to_numpy_rgb(self, x):
        """Convert PIL/Tensor -> numpy RGB uint8"""
        if isinstance(x, torch.Tensor):
            if x.ndim == 3 and x.shape[0] in (1, 3):
                # C,H,W -> H,W,C
                x = x.permute(1, 2, 0).cpu().numpy()
            else:
                x = x.cpu().numpy()

        if isinstance(x, Image.Image):
            x = np.array(x)

        if x.ndim == 2:
            x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)

        if x.dtype != np.uint8:
            x = (x * 255).astype(np.uint8) if x.max() <= 1 else x.astype(np.uint8)

        return x

    def _to_tensor_gray(self, img):
        """numpy HxW -> torch 1xHxW float32"""
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        return torch.from_numpy(img).unsqueeze(0)

    # ---------- augmentation ----------

    def augment_pseudo_thermal(self, image):

        image = self._to_numpy_rgb(image)

        # HSV + blur
        image = self.hsv(image=image)["image"]
        image = self.blur(image=image)["image"]

        # RGB -> grayscale
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)

        # normalize to (-0.5, 0.5)
        image = image / 255.0 - 0.5

        # random cosine transform
        phase = np.pi / 2 + np.random.randn() * self.theta_r
        w = self.w_0 + abs(np.random.randn()) * self.w_r

        image = np.cos(image * w + phase)

        # min-max normalize to 0–1
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        return self._to_tensor_gray(image)

    # ---------- callable ----------

    def __call__(self, x):
        return self.augment_pseudo_thermal(x)


# Combine transformations

class RandomModalityTransform(torch.nn.Module):
    def __init__(self, transforms, probs=None):
        """
        transforms: list of callables (including identity)
        probs: list of probabilities (same length as transforms)
        """
        super().__init__()
        self.transforms = transforms

        if probs is None:
            self.probs = torch.ones(len(transforms)) / len(transforms)
        else:
            self.probs = torch.tensor(probs, dtype=torch.float)

    def forward(self, x):
        idx = torch.multinomial(self.probs, 1).item()
        return self.transforms[idx](x)


identity = T.Compose([
    T.Grayscale(num_output_channels=1),
    # T.ToTensor(),  # [0, 1]
    lambda t: (t - t.min()) / (t.max() - t.min() + 1e-8)  # safe normalization
    
])

sar_transform = T.Compose([
    # 1. RGB → grayscale (SAR is intensity)
    T.Grayscale(num_output_channels=1),

    # 2. Strong contrast shaping
    T.ColorJitter(contrast=(1.5, 3.0)),

    # 3. Convert to tensor
    # T.ToTensor(),  # [0, 1]

    # 4. Log compression (big dynamic range)
    LogCompression(a_range=(10.0, 40.0)),

    # 5. Multiplicative speckle noise
    SpeckleNoise(variance_range=(0.1, 0.4)),

    # 6. Optional smoothing (multi-look effect)
    T.RandomApply(
        [T.GaussianBlur(kernel_size=3, sigma=(0.3, 1.0))],
        p=0.4
    ),

    # 7. Radiometric gradient
    T.RandomApply(
        [RadiometricGradient(strength=0.25)],
        p=0.5
    ),

    # 8: Random resolution degradation
    T.RandomApply(
        [SARResolutionDegradation(scale_range=(2, 8))],
        p=0.6
    ),

    # 9. Approximate radiometric banding artifacts
    T.RandomApply([Banding(0.05)], p=0.3),

    # 10. Clip back to valid range
    T.Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),

    # 11. Normalize
    lambda t: (t - t.min()) / (t.max() - t.min() + 1e-8)  # safe normalization
    
])

dem_transform = T.Compose([
    RGBToHeightSeed(),
    TerrainSmoothing(kernel_range=(31, 71)),
    HeightScaling(scale_range=(0.5, 2.0)),
    T.Lambda(lambda x: torch.clamp(x, -3, 3)),
    lambda t: (t - t.min()) / (t.max() - t.min() + 1e-8)  # safe normalization
])


dsm_transform = T.Compose([
    RGBToHeightSeed(),
    TerrainSmoothing(kernel_range=(21, 51)),
    SurfaceObjects(density=0.003, height_range=(0.3, 1.5)),
    DirectionalSlope(strength=0.2),
    HeightScaling(scale_range=(1.0, 3.0)),
    T.Lambda(lambda x: torch.clamp(x, -4, 4)),
    lambda t: (t - t.min()) / (t.max() - t.min() + 1e-8)  # safe normalization
])

sar_edges_transform = T.Compose([
    SARLikeTransform()
])

thermal_transform = T.Compose([
    RGBThermalAug()
])

resolution_transform = T.Compose([
    # T.ToTensor(),
    T.Grayscale(num_output_channels=1),
    T.RandomApply([SensorBlur()], p=0.7),
    ResolutionDegradation(scale_range=(2, 10)),
    T.RandomApply([Quantization(levels_range=(128, 2048))], p=0.5),
    lambda t: (t - t.min()) / (t.max() - t.min() + 1e-8)  # safe normalization
])

easy_transform = T.Compose([
    # T.ToTensor(),
    T.Grayscale(num_output_channels=1),
    T.RandomApply([T.ColorJitter(brightness=0.3, contrast=0.3)], p=0.7),
    T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
    lambda t: (t - t.min()) / (t.max() - t.min() + 1e-8)  # safe normalization
])


random_transform = RandomModalityTransform(
    transforms=[
        identity,                   # do nothing
        sar_transform,              # SAR
        sar_edges_transform,        # SAR edges
        thermal_transform,          # THERMAL
        dem_transform,              # DEM
        dsm_transform,              # DSM
        resolution_transform,       # low-res
        easy_transform,             # small color and noise change
    ],
    probs=[
        0.15,  # no transform
        0.10,  # SAR
        0.10,  # SAR edges
        0.30,  # THERMAL
        0.05,  # DEM
        0.05,  # DSM
        0.20,  # low-res
        0.05,
    ]
)


# probs=[
#         0.15,  # no transform
#         0.10,  # SAR
#         0.10,  # SAR edges
#         0.30,  # THERMAL
#         0.05,  # DEM
#         0.05,  # DSM
#         0.20,  # low-res
#         0.05,
#     ]


# probs=[
#         0.15,  # no transform
#         0.15,  # SAR
#         0.15,  # SAR edges
#         0.10,  # DEM
#         0.10,  # DSM
#         0.25,  # low-res
#         0.10,
#     ]