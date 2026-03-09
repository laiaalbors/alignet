import random
import numpy as np
from matplotlib.path import Path

import torch
from torch.utils.data import DataLoader

from torchvision import transforms
import torchvision.transforms.functional as F

from utils.transforms import STN
from utils.registry import DATASET_REGISTRY


# ---------------------------------------------------------------------------
# Normalization utility (lives here to avoid circular imports with base_dataset)
# ---------------------------------------------------------------------------

def percentile_minmax(t, low=1.0, high=99.0, max_samples=100_000):
    """Per-channel percentile min-max normalization.

    Args:
        t: Tensor of shape (C, H, W).
        low, high: percentile bounds (e.g. 1.0 and 99.0).
        max_samples: max pixels sampled per channel for quantile estimation.
    """
    C = t.shape[0]
    out = torch.zeros_like(t)
    for c in range(C):
        channel = t[c].flatten()
        if channel.numel() > max_samples:
            idx = torch.randint(0, channel.numel(), (max_samples,), device=channel.device)
            channel_sample = channel[idx]
        else:
            channel_sample = channel
        p_low = torch.quantile(channel_sample, low / 100.0)
        p_high = torch.quantile(channel_sample, high / 100.0)
        out[c] = torch.clamp((t[c] - p_low) / (p_high - p_low + 1e-8), 0.0, 1.0)
    return out


# ---------------------------------------------------------------------------
# Shared in-memory cache (used by MDAS datasets)
# ---------------------------------------------------------------------------

_SHARED_CACHE = {}


def build_shared_cache(file_paths, resample_fn):
    """Pre-load images into a process-shared tensor cache.

    Args:
        file_paths: iterable of path-like objects.
        resample_fn: callable that takes a path and returns a NumPy H×W×C array.
    """
    for p in file_paths:
        key = str(p)
        if key in _SHARED_CACHE:
            continue
        print(f"Loading {key}", flush=True)
        arr = resample_fn(p)
        t = torch.from_numpy(arr).permute(2, 0, 1)  # C×H×W
        t = percentile_minmax(t, 1, 99)
        t.share_memory_()
        _SHARED_CACHE[key] = t
    print(f"Preloaded {len(file_paths)} images.", flush=True)


def get_shared_cache():
    return _SHARED_CACHE


# ---------------------------------------------------------------------------
# Dataloader factory
# ---------------------------------------------------------------------------

def create_dataloader(dataset_cfg, img_size, batch_size, num_workers, train=True):
    dataset_class = DATASET_REGISTRY.get(dataset_cfg["type"])
    dataset = dataset_class(root_path=dataset_cfg["path"], img_size=img_size, train=train, cfg=dataset_cfg)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=12,
        persistent_workers=True,
    )


# ---------------------------------------------------------------------------
# Paired crop helpers
# ---------------------------------------------------------------------------

class PairedRandomCrop:
    """Random crop applied identically to two images, optionally constrained
    so that the crop contains the entire transformed bounding box."""

    def __init__(self, size, contained=False):
        self.size = (size, size)
        self.contained = contained
        self.limit = 100

    def __call__(self, img1, img2, bbox=None):
        i, j, h, w = transforms.RandomCrop.get_params(img1, output_size=self.size)

        all_inside = False
        if self.contained:
            assert bbox is not None, "bbox must not be None when contained=True"
            bbox_path = Path(bbox)
            bbox_crop = np.array([[j, i], [j + w, i], [j + w, i + h], [j, i + h]])
            all_inside = np.all(bbox_path.contains_points(bbox_crop))
            count = 1
            while not all_inside:
                i, j, h, w = transforms.RandomCrop.get_params(img1, output_size=self.size)
                bbox_crop = np.array([[j, i], [j + w, i], [j + w, i + h], [j, i + h]])
                all_inside = np.all(bbox_path.contains_points(bbox_crop))
                count += 1
                if count >= self.limit:
                    break

        return F.crop(img1, i, j, h, w), F.crop(img2, i, j, h, w), i, j, h, w, all_inside


class PairedCenterCrop:
    """Center crop applied identically to two images."""

    def __init__(self, size):
        self.size = (size, size)
        self.trans = transforms.CenterCrop(self.size)

    def __call__(self, img1, img2, bbox=None):
        image_height, image_width = img1.shape[-2:]
        h, w = self.size
        i = (image_height - h) // 2
        j = (image_width - w) // 2
        return self.trans(img1), self.trans(img2), i, j, h, w, True
