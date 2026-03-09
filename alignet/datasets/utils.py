import random
import numpy as np
from matplotlib.path import Path

import torch
from torch.utils.data import DataLoader, RandomSampler

from torchvision import transforms
import torchvision.transforms.functional as F

from utils.transforms import STN
from utils.registry import DATASET_REGISTRY

from .base_dataset import percentile_minmax

_SHARED_CACHE = {}

def build_shared_cache(file_paths, resample_fn):
    for p in file_paths:
        key = str(p)
        if key in _SHARED_CACHE:
            continue  # already loaded, skip
        print(f"Loading {key}", flush=True)
        arr = resample_fn(p)                     # NumPy H×W×C
        t   = torch.from_numpy(arr).permute(2,0,1)  # C×H×W
        t = percentile_minmax(t, 1, 99) #t = (t - t.min()) / (t.max() - t.min())
        t.share_memory_()                          # mark storage shared
        _SHARED_CACHE[key] = t
    print(f"All images from {file_paths} preloaded.", flush=True)


def get_shared_cache():
    return _SHARED_CACHE


def create_dataloader(dataset_cfg, img_size, batch_size, num_workers, train=True):
    dataset_class = DATASET_REGISTRY.get(dataset_cfg["type"])
    dataset = dataset_class(root_path=dataset_cfg["path"], img_size=img_size, train=train, cfg=dataset_cfg)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers, pin_memory=True, prefetch_factor=12, persistent_workers=True)


class PairedRandomCrop:
    def __init__(self, size, contained=False):
        self.size = (size, size)  # e.g., (height, width)
        self.contained = contained
        self.limit = 100
    
    def __call__(self, img1, img2, bbox=None):
        # Sample crop parameters once
        i, j, h, w = transforms.RandomCrop.get_params(img1, output_size=self.size)

        all_inside = False
        if self.contained:
            assert bbox is not None, "bbox must not be None when contained=True"
            bbox_crop = np.array([[j, i], [j+w, i], [j+w, i+h], [j, i+h]])
            bbox_path = Path(bbox)
            all_inside = np.all(bbox_path.contains_points(bbox_crop))
            count = 1
            while not all_inside:
                i, j, h, w = transforms.RandomCrop.get_params(img1, output_size=self.size)
                bbox_crop = np.array([[j, i], [j+w, i], [j+w, i+h], [j, i+h]])
                all_inside = np.all(bbox_path.contains_points(bbox_crop))
                count += 1
                if count >= self.limit:
                    break

        img1 = F.crop(img1, i, j, h, w)
        img2 = F.crop(img2, i, j, h, w)
        return img1, img2, i, j, h, w, all_inside


class PairedCenterCrop:
    def __init__(self, size):
        self.size = (size, size)  # e.g., (height, width)
        self.trans = transforms.CenterCrop(self.size)
    
    def __call__(self, img1, img2, bbox=None):
        image_height, image_width = img1.shape[-2:]
        h, w = self.size

        img1 = self.trans(img1)
        img2 = self.trans(img2)

        i = (image_height - h) // 2
        j = (image_width - w) // 2
        return img1, img2, i, j, h, w, True
