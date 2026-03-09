from torchvision import transforms
from torch.utils.data import Dataset
import torch

from datasets.data_augmentation import random_transform

def percentile_minmax(t, low=1.0, high=99.0, max_samples=100000):
    """
    t: Tensor (C, H, W)
    max_samples: nombre màxim de píxels utilitzats per estimar els percentils
    """
    C = t.shape[0]
    out = torch.zeros_like(t)

    for c in range(C):
        channel = t[c]
        flat = channel.flatten()

        if flat.numel() > max_samples:
            idx = torch.randint(0, flat.numel(), (max_samples,), device=flat.device)
            flat = flat[idx]

        p_low = torch.quantile(flat, low / 100.0)
        p_high = torch.quantile(flat, high / 100.0)

        channel = (channel - p_low) / (p_high - p_low + 1e-8)
        channel = torch.clamp(channel, 0.0, 1.0)

        out[c] = channel

    return out

class BaseDataset(Dataset):
    def __init__(self, cfg=None, train=True):
        self.train = train

        self.sensor_like_augmentation = random_transform

        if cfg and "norm" in cfg:
            norm_type = cfg["norm"]

            if norm_type == "min-max":
                # per-image min-max normalization
                tfs = [transforms.ToTensor()]
                if not cfg.get("traditional_augmentation", False) and not cfg.get("sensor_like_augmentation", False):
                    tfs.append(lambda t: (t - t.min()) / (t.max() - t.min() + 1e-8))
                if cfg.get("traditional_augmentation", False):
                    tfs.append(transforms.RandomHorizontalFlip(p=0.5))
                    tfs.append(transforms.RandomVerticalFlip(p=0.5))
                self.transforms = transforms.Compose(tfs)

                self.transforms_percentile = transforms.Compose([
                    transforms.ToTensor(),
                    lambda t: percentile_minmax(t, low=1.0, high=99.0)
                ])

                self.traditional_augmentation = transforms.Compose([
                    transforms.RandomApply([
                        transforms.ColorJitter(brightness=0.3, contrast=0.3)
                    ], p=0.7),

                    transforms.RandomApply([
                        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
                    ], p=0.5),

                    transforms.RandomErasing(p=0.3, scale=(0.02, 0.08)),

                    lambda t: (t - t.min()) / (t.max() - t.min() + 1e-8),
                ])

            elif norm_type == "mean-std":
                mean = cfg.get("mean", None)
                std = cfg.get("std", None)
                assert mean is not None and std is not None, \
                    "[ERROR] 'mean' and 'std' must be set when using 'mean-std' normalization."

                # Ensure lists
                if not isinstance(mean, list):
                    mean = list(mean) if hasattr(mean, "__iter__") and not isinstance(mean, str) else [mean]
                if not isinstance(std, list):
                    std = list(std) if hasattr(std, "__iter__") and not isinstance(std, str) else [std]

                tfs = [transforms.ToTensor()]
                if not cfg.get("traditional_augmentation", False) and not cfg.get("sensor_like_augmentation", False):
                    tfs.append(transforms.Normalize(mean=mean, std=std))
                elif cfg.get("traditional_augmentation", False):
                    tfs.append(transforms.RandomHorizontalFlip(p=0.5))
                    tfs.append(transforms.RandomVerticalFlip(p=0.5))
                self.transforms = transforms.Compose(tfs)

                self.transforms_percentile = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                ])

                self.traditional_augmentation = transforms.Compose([
                    transforms.RandomApply([
                        transforms.ColorJitter(brightness=0.3, contrast=0.3)
                    ], p=0.7),

                    transforms.RandomApply([
                        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
                    ], p=0.5),

                    transforms.RandomErasing(p=0.3, scale=(0.02, 0.08)),

                    transforms.Normalize(mean=mean, std=std),
                ])
            
            else:
                print(f"[WARNING] Unknown norm '{norm_type}'. No normalization applied "
                      f"to the {'train' if self.train else 'validation'} data.", flush=True)
                self.transforms = transforms.ToTensor()
        else:
            print(f"[WARNING] No normalization is being applied to the "
                  f"{'train' if self.train else 'validation'} input data.", flush=True)
            self.transforms = transforms.ToTensor()
