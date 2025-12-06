from torchvision import transforms
from torch.utils.data import Dataset
import torch


class BaseDataset(Dataset):
    def __init__(self, cfg=None, train=True):
        self.train = train

        if cfg and "norm" in cfg:
            norm_type = cfg["norm"]

            if norm_type == "min-max":
                # per-image min-max normalization
                self.transforms = transforms.Compose([
                    transforms.ToTensor(),
                    lambda t: (t - t.min()) / (t.max() - t.min() + 1e-8)  # avoid /0
                ])
                self.data_augmentation = transforms.Compose([
                    # transforms.ToTensor(),
                    transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3)], p=0.7),
                    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
                    lambda t: (t - t.min()) / (t.max() - t.min() + 1e-8)  # safe normalization
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

                self.transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                ])
                self.data_augmentation = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3)], p=0.7),
                    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
                    transforms.Normalize(mean=mean, std=std)
                ])
            else:
                print(f"[WARNING] Unknown norm '{norm_type}'. No normalization applied "
                      f"to the {'train' if self.train else 'validation'} data.", flush=True)
                self.transforms = transforms.ToTensor()
        else:
            print(f"[WARNING] No normalization is being applied to the "
                  f"{'train' if self.train else 'validation'} input data.", flush=True)
            self.transforms = transforms.ToTensor()
