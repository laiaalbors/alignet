import os
import time
import random
import itertools
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torchvision import io
from torchvision import transforms
import torchvision.transforms.functional as F

import rasterio

from datasets.base_dataset import BaseDataset
from datasets.utils import PairedRandomCrop, PairedCenterCrop
from utils.utils import normalize_affine_matrix, create_masks, adjust_affine_transformation, apply_normalized_affine_to_polygon
from utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register("DREAMDataset")
class DREAMDataset(BaseDataset):
    def __init__(self, root_path, img_size, train=True, cfg=None):
        super().__init__(cfg=cfg, train=train)

        if cfg is None or "mode" not in cfg:
            print("[WARNING] Mode not specified in cfg. Defaulting to 'mono'.", flush=True)
        self.mode = (cfg or {}).get("mode", "mono")
        self.homography = cfg.get("homography", "affine")
        self.radiometric_augmentation = (cfg or {}).get("rad_aug", False)

        self.train = train
        self.data_path = cfg['data_path']
        self.img_size = img_size
        
        self.label_to_idx = {'optique': 0, 'radar': 1, 'S1': 2, 'S2': 3, 'SRTM': 4}

        with open(root_path, "r") as f:
            zones = [line.strip() for line in f]
        
        self.file_paths = []
        self.labels = []
        if self.mode == "mono":
            for zone in zones:
                for mod in self.label_to_idx.keys():
                    folder_path = os.path.join(self.data_path, zone, mod)
                    files = [
                        os.path.join(folder_path, f) for f in os.listdir(folder_path)
                        if os.path.isfile(os.path.join(folder_path, f))
                    ]
                    self.file_paths.extend(files)
                    self.labels.extend([self.label_to_idx[mod]]*len(files))
        elif self.mode == "multi":
            label_permutations = list(itertools.permutations(self.label_to_idx.keys(), 2))
            for zone in zones:
                for mod1, mod2 in label_permutations:
                    folder_path1 = os.path.join(self.data_path, zone, mod1)
                    folder_path2 = os.path.join(self.data_path, zone, mod2)
                    files1 = [os.path.join(folder_path1, f) for f in os.listdir(folder_path1) if os.path.isfile(os.path.join(folder_path1, f))]
                    files2 = [os.path.join(folder_path2, f) for f in os.listdir(folder_path2) if os.path.isfile(os.path.join(folder_path2, f))]
                    self.file_paths.extend(zip(files1, files2))
                    self.labels.extend(zip([self.label_to_idx[mod1]]*len(files1), [self.label_to_idx[mod2]]*len(files2)))
        else:
            Exception(f"Mode {self.mode} not implemented. Choose between multi or mono.")
        
        self.paired_crop = PairedRandomCrop(img_size, contained=True) if train else PairedCenterCrop(img_size)

    def read_tif(self, file_path):
        with rasterio.Env(GDAL_CACHEMAX=64):
            with rasterio.open(file_path) as dataset:
                image = dataset.read(1) # Shape: (H, W)
        # Expand dimensions to (H, W, 1) and cast to float32
        return image[:, :, np.newaxis].astype(np.float32)

    def read_npy(self, file_path):
        """
        Reads a .npy saved as float32 with shape (H, W) (or (H, W, C)).
        Returns HWC float32 (so grayscale becomes (H, W, 1)).
        """
        arr = np.load(file_path, mmap_mode="r")  # memory-mapped read (low RAM)
        
        # Ensure float32 (avoid copying if already float32)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)

        # Convert to HWC
        if arr.ndim == 2:
            arr = arr[:, :, None]          # (H, W, 1)
        elif arr.ndim == 3:
            # assume already HWC
            pass
        else:
            raise ValueError(f"Unexpected npy shape {arr.shape} for {file_path}")

        # If you need a real in-memory array (some ops dislike memmap), uncomment:
        arr = np.ascontiguousarray(arr)

        return arr

    def __getitem__(self, index):
        if self.mode == "mono":
            path_1 = path_2 = self.file_paths[index]
            label_1 = label_2 = self.labels[index]
        elif self.mode == "multi":
            path_1, path_2 = self.file_paths[index]
            label_1, label_2 = self.labels[index]

        image_1 = self.read_npy(path_1[:-3]+'npy')
        image_2 = self.read_npy(path_2[:-3]+'npy')
                
        # Apply the transformations to the original image
        image_1 = self.transforms_percentile(image_1) if row['label'] == 'S1' else self.transforms(image_1)
        image_2 = self.transforms_percentile(image_2) if row['label2'] == 'S1' else self.transforms(image_2)
        image_depth, image_height, image_width = image_2.shape

        label_1 = torch.tensor(label_1, dtype=torch.long)
        label_2 = torch.tensor(label_2, dtype=torch.long)

        if self.homography == 'affine':
            # Dynamically generate a random affine transformation
            angle = random.uniform(-35, 35)  # Random rotation
            translate = (random.uniform(-20, 20), random.uniform(-20, 20))  # Random translation
            scale = random.uniform(0.8, 1.2)  # Random scaling
            shear = random.uniform(-10, 10)  # Random shearing

            # Apply affine transformation
            transformed_image_2 = F.affine(image_2, angle, translate, scale, [shear, shear])

            # Transformation matrix
            inverse_matrix = F._get_inverse_affine_matrix(
                center=(image_2.shape[2] / 2, image_2.shape[1] / 2),
                angle=angle,
                translate=translate,
                scale=scale,
                shear=[shear, shear]
            )
            inverse_matrix = torch.tensor(inverse_matrix, dtype=torch.float32)

            # Inverse transformation matrix
            inv_affine = torch.tensor([
                [inverse_matrix[0], inverse_matrix[1], inverse_matrix[2]],
                [inverse_matrix[3], inverse_matrix[4], inverse_matrix[5]],
                [0, 0, 1]
            ], dtype=torch.float32, device=inverse_matrix.device)
            forward_affine = torch.inverse(inv_affine)
            forward_matrix = forward_affine[:2, :].reshape(-1)

        elif self.homography == 'projective':
            transformed_image, forward_affine = apply_random_homography(image_2, max_displacement=80)
            while torch.isnan(forward_affine).any():
                transformed_image, forward_affine = apply_random_homography(image_2, max_displacement=80)
            forward_matrix = forward_affine.reshape(9)[:-1]
            inv_affine = torch.inverse(forward_affine)
            inv_affine = inv_affine / inv_affine[2,2]
            inverse_matrix = inv_affine.reshape(9)[:-1]
            assert not torch.isnan(forward_matrix).any(), "NaNs detected in forward_matrix!"
            assert not torch.isnan(inverse_matrix).any(), "NaNs detected in inverse_matrix!"

        # Normalize Affine Transformation Matrix
        forward_matrix_norm = normalize_affine_matrix(forward_matrix.squeeze(), image_width, image_height)
        inverse_matrix_norm = normalize_affine_matrix(inverse_matrix.squeeze(), image_width, image_height)

        # Crop images
        bbox_original = np.array([[0, 0], [image_2.shape[2], 0], [image_2.shape[2], image_2.shape[1]], [0, image_2.shape[1]]])
        bbox_transformed = apply_normalized_affine_to_polygon(bbox_original, forward_matrix_norm.numpy(), image_2.shape[1], image_2.shape[2])
        image_1_crop, transformed_image_2_crop, offset1, offset2, h, w, contained = self.paired_crop(image_1, transformed_image_2, bbox=bbox_transformed)
        _, crop_height, crop_width = image_1_crop.shape

        # Adjust transformation to crop size
        forward_matrix_crop = adjust_affine_transformation(forward_matrix, offset1, offset2)
        inverse_matrix_crop = adjust_affine_transformation(inverse_matrix, offset1, offset2)

        # Normalize Crop Affine Transformation Matrix
        forward_matrix_crop_norm = normalize_affine_matrix(forward_matrix_crop.squeeze(), crop_width, crop_height)
        inverse_matrix_crop_norm = normalize_affine_matrix(inverse_matrix_crop.squeeze(), crop_width, crop_height)

        # Create masks
        mask_original, mask_transformed = create_masks(offset1, offset2, h, w, (image_depth, image_height, image_width), inverse_matrix_norm, forward_matrix_norm, inverse_matrix_crop_norm, forward_matrix_crop_norm, contained)

        return (
            image_1_crop, transformed_image_2_crop, 
            label_1, label_2, 
            inverse_matrix_crop_norm, forward_matrix_crop_norm, 
            mask_original, mask_transformed,
        )

    def __len__(self):
        return len(self.file_paths)