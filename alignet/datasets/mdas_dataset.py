from torch.utils.data import Dataset
from torchvision import transforms
import torch
import os
import time
import random
import numpy as np
from PIL import Image
from pathlib import Path
import torchvision.transforms.functional as F

import rasterio
from rasterio.warp import reproject
from rasterio.enums import Resampling

from shapely.geometry import Polygon, box
from shapely.prepared import prep

from datasets.base_dataset import BaseDataset
from datasets.utils import PairedRandomCrop, PairedCenterCrop, get_shared_cache
from utils.utils import normalize_affine_matrix, create_masks, adjust_affine_transformation, apply_normalized_affine_to_polygon
from utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register("MDASDataset")
class MDASDataset(BaseDataset):
    def __init__(self, root_path, img_size, train=True, target_resolution=5, cfg=None, preloaded_cache=None):
        """
        Args:
            root_path (str or Path): Path to the root directory containing subfolders for each label.
            transform (callable, optional): Optional transform to be applied on an image.
            target_resolution (float): Resolution in which the data will be treated in (in meters).
        """
        self.root = Path(root_path)
        self.target_resolution = target_resolution
        self.img_size = img_size
        self._cache = get_shared_cache()
        self.train = train

        if cfg is None or "mode" not in cfg:
            print("[WARNING] Mode not specified in cfg. Defaulting to 'mono'.", flush=True)
        self.mode = (cfg or {}).get("mode", "mono")
        self.homography = cfg.get("homography", "affine")
        self.radiometric_augmentation = (cfg or {}).get("rad_aug", False)

        self.paired_crop = PairedRandomCrop(self.img_size, contained=True)

        self.file_paths = []
        self.labels = []
        self.label_to_idx = {}  # Map label names to integers
        file_paths_per_label = {}
        # Collect all image paths and their corresponding labels
        for idx, filepath in enumerate(self.root.iterdir()):
            if filepath.is_dir():
                label = filepath.name
                self.label_to_idx[label] = idx
                file_paths_per_label[label] = []
                for img_path in filepath.glob("*.tif"):
                    file_paths_per_label[label].append(img_path)
                    self.file_paths.append(img_path)
                    self.labels.append(idx)
            else:
                label = filepath.name.split('.')[-2]
                self.label_to_idx[label] = idx
                self.file_paths.append(filepath)
                file_paths_per_label[label] = [filepath]
                self.labels.append(idx)

        if self.mode == 'multi':
            self.file_paths_pairs = []
            self.labels_pairs = []
            # Generate all possible permutations of labels in pairs (order matters)
            label_permutations = list(itertools.permutations(file_paths_per_label.keys(), 2))

            # Generate all possible path combinations for each ordered pair of labels
            for label1, label2 in label_permutations:
                for pair in zip(sorted(file_paths_per_label[label1], key=str), sorted(file_paths_per_label[label2], key=str)):
                    self.file_paths_pairs.extend([pair])
                self.labels_pairs.extend([[self.label_to_idx[label1], self.label_to_idx[label2]]] * len(file_paths_per_label[label1]))

    def resample_geotiff(self, file_path):
        file_path = str(file_path)
        
        # Open the original dataset
        with rasterio.open(file_path) as dataset:
            # Read only the first band
            if "Sentinel-2" in file_path:
                # print(f"[MDAS DATASET] Reading band 2 instead of band 1 because the file is Sentinel-2: {file_path}", flush=True)
                image = dataset.read(2)
            else:
                image = dataset.read(1)  # Shape: (H, W)

            # Get current resolution and transform
            transform = dataset.transform
            current_resolution = dataset.res[0]  # Assumes square pixels

            # Compute the scale factor and new shape
            scale_factor = current_resolution / self.target_resolution
            new_width = int(dataset.width * scale_factor)
            new_height = int(dataset.height * scale_factor)

            # Compute new transform
            new_transform = transform * transform.scale(
                dataset.width / new_width,
                dataset.height / new_height
            )

            # Prepare output array (1 band only)
            resampled_band = np.empty((new_height, new_width), dtype=dataset.dtypes[0])

            # Resample using rasterio.warp.reproject
            reproject(
                source=image,
                destination=resampled_band,
                src_transform=dataset.transform,
                dst_transform=new_transform,
                src_crs=dataset.crs,
                dst_crs=dataset.crs,
                resampling=Resampling.bilinear
            )

        # Expand dimensions to (H, W, 1) and cast to float32
        return resampled_band[:, :, np.newaxis].astype(np.float32)

    def __getitem__(self, index):
        idx = index % len(self.file_paths)

        # Load the GeoTIFF image
        if self.mode == 'multi':
            file_path_r, file_path_s = self.file_paths_pairs[idx]
            label_r, label_s = self.labels[idx]
            image_r = self._cache[str(file_path_r)]
            image_s = self._cache[str(file_path_s)]
        else:
            file_path = self.file_paths[idx]
            label = self.labels[idx]
            image = self._cache[str(file_path)]

        # Apply the transformations to the original image
        image_r = self.transforms(image if self.mode == 'mono' else image_r)
        image_s = self.transforms(image if self.mode == 'mono' else image_s)
        if self.radiometric_augmentation and self.train:
            image_r = self.data_augmentation(image_r)
            image_s = self.data_augmentation(image_s)
        image_depth, image_height, image_width = image.shape

        label = torch.tensor(label, dtype=torch.long)
        
        # Random horizontal flip
        if random.random() > 0.5:
            if self.mode == 'multi' and self.train:
                image_r = F.hflip(image_r)
                image_s = F.hflip(image_s)
            else:
                image = F.hflip(image)
        # Random vertical flip
        if random.random() > 0.5:
            if self.mode == 'multi' and self.train:
                image_r = F.vflip(image_r)
                image_s = F.vflip(image_s)
            else:
                image = F.vflip(image)

        if self.homography == 'affine':
            # Dynamically generate a random affine transformation
            angle = random.uniform(-35, 35)  # Random rotation
            translate = (random.uniform(-20, 20), random.uniform(-20, 20))  # Random translation
            scale = random.uniform(0.8, 1.2)  # Random scaling
            shear = random.uniform(-10, 10)  # Random shearing

            # Apply affine transformation
            if self.mode == 'multi' and self.train:
                transformed_image = F.affine(image_s, angle, translate, scale, [shear, shear])
                image = image_r
            else:
                transformed_image = F.affine(image, angle, translate, scale, [shear, shear])

            # Transformation matrix
            inverse_matrix = F._get_inverse_affine_matrix(
                center=(image.shape[2] / 2, image.shape[1] / 2),
                angle=angle,
                translate=translate,
                scale=scale,
                shear=[shear, shear]
            )
            inverse_matrix = torch.tensor(inverse_matrix, dtype=torch.float32)

            # Inverse transformation matrix
            inv_affine = np.array([
                [inverse_matrix[0], inverse_matrix[1], inverse_matrix[2]],
                [inverse_matrix[3], inverse_matrix[4], inverse_matrix[5]],
                [0, 0, 1]
            ])
            forward_affine = np.linalg.inv(inv_affine)
            forward_matrix = torch.tensor(forward_affine[:2, :].flatten(), dtype=torch.float32)

        elif self.homography == 'projective':
            transformed_image, forward_affine = apply_random_homography(image_s if self.mode == 'multi' else image, max_displacement=80)
            while torch.isnan(forward_affine).any():
                transformed_image, forward_affine = apply_random_homography(image_s if self.mode == 'multi' else image, max_displacement=80)
            if self.mode == 'multi':
                image = image_r
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
        bbox_original = np.array([[0, 0], [image.shape[2], 0], [image.shape[2], image.shape[1]], [0, image.shape[1]]])
        bbox_transformed = apply_normalized_affine_to_polygon(bbox_original, forward_matrix_norm.numpy(), image.shape[1], image.shape[2])
        image_crop, transformed_image_crop, offset1, offset2, h, w, contained = self.paired_crop(image, transformed_image, bbox=bbox_transformed)
        _, crop_height, crop_width = image_crop.shape

        # Adjust transformation to crop size
        forward_matrix_crop = adjust_affine_transformation(forward_matrix, offset1, offset2)
        inverse_matrix_crop = adjust_affine_transformation(inverse_matrix, offset1, offset2)

        # Normalize Affine Transformation Matrix
        forward_matrix_crop_norm = normalize_affine_matrix(forward_matrix_crop.squeeze(), crop_width, crop_height)
        inverse_matrix_crop_norm = normalize_affine_matrix(inverse_matrix_crop.squeeze(), crop_width, crop_height)

        # Create masks
        mask_original, mask_transformed = create_masks(int(offset1), int(offset2), int(h), int(w), (int(image_depth), int(image_height), int(image_width)), inverse_matrix_norm, forward_matrix_norm, inverse_matrix_crop_norm, forward_matrix_crop_norm, contained)

        return image_crop, transformed_image_crop, label, label, inverse_matrix_crop_norm, forward_matrix_crop_norm, mask_original, mask_transformed


    def __len__(self):
        if self.train:
            return len(self.file_paths) * 6 * 3 * 1000
        else:
            return len(self.file_paths) * 6 * 3 * 10