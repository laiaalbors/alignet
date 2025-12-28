from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import save_image
import torch
import os
import random
import itertools
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from collections import defaultdict
import torchvision.transforms.functional as F

import rasterio
from rasterio.warp import reproject
from rasterio.enums import Resampling

from shapely.geometry import Polygon, box
from shapely.prepared import prep

from datasets.base_dataset import BaseDataset
from datasets.utils import PairedRandomCrop, PairedCenterCrop, get_shared_cache
from utils.utils import normalize_affine_matrix, create_masks, adjust_affine_transformation, apply_normalized_affine_to_polygon, apply_random_homography
from utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register("MDASCSVDataset")
class MDASCSVDataset(BaseDataset):
    def __init__(self, root_path, img_size, train=True, target_resolution=5, cfg=None, preloaded_cache=None):
        """
        Args:
            root (str or Path): Path to the root directory containing subfolders for each label.
            transform (callable, optional): Optional transform to be applied on an image.
            target_resolution (float): Resolution in which the data will be treated in (in meters).
        """
        self.root = Path(root_path)
        self.root_path = root_path
        self.target_resolution = target_resolution
        self.img_size = img_size
        self.train = train
        self.homography = cfg.get("homography", "affine")

        if cfg is None or "mode" not in cfg:
            print("[WARNING] Mode not specified in cfg. Defaulting to 'mono'.", flush=True)
        self.mode = (cfg or {}).get("mode", "mono")
        self.radiometric_augmentation = (cfg or {}).get("rad_aug", False)


        self.file_paths = []
        self.labels = []
        self.label_to_idx = {}  # Map label names to integers
        self._cache = get_shared_cache()

        self.df = pd.read_csv(root_path)

        self.label_to_idx = {}
        file_paths_per_label = {}
        for idx, label in enumerate(self.df['label'].unique()):
            self.label_to_idx[label] = idx
            filepath = self.df[self.df['label'] == label]['image_path'].unique()[0]
            file_paths_per_label[label] = [filepath]

        if 'label2' not in self.df.columns: 
            # Generate all possible permutations of labels in pairs (order matters)
            label_permutations = list(itertools.permutations(file_paths_per_label.keys(), 2))

            # Generate all possible path combinations for each ordered pair of labels
            for label1, label2 in label_permutations:
                for pair in zip(sorted(file_paths_per_label[label1], key=str), sorted(file_paths_per_label[label2], key=str)):
                    self.file_paths.extend([pair])
                self.labels.extend([[self.label_to_idx[label1], self.label_to_idx[label2]]] * len(file_paths_per_label[label1]))

            idx_to_label = {v: k for k, v in self.label_to_idx.items()}

            mapping = defaultdict(list)
            for (fp1, fp2), (idx1, idx2) in zip(self.file_paths, self.labels):
                mapping[fp1].append((fp2, idx_to_label[idx2]))

            expanded_rows = []
            for _, row in self.df.iterrows():
                img1 = row["image_path"]
                for img2, label2 in mapping[img1]:
                    new_row = row.copy()
                    new_row["image_path2"] = img2
                    new_row["label2"] = label2
                    expanded_rows.append(new_row)

            self.df = pd.DataFrame(expanded_rows)

        else:
            for idx, label in enumerate(self.df['label2'].unique()):
                if label not in self.label_to_idx.keys():
                    self.label_to_idx[label] = idx
                    filepath = self.df[self.df['label2'] == label]['image_path2'].unique()[0]
                    file_paths_per_label[label] = [filepath]

        self.file_paths = list(set(self.df['image_path'].unique()) | set(self.df['image_path2'].unique())) #self.df['image_path'].unique()

        print(f"Unique image_path in df: {len(self.df['image_path'].unique())}", flush=True)
        print(f"Unique image_path2 in df: {len(self.df['image_path2'].unique())}", flush=True)
    
    def resample_geotiff(self, file_path):
        file_path = str(file_path)

        # Open the original dataset
        with rasterio.open(file_path) as dataset:
            # Read only the first band
            if "Sentinel-2" in file_path:
                # print(f"[MDASTESTCSVDATASET] Reading band 2 instead of band 1 because the file is Sentinel-2: {file_path}", flush=True)
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
        file_path_r = self.df.iloc[index]['image_path']
        label_r = self.label_to_idx[self.df.iloc[index]['label']]
        
        offset1 = self.df.iloc[index]['offset1']
        offset2 = self.df.iloc[index]['offset2']
        h = self.df.iloc[index]['h']
        w = self.df.iloc[index]['w']

        # Load the GeoTIFF image
        image_r = self._cache[str(file_path_r)]

        if self.mode == 'multi':
            file_path_s = self.df.iloc[index]['image_path2']
            label_s = self.label_to_idx[self.df.iloc[index]['label2']]
            image_s = self._cache[str(file_path_s)]

        image_r = self.transforms(image_r)
        if self.mode == 'multi':
            image_s = self.transforms(image_s)
        if self.radiometric_augmentation and self.train:
            image_r = self.data_augmentation(image_r)
            if self.mode == 'multi':
                image_s = self.data_augmentation(image_s)
        image_depth, image_height, image_width = image_r.shape

        # Random horizontal flip
        if self.train and random.random() > 0.5:
            image_r = F.hflip(image_r)
            if self.mode == 'multi':
                image_s = F.hflip(image_s)
        # Random vertical flip
        if self.train and random.random() > 0.5:
            image_r = F.vflip(image_r)
            if self.mode == 'multi':
                image_s = F.vflip(image_s)

        if self.homography == 'affine':
            angle = self.df.iloc[index]['angle']
            translate = (self.df.iloc[index]['translate_x'], self.df.iloc[index]['translate_y'])
            scale = self.df.iloc[index]['scale']
            shear = self.df.iloc[index]['shear']

            # Apply affine transformation
            transformed_image_s = F.affine(image_s if self.mode == 'multi' else image_r, angle, translate, scale, [shear, shear])

            # Transformation matrix
            inverse_matrix = F._get_inverse_affine_matrix(
                center=(image_r.shape[2] / 2, image_r.shape[1] / 2),
                angle=angle,
                translate=translate,
                scale=scale,
                shear=[shear, shear]
            )
            inverse_matrix = torch.tensor(inverse_matrix, dtype=torch.float32)
            label_r = torch.tensor(label_r, dtype=torch.long)
            if self.mode == 'multi':
                label_s = torch.tensor(label_s, dtype=torch.long)
            else:
                label_s = label_r

            # Inverse transformation matrix
            inv_affine = np.array([
                [inverse_matrix[0], inverse_matrix[1], inverse_matrix[2]],
                [inverse_matrix[3], inverse_matrix[4], inverse_matrix[5]],
                [0, 0, 1]
            ])
            forward_affine = np.linalg.inv(inv_affine)
            forward_matrix = torch.tensor(forward_affine[:2, :].flatten(), dtype=torch.float32)

        elif self.homography == 'projective':
            cols = ["h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21"]
            row_values = self.df.loc[index, cols].tolist()
            row_values.append(1.0)
            tensor_vals = torch.tensor(row_values, dtype=torch.float32)
            H = tensor_vals.reshape(3, 3)
            transformed_image_s, forward_affine = apply_random_homography(image_s if self.mode == 'multi' else image_r, H)
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
        image_r_crop = F.crop(image_r, offset1, offset2, h, w)
        transformed_image_s_crop = F.crop(transformed_image_s, offset1, offset2, h, w)
        _, crop_height, crop_width = image_r_crop.shape

        # Adjust transformation to crop size
        forward_matrix_crop = adjust_affine_transformation(forward_matrix, offset1, offset2)
        inverse_matrix_crop = adjust_affine_transformation(inverse_matrix, offset1, offset2)

        # Normalize Affine Transformation Matrix
        forward_matrix_crop_norm = normalize_affine_matrix(forward_matrix_crop.squeeze(), crop_width, crop_height)
        inverse_matrix_crop_norm = normalize_affine_matrix(inverse_matrix_crop.squeeze(), crop_width, crop_height)

        # Create masks
        mask_original, mask_transformed = create_masks(offset1, offset2, h, w, (image_depth, image_height, image_width), inverse_matrix_norm, forward_matrix_norm, inverse_matrix_crop_norm, forward_matrix_crop_norm, False)
        
        return image_r_crop, transformed_image_s_crop, label_r, label_s, inverse_matrix_crop_norm, forward_matrix_crop_norm, mask_original, mask_transformed


    def __len__(self):
        return len(self.df)