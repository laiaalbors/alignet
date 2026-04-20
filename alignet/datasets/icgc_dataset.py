import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torchvision import transforms

import rasterio
from rasterio.warp import reproject
from rasterio.enums import Resampling

from datasets.base_dataset import BaseDataset
from datasets.utils import get_shared_cache
from utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register("ICGCDataset")
class ICGCDataset(BaseDataset):
    def __init__(self, root_path, img_size, train=True, cfg=None):
        super().__init__(cfg=cfg, train=train, img_size=img_size)

        self.target_resolution = (cfg or {}).get("target_resolution", 1)
        self.img_size = img_size
        self._cache = get_shared_cache()
        
        self.label_to_idx = {'hr': 0, 'lr': 1}

        # self.file_paths = pd.read_csv(root_path)
        self.file_paths_pairs = []
        df = pd.read_csv(root_path)
        for _, row in df.iterrows():
            area = row['area']
            date = row['date']
            self.file_paths_pairs.append((
                os.path.join(self.data_path, f"hr/area{area}_orto.tif"),
                os.path.join(self.data_path, f"lr/area{area}_s2_{date}.tif"),
            ))
        print(f"self.file_paths_pairs: {len(self.file_paths_pairs)}", flush=True)

        self.file_paths = list(set(
            path for pair in self.file_paths_pairs for path in pair
        ))
        print(f"self.file_paths: {len(self.file_paths)}", flush=True)

    
    def resample_geotiff(self, file_path, min_res=10.0, max_res=5.0):
        file_path = str(file_path)

        with rasterio.open(file_path) as dataset:
            image = dataset.read(1)  # Shape: (H, W)
            transform = dataset.transform
            current_resolution = dataset.res[0]  # Assumes square pixels

            # If low-res path, first downsample to 30m, then to target_resolution
            if "/lr/" in file_path and current_resolution < min_res:
                print(f"Downsampling lr data to {min_res}m...", flush=True)
                # Step 1: downsample to 30m
                scale_factor_30m = current_resolution / min_res
                w_30m = int(dataset.width * scale_factor_30m)
                h_30m = int(dataset.height * scale_factor_30m)

                transform_30m = transform * transform.scale(
                    dataset.width / w_30m,
                    dataset.height / h_30m
                )

                image_30m = np.empty((h_30m, w_30m), dtype=dataset.dtypes[0])
                reproject(
                    source=image,
                    destination=image_30m,
                    src_transform=transform,
                    dst_transform=transform_30m,
                    src_crs=dataset.crs,
                    dst_crs=dataset.crs,
                    resampling=Resampling.bilinear
                )

                # Step 2: from 30m to target_resolution
                scale_factor = min_res / max_res
                new_width = int(w_30m * scale_factor)
                new_height = int(h_30m * scale_factor)

                new_transform = transform_30m * transform_30m.scale(
                    w_30m / new_width,
                    h_30m / new_height
                )

                resampled_band = np.empty((new_height, new_width), dtype=dataset.dtypes[0])
                reproject(
                    source=image_30m,
                    destination=resampled_band,
                    src_transform=transform_30m,
                    dst_transform=new_transform,
                    src_crs=dataset.crs,
                    dst_crs=dataset.crs,
                    resampling=Resampling.bilinear
                )
            else:
                # Direct resample to target_resolution
                scale_factor = current_resolution / max_res
                new_width = int(dataset.width * scale_factor)
                new_height = int(dataset.height * scale_factor)

                new_transform = transform * transform.scale(
                    dataset.width / new_width,
                    dataset.height / new_height
                )

                resampled_band = np.empty((new_height, new_width), dtype=dataset.dtypes[0])
                reproject(
                    source=image,
                    destination=resampled_band,
                    src_transform=transform,
                    dst_transform=new_transform,
                    src_crs=dataset.crs,
                    dst_crs=dataset.crs,
                    resampling=Resampling.bilinear
                )

        return resampled_band[:, :, np.newaxis].astype(np.float32)
        

    def resample_geotiff_old(self, file_path):
        file_path = str(file_path)
        
        # Open the original dataset
        with rasterio.open(file_path) as dataset:
            # Read only the first band
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


    # Affine parameter ranges for ICGC
    _ANGLE_RANGE = (-45, 45)
    _TRANSLATE_RANGE = (-40, 40)
    _SCALE_RANGE = (0.8, 1.2)
    _SHEAR_RANGE = (-20, 20)

    def __getitem__(self, index):
        # idx = index % len(self.file_paths)
        # row = self.file_paths.iloc[idx]
        # area = row['area']
        # date = row['date']

        # templates = {
        #     "hr": lambda area, date: os.path.join(self.data_path, f"hr/area{area}_orto.tif"),
        #     "lr": lambda area, date: os.path.join(self.data_path, f"lr/area{area}_s2_{date}.tif"),
        # }

        # if self.mode == "mono":
        #     modality1 = modality2 = random.choice(["hr", "lr"])
        # else:
        #     modality1, modality2 = "hr", "lr"

        # path_1 = templates[modality1](area, date)
        # path_2 = templates[modality2](area, date)

        path_hr, path_lr = self.file_paths_pairs[index % len(self.file_paths_pairs)]
        p1, p2 = random.choice([(5,10), (5,20), (3,30), (5,50)])
        path_hr = path_hr+f"_{p1}"+f"_{p2}"
        path_lr = path_lr+f"_{p1}"+f"_{p2}"

        if self.mode == "mono":
            choice = random.choice(["hr", "lr"])
            path_1 = path_hr if choice == "hr" else path_lr
            path_2 = path_1
            modality1 = modality2 = choice
        else:
            path_1, path_2 = path_hr, path_lr
            modality1, modality2 = "hr", "lr"

        try:
            image_1 = self._cache[path_1]
            image_2 = self._cache[path_2]
        except KeyError:
            print(f"Reading images {path_1} and {path_2}, and saving them in cache...", flush=True)
            image_1 = self.resample_geotiff(path_1)
            image_2 = self.resample_geotiff(path_2)
            # Apply the transformations to the original image
            image_1 = self.transforms_percentile(image_1)
            image_2 = self.transforms_percentile(image_2)
            self._cache[path_1] = image_1 #.numpy().float()
            self._cache[path_2] = image_2 #.numpy().float()
                
        image_shape = image_1.shape  # (C, H, W)

        label_1 = torch.tensor(self.label_to_idx[modality1], dtype=torch.long)
        label_2 = torch.tensor(self.label_to_idx[modality2], dtype=torch.long)

        if self.homography == 'affine':
            angle = random.uniform(*self._ANGLE_RANGE)
            translate = (random.uniform(*self._TRANSLATE_RANGE), random.uniform(*self._TRANSLATE_RANGE))
            scale = random.uniform(*self._SCALE_RANGE)
            shear = random.uniform(*self._SHEAR_RANGE)
            transformed_image_2 = F.affine(image_2, angle, translate, scale, [shear, shear])
            forward_matrix, inverse_matrix = self._build_affine_matrices(image_2, angle, translate, scale, shear)

        elif self.homography == 'projective':
            transformed_image_2, forward_matrix, inverse_matrix = self._build_random_projective_matrices(image_2)

        elif self.homography == "affine_projective":
            angle = random.uniform(*self._ANGLE_RANGE)
            translate = (random.uniform(*self._TRANSLATE_RANGE), random.uniform(*self._TRANSLATE_RANGE))
            scale = random.uniform(*self._SCALE_RANGE)
            shear = random.uniform(*self._SHEAR_RANGE)
            transformed_image_2, forward_matrix, inverse_matrix = self._build_affine_projective_matrices(image_2, angle, translate, scale, shear)

        image_1_crop, transformed_image_2_crop, inv_norm, fwd_norm, mask1, mask2 = self._crop_and_finalize(
            image_1, transformed_image_2, forward_matrix, inverse_matrix, image_shape
        )

        return image_1_crop, transformed_image_2_crop, label_1, label_2, inv_norm, fwd_norm, mask1, mask2


    def __len__(self):
        return len(self.file_paths_pairs) * 130