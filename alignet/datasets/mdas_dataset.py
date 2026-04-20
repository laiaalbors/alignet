import random
import itertools
import numpy as np
from pathlib import Path

import torch
import torchvision.transforms.functional as F

import rasterio
from rasterio.warp import reproject
from rasterio.enums import Resampling

from datasets.base_dataset import BaseDataset
from datasets.utils import PairedRandomCrop, get_shared_cache
from utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register("MDASDataset")
class MDASDataset(BaseDataset):
    def __init__(self, root_path, img_size, train=True, target_resolution=5, cfg=None, preloaded_cache=None):
        super().__init__(cfg=cfg, train=train)

        self.root = Path(root_path)
        self.target_resolution = target_resolution
        self.img_size = img_size
        self._cache = get_shared_cache()

        # MDAS always uses random crop (no center crop for validation — images are large enough)
        self.paired_crop = PairedRandomCrop(self.img_size, contained=True)

        # Collect file paths grouped by label (modality subfolder name)
        self.file_paths = []
        self.labels = []
        self.label_to_idx = {}
        file_paths_per_label = {}

        for idx, filepath in enumerate(sorted(self.root.iterdir())):
            if filepath.is_dir():
                label = filepath.name
                self.label_to_idx[label] = idx
                paths = sorted(filepath.glob("*.tif"))
                file_paths_per_label[label] = paths
                self.file_paths.extend(paths)
                self.labels.extend([idx] * len(paths))
            else:
                label = filepath.stem
                self.label_to_idx[label] = idx
                file_paths_per_label[label] = [filepath]
                self.file_paths.append(filepath)
                self.labels.append(idx)

        # Build paired lists for multi mode
        self.file_paths_pairs = []
        self.labels_pairs = []
        if self.mode == "multi":
            for label1, label2 in itertools.permutations(file_paths_per_label.keys(), 2):
                paths1 = sorted(file_paths_per_label[label1], key=str)
                paths2 = sorted(file_paths_per_label[label2], key=str)
                pairs = list(zip(paths1, paths2))
                self.file_paths_pairs.extend(pairs)
                self.labels_pairs.extend(
                    [[self.label_to_idx[label1], self.label_to_idx[label2]]] * len(pairs)
                )

    def resample_geotiff(self, file_path, min_res=None, max_res=None):
        file_path = str(file_path)
        with rasterio.open(file_path) as dataset:
            band_idx = 2 if "Sentinel-2" in file_path else 1
            image = dataset.read(band_idx)
            scale_factor = dataset.res[0] / self.target_resolution
            new_width = int(dataset.width * scale_factor)
            new_height = int(dataset.height * scale_factor)
            new_transform = dataset.transform * dataset.transform.scale(
                dataset.width / new_width, dataset.height / new_height
            )
            resampled = np.empty((new_height, new_width), dtype=dataset.dtypes[0])
            reproject(
                source=image,
                destination=resampled,
                src_transform=dataset.transform,
                dst_transform=new_transform,
                src_crs=dataset.crs,
                dst_crs=dataset.crs,
                resampling=Resampling.bilinear,
            )
        return resampled[:, :, np.newaxis].astype(np.float32)

    def _load_image_pair(self, index):
        """Load image pair and labels based on the current mode.

        Returns:
            image_r, image_s: Tensors (C, H, W) — reference and source images.
            label_r, label_s: integer class indices.
        """
        if self.mode == "multi":
            idx = index % len(self.file_paths_pairs)
            file_path_r, file_path_s = self.file_paths_pairs[idx]
            label_r_idx, label_s_idx = self.labels_pairs[idx]
            image_r = self._cache[str(file_path_r)]
            image_s = self._cache[str(file_path_s)]
            if not isinstance(image_r, torch.Tensor):
                image_r = self.transforms_percentile(image_r)
            if not isinstance(image_s, torch.Tensor):
                image_s = self.transforms_percentile(image_s)
        else:  # mono
            idx = index % len(self.file_paths)
            file_path = self.file_paths[idx]
            image = self._cache[str(file_path)]
            if not isinstance(image, torch.Tensor):
                image = self.transforms_percentile(image)
            image_r = image_s = image
            label_r_idx = label_s_idx = self.labels[idx]

        return image_r, image_s, label_r_idx, label_s_idx

    # Affine parameter ranges for MDAS
    _ANGLE_RANGE = (-35, 35)
    _TRANSLATE_RANGE = (-20, 20)
    _SCALE_RANGE = (0.8, 1.2)
    _SHEAR_RANGE = (-10, 10)

    def __getitem__(self, index):
        image_r, image_s, label_r_idx, label_s_idx = self._load_image_pair(index)

        # Random flips applied identically to both images
        if self.train and random.random() > 0.5:
            image_r = F.hflip(image_r)
            image_s = F.hflip(image_s)
        if self.train and random.random() > 0.5:
            image_r = F.vflip(image_r)
            image_s = F.vflip(image_s)

        image_shape = image_r.shape  # (C, H, W)

        if self.homography == "affine":
            angle = random.uniform(*self._ANGLE_RANGE)
            translate = (random.uniform(*self._TRANSLATE_RANGE), random.uniform(*self._TRANSLATE_RANGE))
            scale = random.uniform(*self._SCALE_RANGE)
            shear = random.uniform(*self._SHEAR_RANGE)
            transformed_image_s = F.affine(image_s, angle, translate, scale, [shear, shear])
            forward_matrix, inverse_matrix = self._build_affine_matrices(image_s, angle, translate, scale, shear)

        elif self.homography == "projective":
            transformed_image_s, forward_matrix, inverse_matrix = self._build_random_projective_matrices(image_s)

        elif self.homography == "affine_projective":
            angle = random.uniform(*self._ANGLE_RANGE)
            translate = (random.uniform(*self._TRANSLATE_RANGE), random.uniform(*self._TRANSLATE_RANGE))
            scale = random.uniform(*self._SCALE_RANGE)
            shear = random.uniform(*self._SHEAR_RANGE)
            transformed_image_s, forward_matrix, inverse_matrix = self._build_affine_projective_matrices(image_s, angle, translate, scale, shear)

        label_r = torch.tensor(label_r_idx, dtype=torch.long)
        label_s = torch.tensor(label_s_idx, dtype=torch.long)

        image_r_crop, transformed_image_s_crop, inv_norm, fwd_norm, mask1, mask2 = self._crop_and_finalize(
            image_r, transformed_image_s, forward_matrix, inverse_matrix, image_shape
        )

        return image_r_crop, transformed_image_s_crop, label_r, label_s, inv_norm, fwd_norm, mask1, mask2

    def __len__(self):
        if self.train:
            return len(self.file_paths) * 6 * 3 * 1000
        else:
            return len(self.file_paths) * 6 * 3 * 10
