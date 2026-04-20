import os
import random
import itertools
import numpy as np

import torch

import rasterio

from datasets.base_dataset import BaseDataset
from utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register("DREAMDataset")
class DREAMDataset(BaseDataset):
    def __init__(self, root_path, img_size, train=True, cfg=None):
        super().__init__(cfg=cfg, train=train, img_size=img_size)

        self.img_size = img_size
        self.label_to_idx = {"optique": 0, "radar": 1, "S1": 2, "S2": 3, "SRTM": 4}
        self._sar_idx = self.label_to_idx["S1"]

        with open(root_path, "r") as f:
            zones = [line.strip() for line in f]

        self.file_paths = []
        self.labels = []

        if self.mode == "mono":
            for zone in zones:
                for mod, idx in self.label_to_idx.items():
                    folder = os.path.join(self.data_path, zone, mod)
                    files = [
                        os.path.join(folder, f)
                        for f in os.listdir(folder)
                        if os.path.isfile(os.path.join(folder, f))
                    ]
                    self.file_paths.extend(files)
                    self.labels.extend([idx] * len(files))

        elif self.mode == "multi":
            for zone in zones:
                for mod1, mod2 in itertools.permutations(self.label_to_idx.keys(), 2):
                    folder1 = os.path.join(self.data_path, zone, mod1)
                    folder2 = os.path.join(self.data_path, zone, mod2)
                    files1 = sorted(
                        os.path.join(folder1, f)
                        for f in os.listdir(folder1)
                        if os.path.isfile(os.path.join(folder1, f))
                    )
                    files2 = sorted(
                        os.path.join(folder2, f)
                        for f in os.listdir(folder2)
                        if os.path.isfile(os.path.join(folder2, f))
                    )
                    self.file_paths.extend(zip(files1, files2))
                    self.labels.extend(
                        list(zip([self.label_to_idx[mod1]] * len(files1),
                                 [self.label_to_idx[mod2]] * len(files2)))
                    )
        else:
            raise ValueError(f"Mode '{self.mode}' not implemented. Choose 'mono' or 'multi'.")

    def read_npy(self, file_path):
        arr = np.load(file_path, mmap_mode="r")
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        elif arr.ndim != 3:
            raise ValueError(f"Unexpected npy shape {arr.shape} for {file_path}")
        return np.ascontiguousarray(arr)

    def _normalize_image(self, image, label_idx):
        """Apply percentile normalization for SAR (S1) data, regular otherwise."""
        if label_idx == self._sar_idx:
            return self.transforms_percentile(image)
        return self.transforms(image)

    # Affine parameter ranges for MDAS
    _ANGLE_RANGE = (-35, 35)
    _TRANSLATE_RANGE = (-20, 20)
    _SCALE_RANGE = (0.8, 1.2)
    _SHEAR_RANGE = (-10, 10)

    def __getitem__(self, index):
        if self.mode == "mono":
            path_1 = path_2 = self.file_paths[index]
            label_1_idx = label_2_idx = self.labels[index]
        else:  # multi
            path_1, path_2 = self.file_paths[index]
            label_1_idx, label_2_idx = self.labels[index]

        image_1 = self.read_npy(path_1[:-3] + "npy")
        image_2 = self.read_npy(path_2[:-3] + "npy")

        image_1 = self._normalize_image(image_1, label_1_idx)
        image_2 = self._normalize_image(image_2, label_2_idx)
        image_shape = image_2.shape  # (C, H, W)

        label_1 = torch.tensor(label_1_idx, dtype=torch.long)
        label_2 = torch.tensor(label_2_idx, dtype=torch.long)

        if self.homography == "affine":
            angle = random.uniform(*self._ANGLE_RANGE)
            translate = (random.uniform(*self._TRANSLATE_RANGE), random.uniform(*self._TRANSLATE_RANGE))
            scale = random.uniform(*self._SCALE_RANGE)
            shear = random.uniform(*self._SHEAR_RANGE)
            transformed_image_2 = F.affine(image_2, angle, translate, scale, [shear, shear])
            forward_matrix, inverse_matrix = self._build_affine_matrices(image_2, angle, translate, scale, shear)

        elif self.homography == "projective":
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
        return len(self.file_paths)
