import os
import math
import random
import numpy as np
from PIL import Image

import torch
import torchvision.transforms.functional as F

from datasets.base_dataset import BaseDataset
from datasets.utils import PairedRandomCrop, PairedCenterCrop
from utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register("ImageNetDataset")
class ImageNetDataset(BaseDataset):
    def __init__(self, root_path, img_size, train=True, cfg=None):
        super().__init__(cfg=cfg, train=train, img_size=img_size)

        with open(root_path, "r") as f:
            self.file_paths = [os.path.join(self.data_path, line.strip()) for line in f]

        self.img_size = img_size
        self.labels = [0]  # placeholder

        if self.mode == "multi":
            print("[Warning] ImageNet cannot be used with multimodal training. Switching to mono.", flush=True)
            self.mode = "mono"

        self.use_subset = cfg.get("use_subset", False)

        # img_size is passed to super() so self.paired_crop is already set, but
        # ImageNet always uses "contained=True" for random crop, which matches the base default.

    # Affine parameter ranges for ImageNet (wider than RS datasets)
    _ANGLE_RANGE = (-45, 45)
    _TRANSLATE_RANGE = (-40, 40)
    _SCALE_RANGE = (0.8, 1.2)
    _SHEAR_RANGE = (-20, 20)

    def __getitem__(self, index):
        with Image.open(self.file_paths[index]) as image_pil:
            image_pil = image_pil.convert("RGB" if self.train else "L")
            image = self.transforms(image_pil)

            # Ensure image is at least img_size in both dimensions
            if image.shape[1] < self.img_size or image.shape[2] < self.img_size:
                scale_factor = max(self.img_size / image.shape[1], self.img_size / image.shape[2])
                new_h = math.ceil(image.shape[1] * scale_factor)
                new_w = math.ceil(image.shape[2] * scale_factor)
                image = F.resize(image, [new_h, new_w])

        image_depth, image_height, image_width = image.shape
        label = torch.tensor(self.labels[0], dtype=torch.long)

        if self.homography == "affine":
            angle = random.uniform(*self._ANGLE_RANGE)
            translate = (random.uniform(*self._TRANSLATE_RANGE), random.uniform(*self._TRANSLATE_RANGE))
            scale = random.uniform(*self._SCALE_RANGE)
            shear = random.uniform(*self._SHEAR_RANGE)
            image2 = F.affine(image, angle, translate, scale, [shear, shear])
            forward_matrix, inverse_matrix = self._build_affine_matrices(image, angle, translate, scale, shear)

        elif self.homography == "projective":
            image2, forward_matrix, inverse_matrix = self._build_random_projective_matrices(image)

        elif self.homography == "affine_projective":
            angle = random.uniform(*self._ANGLE_RANGE)
            translate = (random.uniform(*self._TRANSLATE_RANGE), random.uniform(*self._TRANSLATE_RANGE))
            scale = random.uniform(*self._SCALE_RANGE)
            shear = random.uniform(*self._SHEAR_RANGE)
            image2, forward_matrix, inverse_matrix = self._build_affine_projective_matrices(image, angle, translate, scale, shear)

        # Augmentation applied independently to each image (simulates different sensors)
        image1 = self._apply_augmentation(image)
        image2 = self._apply_augmentation(image2)

        image1_crop, image2_crop, inv_norm, fwd_norm, mask1, mask2 = self._crop_and_finalize(
            image1, image2, forward_matrix, inverse_matrix, (image_depth, image_height, image_width)
        )

        return image1_crop, image2_crop, label, label, inv_norm, fwd_norm, mask1, mask2

    def __len__(self):
        if self.use_subset:
            return 15000    # Just use a subset when fine-tuning the model together with real RS datasets so that it does not dominate
        return len(self.file_paths)
