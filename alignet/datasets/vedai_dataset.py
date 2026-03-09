import random
import numpy as np
from PIL import Image

import torch
import torchvision.transforms.functional as F

from datasets.base_dataset import BaseDataset
from datasets.utils import PairedRandomCrop, PairedCenterCrop
from utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register("VEDAIDataset")
class VEDAIDataset(BaseDataset):
    def __init__(self, root_path, img_size, train=True, cfg=None):
        super().__init__(cfg=cfg, train=train, img_size=img_size)

        if cfg is None or "mode" not in cfg:
            print("[WARNING] Mode not specified in cfg. Defaulting to 'mono'.", flush=True)

        self.label_to_idx = {"ir": 0, "co": 1}

        with open(root_path, "r") as f:
            self.file_paths = [line.strip() for line in f]

        self.img_size = img_size

    # Affine parameter ranges for VEDAI (same wide range as ImageNet)
    _ANGLE_RANGE = (-45, 45)
    _TRANSLATE_RANGE = (-40, 40)
    _SCALE_RANGE = (0.8, 1.2)
    _SHEAR_RANGE = (-20, 20)

    def __getitem__(self, index):
        idx = index % len(self.file_paths)
        sample = self.file_paths[idx]

        if self.mode == "mono":
            modality = random.choice(["ir", "co"])
            modality1 = modality2 = modality
        else:
            modality1, modality2 = "ir", "co"

        image_1 = Image.open(f"data/VEDAI/{modality1}/{sample}_{modality1}.png").convert("L")
        image_2 = Image.open(f"data/VEDAI/{modality2}/{sample}_{modality2}.png").convert("L")

        image_1 = self.transforms(image_1)
        image_2 = self.transforms(image_2)
        image_shape = image_2.shape  # (C, H, W)

        label_1 = torch.tensor(self.label_to_idx[modality1], dtype=torch.long)
        label_2 = torch.tensor(self.label_to_idx[modality2], dtype=torch.long)

        if self.homography == "affine":
            angle = random.uniform(*self._ANGLE_RANGE)
            translate = (random.uniform(*self._TRANSLATE_RANGE), random.uniform(*self._TRANSLATE_RANGE))
            scale = random.uniform(*self._SCALE_RANGE)
            shear = random.uniform(*self._SHEAR_RANGE)
            transformed_image_2 = F.affine(image_2, angle, translate, scale, [shear, shear])
            forward_matrix, inverse_matrix = self._build_affine_matrices(image_2, angle, translate, scale, shear)

        elif self.homography == "projective":
            transformed_image_2, forward_matrix, inverse_matrix = self._build_random_projective_matrices(image_2)

        image_1_crop, transformed_image_2_crop, inv_norm, fwd_norm, mask1, mask2 = self._crop_and_finalize(
            image_1, transformed_image_2, forward_matrix, inverse_matrix, image_shape
        )

        return image_1_crop, transformed_image_2_crop, label_1, label_2, inv_norm, fwd_norm, mask1, mask2

    def __len__(self):
        return len(self.file_paths) * 5 * (100 if self.train else 10)
