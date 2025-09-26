import os
import random
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
import torchvision.transforms.functional as F

from datasets.base_dataset import BaseDataset
from datasets.utils import PairedRandomCrop, PairedCenterCrop
from utils.utils import (
    normalize_affine_matrix,
    create_masks,
    adjust_affine_transformation,
    apply_normalized_affine_to_polygon,
)
from utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register("ImageNetDataset")
class ImageNetDataset(BaseDataset):
    def __init__(self, root_path, img_size, train=True, cfg=None):
        super().__init__(cfg=cfg, train=train)

        # read file paths from txt
        with open(root_path, "r") as f:
            self.file_paths = [line.strip() for line in f]

        self.img_size = img_size
        self.labels = [0]  # placeholder

        # cropping strategy
        self.paired_crop = PairedRandomCrop(img_size, contained=True) if train else PairedCenterCrop(img_size)

    def __getitem__(self, index):
        file_path = self.file_paths[index]

        with Image.open(file_path).convert("L") as image_pil:
            # Apply base transforms (tensor + normalization)
            image = self.transforms(image_pil)
            image_depth, image_height, image_width = image.shape

            # --- Random affine ---
            angle = random.uniform(-45, 45)
            translate = (random.uniform(-40, 40), random.uniform(-40, 40))
            scale = random.uniform(0.8, 1.2)
            shear = random.uniform(-20, 20)

            transformed_image = F.affine(image, angle, translate, scale, [shear, shear])

            # Transformation matrix (inverse)
            inverse_matrix = F._get_inverse_affine_matrix(
                center=(image.shape[2] / 2, image.shape[1] / 2),
                angle=angle,
                translate=translate,
                scale=scale,
                shear=[shear, shear],
            )
            inverse_matrix = torch.tensor(inverse_matrix, dtype=torch.float32)
            label = torch.tensor(self.labels[0], dtype=torch.long)

            # Forward + inverse matrices
            inv_affine = torch.tensor(
                [
                    [inverse_matrix[0], inverse_matrix[1], inverse_matrix[2]],
                    [inverse_matrix[3], inverse_matrix[4], inverse_matrix[5]],
                    [0, 0, 1],
                ],
                dtype=torch.float32,
                device=inverse_matrix.device,
            )
            forward_affine = torch.inverse(inv_affine)
            forward_matrix = forward_affine[:2, :].reshape(-1)

            # Normalize affine matrices
            forward_matrix_norm = normalize_affine_matrix(
                forward_matrix.view(2, 3), image_width, image_height
            ).view_as(forward_matrix)
            inverse_matrix_norm = normalize_affine_matrix(
                inverse_matrix.view(2, 3), image_width, image_height
            ).view_as(inverse_matrix)

            # Crop images
            bbox_original = np.array(
                [[0, 0], [image.shape[2], 0], [image.shape[2], image.shape[1]], [0, image.shape[1]]]
            )
            bbox_transformed = apply_normalized_affine_to_polygon(
                bbox_original,
                forward_matrix_norm.numpy().reshape(2, 3),
                image.shape[1],
                image.shape[2],
            )
            image_crop, transformed_image_crop, offset1, offset2, h, w, contained = self.paired_crop(
                image, transformed_image, bbox=bbox_transformed
            )
            _, crop_height, crop_width = image_crop.shape

            # Adjust transformation for crop
            forward_matrix_crop = adjust_affine_transformation(forward_matrix, offset1, offset2)
            inverse_matrix_crop = adjust_affine_transformation(inverse_matrix, offset1, offset2)

            # Normalize crop matrices
            forward_matrix_crop_norm = normalize_affine_matrix(
                forward_matrix_crop.view(2, 3), crop_width, crop_height
            ).view_as(forward_matrix_crop)
            inverse_matrix_crop_norm = normalize_affine_matrix(
                inverse_matrix_crop.view(2, 3), crop_width, crop_height
            ).view_as(inverse_matrix_crop)

            # Create masks
            mask_original, mask_transformed = create_masks(
                offset1,
                offset2,
                h,
                w,
                (image_depth, image_height, image_width),
                inverse_matrix_norm,
                forward_matrix_norm,
                inverse_matrix_crop_norm,
                forward_matrix_crop_norm,
                contained,
            )

        return (
            image_crop,
            transformed_image_crop,
            label,
            label,
            inverse_matrix_crop_norm,
            forward_matrix_crop_norm,
            mask_original,
            mask_transformed,
        )

    def __len__(self):
        return len(self.file_paths)
