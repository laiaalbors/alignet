import os
import time
import random
import numpy as np
from PIL import Image

import torch
from torchvision import io
from torchvision import transforms
import torchvision.transforms.functional as F

from datasets.base_dataset import BaseDataset
from datasets.utils import PairedRandomCrop, PairedCenterCrop
from utils.utils import normalize_affine_matrix, create_masks, adjust_affine_transformation, apply_normalized_affine_to_polygon
from utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register("VEDAIDataset")
class VEDAIDataset(BaseDataset):
    def __init__(self, root_path, img_size, train=True, cfg=None):
        super().__init__(cfg=cfg, train=train)

        if cfg is None or "mode" not in cfg:
            print("[WARNING] Mode not specified in cfg. Defaulting to 'mono'.", flush=True)
        self.mode = (cfg or {}).get("mode", "mono")
        self.homography = cfg.get("homography", "affine")
        self.radiometric_augmentation = (cfg or {}).get("rad_aug", False)
        self.train = train
        
        self.label_to_idx = {'ir': 0, 'co': 1}

        with open(root_path, "r") as f:
            self.file_paths = [line.strip() for line in f]
        
        self.img_size = img_size
        self.labels = [0, 1]

        self.paired_crop = PairedRandomCrop(img_size, contained=True) if train else PairedCenterCrop(img_size)


    def __getitem__(self, index):
        idx = index % len(self.file_paths)
        sample = self.file_paths[idx]

        if self.mode == "mono":
            modality1 = modality2 = random.choice(['ir', 'co'])
        else:
            modality1, modality2 = 'ir', 'co'

        path_1 = f"data/VEDAI/{modality1}/{sample}_{modality1}.png"
        path_2 = f"data/VEDAI/{modality2}/{sample}_{modality2}.png"

        image_1 = Image.open(path_1).convert('L')
        image_2 = Image.open(path_2).convert('L')
                
        # Apply the transformations to the original image
        image_1 = self.transforms(image_1)
        image_2 = self.transforms(image_2)
        if self.radiometric_augmentation and self.train:
            image_1 = self.data_augmentation(image_1)
            image_2 = self.data_augmentation(image_2)
        image_depth, image_height, image_width = image_2.shape

        label_1 = torch.tensor(self.labels[0], dtype=torch.long)
        label_2 = torch.tensor(self.labels[1], dtype=torch.long)

        if self.homography == 'affine':
            # Dynamically generate a random affine transformation
            angle = random.uniform(-45, 45)  # Random rotation
            translate = (random.uniform(-40, 40), random.uniform(-40, 40))  # Random translation
            scale = random.uniform(0.8, 1.2)  # Random scaling
            shear = random.uniform(-20, 20)  # Random shearing

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
            transformed_image_2, forward_affine = apply_random_homography(image_2, max_displacement=80)
            while torch.isnan(forward_affine).any():
                transformed_image_2, forward_affine = apply_random_homography(image_2, max_displacement=80)
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
            mask_original, mask_transformed
        )


    def __len__(self):
        return len(self.file_paths) * 5 * (100 if self.train else 10)