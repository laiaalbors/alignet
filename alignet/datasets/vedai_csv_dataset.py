import os
import time
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torchvision import io
from torchvision import transforms
import torchvision.transforms.functional as F

from datasets.base_dataset import BaseDataset
from datasets.utils import PairedRandomCrop, PairedCenterCrop
from utils.utils import normalize_affine_matrix, create_masks, adjust_affine_transformation, apply_normalized_affine_to_polygon
from utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register("VEDAICSVDataset")
class VEDAICSVDataset(BaseDataset):
    def __init__(self, root_path, img_size, train=True, cfg=None):
        super().__init__(cfg=cfg, train=train)

        if cfg is None or "mode" not in cfg:
            print("[WARNING] Mode not specified in cfg. Defaulting to 'mono'.", flush=True)
        self.mode = (cfg or {}).get("mode", "mono")

        self.label_to_idx = {'ir': 0, 'co': 1}
        self.df = pd.read_csv(root_path)
        self.img_size = img_size

        # Cropping strategy
        self.paired_crop = PairedRandomCrop(img_size, contained=True) if train else PairedCenterCrop(img_size)


    def __getitem__(self, index):
        if self.mode=='mono':
            idx = index.copy()
            index = index % len(self.df)
            modal, label = ("ir", 0) if idx % 2 else ("co", 1)
            modal1 = modal2 = modal
            label_ir = label_co = label
        else:
            modal1, label_1 = "ir", 0
            modal2, label_2 = "co", 1

        sample = self.df.iloc[index]['image_number']

        angle = self.df.iloc[index]['angle']
        translate = (self.df.iloc[index]['translate_x'], self.df.iloc[index]['translate_y'])
        scale = self.df.iloc[index]['scale']
        shear = self.df.iloc[index]['shear']

        overlap = self.df.iloc[index]['overlap']

        path_1 = f"/mnt/gpid08/datasets/registrat/VEDAI/Vehicules512/{modal1}/{sample:08d}_{modal1}.png"
        path_2 = f"/mnt/gpid08/datasets/registrat/VEDAI/Vehicules512/{modal2}/{sample:08d}_{modal2}.png"

        image_1 = Image.open(path_1).convert('L')
        image_2 = Image.open(path_2).convert('L')
                
        # Apply the transformations to the original image
        image_1 = self.transforms(image_1)
        image_2 = self.transforms(image_2)
        image_depth, image_height, image_width = image_2.shape

        # Apply affine transformation
        transformed_image_1 = F.affine(image_1, angle, translate, scale, [shear, shear])
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
        label_1 = torch.tensor(label_1, dtype=torch.long)
        label_2 = torch.tensor(label_2, dtype=torch.long)

        # Inverse transformation matrix
        inv_affine = torch.tensor([
            [inverse_matrix[0], inverse_matrix[1], inverse_matrix[2]],
            [inverse_matrix[3], inverse_matrix[4], inverse_matrix[5]],
            [0, 0, 1]
        ], dtype=torch.float32, device=inverse_matrix.device)
        forward_affine = torch.inverse(inv_affine)
        forward_matrix = forward_affine[:2, :].reshape(-1)

        # Normalize Affine Transformation Matrix
        forward_matrix_norm = normalize_affine_matrix(forward_matrix.squeeze().view(2, 3), image_width, image_height).view_as(forward_matrix)
        inverse_matrix_norm = normalize_affine_matrix(inverse_matrix.squeeze().view(2, 3), image_width, image_height).view_as(inverse_matrix)

        # Crop images
        bbox_original = np.array([[0, 0], [image_2.shape[2], 0], [image_2.shape[2], image_2.shape[1]], [0, image_2.shape[1]]])
        bbox_transformed = apply_normalized_affine_to_polygon(bbox_original, forward_matrix_norm.numpy().reshape(2, 3), image_2.shape[1], image_2.shape[2])
        image_1_crop, transformed_image_2_crop, offset1, offset2, h, w, contained = self.paired_crop(image_1, transformed_image_2, bbox=bbox_transformed)
        _, crop_height, crop_width = image_1_crop.shape

        # Adjust transformation to crop size
        forward_matrix_crop = adjust_affine_transformation(forward_matrix, offset1, offset2)
        inverse_matrix_crop = adjust_affine_transformation(inverse_matrix, offset1, offset2)

        # Normalize Crop Affine Transformation Matrix
        forward_matrix_crop_norm = normalize_affine_matrix(forward_matrix_crop.squeeze().view(2, 3), crop_width, crop_height).view_as(forward_matrix_crop)
        inverse_matrix_crop_norm = normalize_affine_matrix(inverse_matrix_crop.squeeze().view(2, 3), crop_width, crop_height).view_as(inverse_matrix_crop)

        # Create masks
        mask_original, mask_transformed = create_masks(offset1, offset2, h, w, (image_depth, image_height, image_width), inverse_matrix_norm, forward_matrix_norm, inverse_matrix_crop_norm, forward_matrix_crop_norm, contained)
        num_true_r = mask_original.sum().item()
        overlap_compute_r = num_true_r*100/(256*256)
        num_true_s = mask_transformed.sum().item()
        overlap_compute_s = num_true_s*100/(256*256)

        return image_1_crop, transformed_image_2_crop, label_1, label_2, inverse_matrix_crop_norm, forward_matrix_crop_norm, mask_original, mask_transformed


    def __len__(self):
        return len(self.df) * (2 if self.mode=='mono' else 1)