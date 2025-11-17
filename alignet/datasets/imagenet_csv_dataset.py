import numpy as np
import pandas as pd
from PIL import Image

import torch
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


@DATASET_REGISTRY.register("ImageNetCSVDataset")
class ImageNetCSVDataset(BaseDataset):
    def __init__(self, root_path, img_size, train=True, cfg=None):
        super().__init__(cfg=cfg, train=train)

        # Load dataframe
        self.dict = pd.read_csv(root_path, sep=None, engine="python")
        self.file_paths = self.dict["image_path"].unique()
        self.img_size = img_size
        self.labels = [0]  # placeholder
        self.train = train
        self.radiometric_augmentation = cfg.get("rad_aug", True)
        self.mode = (cfg or {}).get("mode", "mono")
        if self.mode == 'multi':
            print(f"[Warning] ImageNet cannot be used with multimodal training. Switching to mono.", flush=True)

        # Cropping strategy
        self.paired_crop = PairedRandomCrop(img_size, contained=True) if train else PairedCenterCrop(img_size)

    def __getitem__(self, index):
        # Get row with both image and transform info
        row = self.dict.iloc[index]
        file_path = row["image_path"]

        with Image.open(file_path).convert("L") as image_pil:
            # Apply base transforms (tensor + normalization)
            image = self.transforms(image_pil)
            image_depth, image_height, image_width = image.shape

            if self.homography == 'affine':
                angle = row["angle"]
                translate = (row["translate_x"], row["translate_y"])
                scale = row["scale"]
                shear = row["shear"]

                # Apply affine transformation
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

                # Inverse + forward matrices
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

            elif self.homography == 'projective':
                cols = ["h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21"]
                row_values = self.dict.loc[index, cols].tolist()
                row_values.append(1.0)
                tensor_vals = torch.tensor(row_values, dtype=torch.float32)
                H = tensor_vals.reshape(3, 3)
                transformed_image, forward_affine = apply_random_homography(image, H)
                forward_matrix = forward_affine.reshape(9)[:-1]
                inv_affine = torch.inverse(forward_affine)
                inv_affine = inv_affine / inv_affine[2,2]
                inverse_matrix = inv_affine.reshape(9)[:-1]
                assert not torch.isnan(forward_matrix).any(), "NaNs detected in forward_matrix!"
                assert not torch.isnan(inverse_matrix).any(), "NaNs detected in inverse_matrix!"

            # Radiometric augmentation
            if self.train and self.radiometric_augmentation:
                image = self.data_augmentation(image)
                transformed_image = self.data_augmentation(transformed_image)

            # Normalize affine matrices
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

            # Normalize Crop Affine Transformation Matrix
            forward_matrix_crop_norm = normalize_affine_matrix(forward_matrix_crop.squeeze(), crop_width, crop_height)
            inverse_matrix_crop_norm = normalize_affine_matrix(inverse_matrix_crop.squeeze(), crop_width, crop_height)

            # Create masks
            mask_original, mask_transformed = create_masks(offset1, offset2, h, w, (image_depth, image_height, image_width), inverse_matrix_norm, forward_matrix_norm, inverse_matrix_crop_norm, forward_matrix_crop_norm, contained)

        return (
            image_crop, transformed_image_crop,
            label, label,
            inverse_matrix_crop_norm, forward_matrix_crop_norm,
            mask_original, mask_transformed,
            
        )

    def __len__(self):
        return len(self.dict)
