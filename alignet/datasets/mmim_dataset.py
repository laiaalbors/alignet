import os
import time
import random
import itertools
import numpy as np
from pathlib import Path
from scipy.io import loadmat

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F

from datasets.base_dataset import BaseDataset
from datasets.utils import PairedRandomCrop, PairedCenterCrop, get_shared_cache
from utils.utils import normalize_affine_matrix, create_masks, adjust_affine_transformation, apply_normalized_affine_to_polygon
from utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register("MMIMDataset")
class MMIMDataset(BaseDataset):
    def __init__(self, root_path, img_size, train=True, cfg=None):
        """
        Args:
            root_path (str or Path): Path to the root directory containing subfolders for each label.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        super().__init__(cfg=cfg, train=train, img_size=img_size)

        self.img_size = img_size
        
        if root_path[-4:] == '.mat':
            self.file_paths = [root_path]
        else:
            self.root = Path(root_path)
            self.file_paths = []
            for mat_file in self.root.rglob("*.mat"):
                self.file_paths.append(mat_file)
        
        self.paired_crop = PairedCenterCrop(img_size)
    

    def __getitem__(self, index):
        mat_path = self.file_paths[index]

        data = loadmat(mat_path)
        image = self.transforms(data["I_fix"][0])
        transformed_image = self.transforms(data["I_move"][0])
        image_depth, image_height, image_width = image.shape

        image_crop, transformed_image_crop, offset1, offset2, h, w, contained = self.paired_crop(image, transformed_image)

        label = torch.tensor(0, dtype=torch.long)

        T = data['T']   # ex: 'H', 'tform', 'transform', etc.
        T = T / T[2,2]
        T = T.T
        inverse_matrix = torch.from_numpy(T).float()
        forward_matrix = torch.from_numpy(np.linalg.inv(T)).float()

        inverse_matrix = inverse_matrix.reshape(9)[:-1]
        forward_matrix = forward_matrix.reshape(9)[:-1]

        inverse_matrix_norm = normalize_affine_matrix(inverse_matrix, image_width, image_height)
        forward_matrix_norm = normalize_affine_matrix(forward_matrix, image_width, image_height)

        inverse_matrix = adjust_affine_transformation(inverse_matrix, offset1, offset2)
        forward_matrix = adjust_affine_transformation(forward_matrix, offset1, offset2)

        inverse_matrix_crop_norm = normalize_affine_matrix(inverse_matrix, width=w, height=h)
        forward_matrix_crop_norm = normalize_affine_matrix(forward_matrix, width=w, height=h)

        # Create masks
        # mask_original, mask_transformed = create_masks(None, None, int(image_height), int(image_width), (1, int(image_height), int(image_width)), None, None, inverse_matrix, forward_matrix, True)
        mask_original, mask_transformed = create_masks(offset1, offset2, h, w, (image_depth, image_height, image_width), inverse_matrix_norm, forward_matrix_norm, inverse_matrix_crop_norm, forward_matrix_crop_norm, contained)

        return (
            image_crop, transformed_image_crop, 
            label, label, 
            inverse_matrix_crop_norm, forward_matrix_crop_norm, 
            mask_original, mask_transformed,
        )


    def __len__(self):
        return len(self.file_paths)