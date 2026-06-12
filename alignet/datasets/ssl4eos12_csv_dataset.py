import os
import math
import rasterio
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torchvision.transforms.functional as F

from datasets.base_dataset import BaseDataset
from datasets.utils import PairedRandomCrop, PairedCenterCrop
from utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register("SSL4EOS12CSVDataset")
class SSL4EOS12CSVDataset(BaseDataset):
    def __init__(self, root_path, img_size, train=True, cfg=None):
        super().__init__(cfg=cfg, train=train, img_size=img_size)

        self.df = pd.read_csv(root_path, sep=None, engine="python")
        self.img_size = img_size
        self.label_to_idx = {"s1": 0, "s2a": 1}

    def read_geotiff(self, file_path):
        file_path = str(file_path)
        with rasterio.open(file_path) as dataset:
            image = dataset.read(1)
        return image[:, :, np.newaxis].astype(np.float32)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        label_1_str = row["label"]
        label_2_str = row["label2"] if self.mode == "multi" else row["label"]

        path_1 = os.path.join(self.data_path, row["image_path"])
        path_2 = os.path.join(self.data_path, row["image_path2"] if self.mode == "multi" else row["image_path"])

        img1 = self.read_geotiff(path_1)
        image_1 = self.transforms_percentile(img1)
        img2 = self.read_geotiff(path_2)
        image_2 = self.transforms_percentile(img2)

        # resize 264x264 original images to a bigger size
        new_h = math.ceil(image_1.shape[1] * 1.75)
        new_w = math.ceil(image_1.shape[2] * 1.75)
        image_1 = F.resize(image_1, [new_h, new_w])
        image_2 = F.resize(image_2, [new_h, new_w])

        image_depth, image_height, image_width = image_1.shape
        image_shape = image_2.shape  # (C, H, W)

        label_1 = torch.tensor(self.label_to_idx[label_1_str], dtype=torch.long)
        label_2 = torch.tensor(self.label_to_idx[label_2_str], dtype=torch.long)

        if self.homography == "affine":
            angle = row["angle"]
            translate = (row["translate_x"], row["translate_y"])
            scale = row["scale"]
            shear = row["shear"]
            transformed_image_2 = F.affine(image_2, angle, translate, scale, [shear, shear])
            forward_matrix, inverse_matrix = self._build_affine_matrices(image_2, angle, translate, scale, shear)

        elif self.homography == "projective":
            cols = ["h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21"]
            # row is a dict (list-of-dicts format), so access by key
            H = torch.tensor([row[c] for c in cols] + [1.0], dtype=torch.float32).reshape(3, 3)
            transformed_image_2, forward_matrix, inverse_matrix = self._build_projective_matrices_from_H(image_2, H)

        image_1_crop, transformed_image_2_crop, inv_norm, fwd_norm, mask1, mask2 = self._crop_and_finalize(
            image_1, transformed_image_2, forward_matrix, inverse_matrix, image_shape
        )

        return image_1_crop, transformed_image_2_crop, label_1, label_2, inv_norm, fwd_norm, mask1, mask2 #, row["image_path"], row["image_path2"]

    def __len__(self):
        return len(self.df)
