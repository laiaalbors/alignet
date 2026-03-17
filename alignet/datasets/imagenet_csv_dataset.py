import numpy as np
import pandas as pd
from PIL import Image

import torch
import torchvision.transforms.functional as F

from datasets.base_dataset import BaseDataset
from datasets.utils import PairedRandomCrop, PairedCenterCrop
from utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register("ImageNetCSVDataset")
class ImageNetCSVDataset(BaseDataset):
    def __init__(self, root_path, img_size, train=True, cfg=None):
        super().__init__(cfg=cfg, train=train, img_size=img_size)

        self.df = pd.read_csv(root_path, sep=None, engine="python")
        self.img_size = img_size
        self.labels = [0]  # placeholder

        if self.mode == "multi":
            print("[Warning] ImageNet cannot be used with multimodal training. Switching to mono.", flush=True)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        with Image.open(os.path.join(self.data_path, row["image_path"])) as image_pil:
            image_pil = image_pil.convert("RGB" if self.train else "L")
            image = self.transforms(image_pil)

        image_depth, image_height, image_width = image.shape
        label = torch.tensor(self.labels[0], dtype=torch.long)

        if self.homography == "affine":
            angle = row["angle"]
            translate = (row["translate_x"], row["translate_y"])
            scale = row["scale"]
            shear = row["shear"]
            image2 = F.affine(image, angle, translate, scale, [shear, shear])
            forward_matrix, inverse_matrix = self._build_affine_matrices(image, angle, translate, scale, shear)

        elif self.homography == "projective":
            cols = ["h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21"]
            H = torch.tensor(self.df.loc[index, cols].tolist() + [1.0], dtype=torch.float32).reshape(3, 3)
            image2, forward_matrix, inverse_matrix = self._build_projective_matrices_from_H(image, H)

        image1 = self._apply_augmentation(image)
        image2 = self._apply_augmentation(image2)

        image1_crop, image2_crop, inv_norm, fwd_norm, mask1, mask2 = self._crop_and_finalize(
            image1, image2, forward_matrix, inverse_matrix, (image_depth, image_height, image_width)
        )

        return image1_crop, image2_crop, label, label, inv_norm, fwd_norm, mask1, mask2

    def __len__(self):
        return len(self.df)
