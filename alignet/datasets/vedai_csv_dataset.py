import numpy as np
import pandas as pd
from PIL import Image

import torch
import torchvision.transforms.functional as F

from datasets.base_dataset import BaseDataset
from datasets.utils import PairedRandomCrop, PairedCenterCrop
from utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register("VEDAICSVDataset")
class VEDAICSVDataset(BaseDataset):
    def __init__(self, root_path, img_size, train=True, cfg=None):
        super().__init__(cfg=cfg, train=train, img_size=img_size)

        if cfg is None or "mode" not in cfg:
            print("[WARNING] Mode not specified in cfg. Defaulting to 'multi'.", flush=True)
        # VEDAICSVDataset defaults to 'multi' (override base class which defaults to 'mono')
        self.mode = (cfg or {}).get("mode", "multi")

        self.label_to_idx = {"ir": 0, "co": 1}
        self.df = pd.read_csv(root_path)
        self.img_size = img_size

    def __getitem__(self, index):
        if self.mode == "mono":
            # Alternate between IR and CO modalities based on index parity
            base_idx = index % len(self.df)
            modal = "ir" if index % 2 == 0 else "co"
            modality1 = modality2 = modal
            label_1_idx = label_2_idx = self.label_to_idx[modal]
        else:
            base_idx = index
            modality1, label_1_idx = "ir", 0
            modality2, label_2_idx = "co", 1

        sample = int(self.df.iloc[base_idx]["image_number"])
        path_1 = f"data/VEDAI/{modality1}/{sample:08d}_{modality1}.png"
        path_2 = f"data/VEDAI/{modality2}/{sample:08d}_{modality2}.png"

        image_1 = self.transforms(Image.open(path_1).convert("L"))
        image_2 = self.transforms(Image.open(path_2).convert("L"))
        image_shape = image_2.shape

        label_1 = torch.tensor(label_1_idx, dtype=torch.long)
        label_2 = torch.tensor(label_2_idx, dtype=torch.long)

        if self.homography == "affine":
            row = self.df.iloc[base_idx]
            angle = row["angle"]
            translate = (row["translate_x"], row["translate_y"])
            scale = row["scale"]
            shear = row["shear"]
            transformed_image_2 = F.affine(image_2, angle, translate, scale, [shear, shear])
            forward_matrix, inverse_matrix = self._build_affine_matrices(image_2, angle, translate, scale, shear)

        elif self.homography == "projective":
            cols = ["h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21"]
            H = torch.tensor(
                self.df.loc[base_idx, cols].tolist() + [1.0], dtype=torch.float32
            ).reshape(3, 3)
            transformed_image_2, forward_matrix, inverse_matrix = self._build_projective_matrices_from_H(image_2, H)

        image_1_crop, transformed_image_2_crop, inv_norm, fwd_norm, mask1, mask2 = self._crop_and_finalize(
            image_1, transformed_image_2, forward_matrix, inverse_matrix, image_shape
        )

        return image_1_crop, transformed_image_2_crop, label_1, label_2, inv_norm, fwd_norm, mask1, mask2

    def __len__(self):
        if self.mode == "mono":
            return len(self.df) * 2
        return len(self.df)
