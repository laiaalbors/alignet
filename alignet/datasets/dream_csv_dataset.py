import os
import numpy as np
import pandas as pd

import torch
import torchvision.transforms.functional as F

from datasets.base_dataset import BaseDataset
from datasets.utils import PairedRandomCrop, PairedCenterCrop
from utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register("DREAMCSVDataset")
class DREAMCSVDataset(BaseDataset):
    def __init__(self, root_path, img_size, train=True, cfg=None):
        super().__init__(cfg=cfg, train=train, img_size=img_size)

        if cfg is None or "mode" not in cfg:
            print("[WARNING] Mode not specified in cfg. Defaulting to 'mono'.", flush=True)

        self.data_path = cfg["data_path"]
        self.img_size = img_size
        self.label_to_idx = {"optique": 0, "radar": 1, "S1": 2, "S2": 3, "SRTM": 4}
        self._sar_idx = self.label_to_idx["S1"]

        records = pd.read_csv(root_path)
        records = records[int(cfg.get("in", 0)):int(cfg.get("fi", len(records)))]
        self.df_columns = records.columns
        # Store as list-of-dicts for fast row access
        self.df = records.to_dict(orient="records")
        del records

    def read_npy(self, file_path):
        arr = np.load(file_path, mmap_mode="r")
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        elif arr.ndim != 3:
            raise ValueError(f"Unexpected npy shape {arr.shape} for {file_path}")
        return np.ascontiguousarray(arr)

    def _normalize_image(self, image, label_str):
        if label_str == "S1":
            return self.transforms_percentile(image)
        return self.transforms(image)

    def __getitem__(self, index):
        row = self.df[index]
        label_1_str = row["label"]
        label_2_str = row["label2"] if self.mode == "multi" else row["label"]

        path_1 = os.path.join(self.data_path, row["zone"], label_1_str, row["image_path"])
        path_2 = os.path.join(self.data_path, row["zone"], label_2_str, row["image_path"])

        image_1 = self.read_npy(path_1[:-3] + "npy")
        image_2 = self.read_npy(path_2[:-3] + "npy")

        image_1 = self._normalize_image(image_1, label_1_str)
        image_2 = self._normalize_image(image_2, label_2_str)
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

        if "offset1" in self.df_columns:
            crop_coords = (row["offset1"], row["offset2"], self.img_size, self.img_size, True)
        else:
            crop_coords = None

        image_1_crop, transformed_image_2_crop, inv_norm, fwd_norm, mask1, mask2 = self._crop_and_finalize(
            image_1, transformed_image_2, forward_matrix, inverse_matrix, image_shape,
            crop_coords=crop_coords,
        )

        return image_1_crop, transformed_image_2_crop, label_1, label_2, inv_norm, fwd_norm, mask1, mask2

    def __len__(self):
        return len(self.df)
