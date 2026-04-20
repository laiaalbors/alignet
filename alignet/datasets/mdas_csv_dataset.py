import os
import random
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

import torch
import torchvision.transforms.functional as F

import rasterio
from rasterio.warp import reproject
from rasterio.enums import Resampling

from datasets.base_dataset import BaseDataset
from datasets.utils import get_shared_cache
from utils.utils import apply_random_homography
from utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register("MDASCSVDataset")
class MDASCSVDataset(BaseDataset):
    def __init__(self, root_path, img_size, train=True, target_resolution=5, cfg=None, preloaded_cache=None):
        super().__init__(cfg=cfg, train=train)

        self.root = Path(root_path)
        self.target_resolution = target_resolution
        self.img_size = img_size
        self._cache = get_shared_cache()

        from datasets.utils import PairedRandomCrop, PairedCenterCrop
        self.paired_crop = PairedRandomCrop(self.img_size, contained=True)

        self.label_to_idx = {
            "EeteS_EnMAP_10m": 0, "EeteS_EnMAP_30m": 1, "Sentinel-1": 2,
            "3K_DSM": 3, "3K_RGB": 4, "EeteS_Sentinel_2_10m": 5, "HySpex": 6, "Sentinel-2": 7,
        }

        df = pd.read_csv(root_path)
        file_paths_per_label = {
            label: [df[df["label"] == label]["image_path"].unique()[0]]
            for label in df["label"].unique()
        }

        if "label2" not in df.columns:
            # Expand single-modality CSV to all ordered pairs
            label_permutations = list(itertools.permutations(file_paths_per_label.keys(), 2))
            file_paths_pairs = []
            labels_pairs = []
            for label1, label2 in label_permutations:
                for pair in zip(
                    sorted(file_paths_per_label[label1], key=str),
                    sorted(file_paths_per_label[label2], key=str),
                ):
                    file_paths_pairs.append(pair)
                labels_pairs.extend(
                    [[self.label_to_idx[label1], self.label_to_idx[label2]]]
                    * len(file_paths_per_label[label1])
                )

            idx_to_label = {v: k for k, v in self.label_to_idx.items()}
            mapping = defaultdict(list)
            for (fp1, fp2), (idx1, idx2) in zip(file_paths_pairs, labels_pairs):
                mapping[fp1].append((fp2, idx_to_label[idx2]))

            expanded_rows = []
            for _, row in df.iterrows():
                img1 = row["image_path"]
                for img2, label2 in mapping[img1]:
                    new_row = row.copy()
                    new_row["image_path2"] = img2
                    new_row["label2"] = label2
                    expanded_rows.append(new_row)
            df = pd.DataFrame(expanded_rows)
        else:
            for label in df["label2"].unique():
                if label not in self.label_to_idx:
                    fp = df[df["label2"] == label]["image_path2"].unique()[0]
                    file_paths_per_label[label] = [fp]

        print(f"self.label_to_idx: {self.label_to_idx}", flush=True)

        self.df = df
        paths1 = {os.path.join(self.data_path, p) for p in df["image_path"].unique()}
        paths2 = {os.path.join(self.data_path, p) for p in df["image_path2"].unique()}
        self.file_paths = list(paths1 | paths2)
        print(f"Unique image_path: {len(df['image_path'].unique())}", flush=True)
        print(f"Unique image_path2: {len(df['image_path2'].unique())}", flush=True)

    def resample_geotiff(self, file_path, min_res=None, max_res=None):
        file_path = str(file_path)
        with rasterio.open(file_path) as dataset:
            band_idx = 2 if "Sentinel-2" in file_path else 1
            image = dataset.read(band_idx)
            scale_factor = dataset.res[0] / self.target_resolution
            new_width = int(dataset.width * scale_factor)
            new_height = int(dataset.height * scale_factor)
            new_transform = dataset.transform * dataset.transform.scale(
                dataset.width / new_width, dataset.height / new_height
            )
            resampled = np.empty((new_height, new_width), dtype=dataset.dtypes[0])
            reproject(
                source=image,
                destination=resampled,
                src_transform=dataset.transform,
                dst_transform=new_transform,
                src_crs=dataset.crs,
                dst_crs=dataset.crs,
                resampling=Resampling.bilinear,
            )
        return resampled[:, :, np.newaxis].astype(np.float32)

    def __getitem__(self, index):
        import os
        row = self.df.iloc[index]
        file_path_r = os.path.join(self.data_path, row["image_path"])
        label_r_idx = self.label_to_idx[row["label"]]

        offset1 = row["offset1"]
        offset2 = row["offset2"]
        # h = row["h"]
        # w = row["w"]
        h, w = self.img_size, self.img_size

        image_r = self._cache[str(file_path_r)]
        if not isinstance(image_r, torch.Tensor):
            image_r = self.transforms_percentile(image_r)

        if self.mode == "multi":
            file_path_s = os.path.join(self.data_path, row["image_path2"])
            label_s_idx = self.label_to_idx[row["label2"]]
            image_s = self._cache[str(file_path_s)]
            if not isinstance(image_s, torch.Tensor):
                image_s = self.transforms_percentile(image_s)
        else:
            image_s = image_r
            label_s_idx = label_r_idx

        # Random flips
        if self.train and random.random() > 0.5:
            image_r = F.hflip(image_r)
            image_s = F.hflip(image_s)
        if self.train and random.random() > 0.5:
            image_r = F.vflip(image_r)
            image_s = F.vflip(image_s)

        image_shape = image_r.shape

        if self.homography == "affine":
            angle = row["angle"]
            translate = (row["translate_x"], row["translate_y"])
            scale = row["scale"]
            shear = row["shear"]
            transformed_image_s = F.affine(image_s, angle, translate, scale, [shear, shear])
            forward_matrix, inverse_matrix = self._build_affine_matrices(image_r, angle, translate, scale, shear)

        elif self.homography == "projective":
            cols = ["h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21"]
            H = torch.tensor(
                [row[c] for c in cols] + [1.0], dtype=torch.float32
            ).reshape(3, 3)
            transformed_image_s, forward_matrix, inverse_matrix = self._build_projective_matrices_from_H(image_s, H)

        label_r = torch.tensor(label_r_idx, dtype=torch.long)
        label_s = torch.tensor(label_s_idx, dtype=torch.long)

        image_r_crop, transformed_image_s_crop, inv_norm, fwd_norm, mask1, mask2 = self._crop_and_finalize(
            image_r, transformed_image_s, forward_matrix, inverse_matrix, image_shape,
            crop_coords=(offset1, offset2, h, w, False),
        )

        return image_r_crop, transformed_image_s_crop, label_r, label_s, inv_norm, fwd_norm, mask1, mask2

    def __len__(self):
        return len(self.df)
