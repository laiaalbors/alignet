import random
import itertools
import numpy as np
from pathlib import Path

import torch
import torchvision.transforms.functional as F

import rasterio
from skimage.transform import resize, rescale

from datasets.base_dataset import BaseDataset
from datasets.utils import PairedRandomCrop, get_shared_cache
from utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register("InferenceDataset")
class InferenceDataset(BaseDataset):
    def __init__(self, root_path, img_size, train=True, target_resolution=5, cfg=None, preloaded_cache=None):
        super().__init__(cfg=cfg, train=train)

        self.root_path = Path(root_path)
        self.img_size = img_size


    def read_geotiff(self, file_path):
        file_path = str(file_path)
        with rasterio.open(file_path) as dataset:
            image = dataset.read(1)
        return image[:, :, np.newaxis].astype(np.float32)


    def __getitem__(self, index):
        # imagepath_1 = self.root_path / "img_r.png"
        # imagepath_2 = self.root_path / "img_s.png"
        imagepath_1 = self.root_path / "before.jpg" #"MS_crop.tif"
        imagepath_2 = self.root_path / "after.jpg" #"WV_corregida_crop2.tif"         # "WV_corregida_crop2.tif" / "WV_original_crop.tif"
        
        image1 = self.read_geotiff(imagepath_1)
        image1 = rescale(image1, 0.1)
        # image1 = resize(image1, (self.img_size, self.img_size), order=1, mode='reflect', anti_aliasing=True)
        image2 = self.read_geotiff(imagepath_2)
        image2 = rescale(image2, 0.1)
        # image2 = resize(image2, (self.img_size, self.img_size), order=1, mode='reflect', anti_aliasing=True)

        # image1 = torch.from_numpy(image1).permute(2, 0, 1)
        # image2 = torch.from_numpy(image2).permute(2, 0, 1)
        image1 = self.transforms_percentile(image1)
        image2 = self.transforms_percentile(image2)

        image_shape = image1.shape  # (C, H, W)

        inv_norm = torch.eye(3,3).reshape(9)[:-1]
        fwd_norm = torch.eye(3,3).reshape(9)[:-1]

        label_1 = torch.tensor(0, dtype=torch.long)
        label_2 = torch.tensor(1, dtype=torch.long)

        mask1 = torch.ones(image1.shape, dtype=image1.dtype)
        mask2 = torch.ones(image2.shape, dtype=image2.dtype)

        return image1, image2, label_1, label_2, inv_norm, fwd_norm, mask1, mask2


    def __len__(self):
        return 1
