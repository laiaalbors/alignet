import torch
from typing import Tuple
import torch.nn.functional as F

def STN(img, pre_tps):
    aff_mat = pre_tps.reshape(-1, 2, 3)
    img_grid = F.affine_grid(aff_mat, img.size(), align_corners=True)
    img_reg = F.grid_sample(img, img_grid, align_corners=True)
    return img_reg

def STN_mask(img, pre_tps):
    aff_mat = pre_tps.reshape(-1, 2, 3)
    img_grid = F.affine_grid(aff_mat, img.size(), align_corners=True)
    d_type = img.dtype
    img_reg = F.grid_sample(img.to(dtype=torch.float32), img_grid.to(dtype=torch.float32), mode="nearest", padding_mode="zeros", align_corners=True).to(dtype=d_type)
    return img_reg
