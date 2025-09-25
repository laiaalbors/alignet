import numpy as np
from typing import Tuple

import os
import torch
import torchvision.transforms.functional as F

from utils.transforms import STN, STN_mask

def inverse_affine_matrix(affine_matrix, device):
    b = affine_matrix.shape[0]
    idx3 = torch.Tensor([[[0,0,1]]]).to(device).repeat(b, 1, 1)
    affine_matrix = affine_matrix.reshape(-1, 2, 3)
    affine_matrix = torch.cat((affine_matrix, idx3), dim=1)
    try:
        affine_matrix_inverse = torch.linalg.inv(affine_matrix)
    except:
        print(f"\n[inverse_affine_matrix] matrix not invertible, returning identity. batchsize={b}", flush=True)
        return torch.eye(2,3, device=device).reshape(1,6).repeat(b, 1)
    return affine_matrix_inverse[:, :2, :].reshape(b, 6)


def normalize_affine_matrix(matrix, width=256, height=256):
    # Convert 2x3 to 3x3
    matrix_h = torch.cat([matrix, torch.tensor([[0, 0, 1]], device=matrix.device, dtype=matrix.dtype)], dim=0)
    T = torch.tensor([[width/2, 0, width/2],
                      [0, height/2, height/2],
                      [0, 0, 1]], device=matrix.device, dtype=matrix.dtype)
    T_inv = torch.tensor([[2/width, 0, -1],
                          [0, 2/height, -1],
                          [0, 0, 1]], device=matrix.device, dtype=matrix.dtype)
    normalized_matrix = T_inv @ matrix_h @ T
    return normalized_matrix[:2, :]


def adjust_affine_transformation(affine_params, offset1, offset2):
    # 3×3 crop←big and big←crop
    H_uncrop = np.array([[1,0, offset2],
                         [0,1, offset1],
                         [0,0, 1]])
    H_crop   = np.array([[1,0,-offset2],
                         [0,1,-offset1],
                         [0,0, 1]])
    # lift to 3×3
    A = np.vstack([affine_params.reshape(2,3),
                   [0,0,1]])
    M = H_crop.dot(A.dot(H_uncrop))
    return torch.from_numpy(M[:2, :]).float().view_as(affine_params)


def apply_normalized_affine_to_polygon(polygon, affine_matrix, H, W, scale=1):
    polygon = np.array(polygon, dtype=np.float32)

    # 1) normalize to [-1,1]
    x = (polygon[:,0]/(W-1))*2 - 1
    y = (polygon[:,1]/(H-1))*2 - 1
    homo = np.vstack([x, y, np.ones_like(x)])         # (3,N)

    # 2) apply normalized affine
    trans_norm = (affine_matrix @ homo).T            # (N,2)

    # 3) denormalize to pixels
    px = (trans_norm[:,0]+1)/2*(W-1)
    py = (trans_norm[:,1]+1)/2*(H-1)
    transformed = np.stack([px, py], axis=1)

    # 4) now scale **around** image center
    if scale != 1:
        cx, cy = (W-1)/2, (H-1)/2
        transformed = (transformed - [cx, cy]) * scale + [cx, cy]

    return transformed


def create_masks(
    i: int,
    j: int,
    h: int,
    w: int,
    shape: Tuple[int, int, int],
    gt_inverse: torch.Tensor,
    gt_forward: torch.Tensor,
    gt_inverse_crop: torch.Tensor,
    gt_forward_crop: torch.Tensor,
    contained: bool
) -> Tuple[torch.Tensor, torch.Tensor]:

    if not contained:
        ones_images = torch.ones(shape, dtype=torch.float32)
        transformed_ones_images = STN_mask(ones_images.unsqueeze(0), gt_inverse).squeeze(0)

        ones_images_crop = F.crop(ones_images, i, j, h, w)
        transformed_ones_images_crop = F.crop(transformed_ones_images, i, j, h, w)
    else:
        ones_images_crop = torch.ones((1, h, w), dtype=torch.float32)
        transformed_ones_images_crop = torch.ones((1, h, w), dtype=torch.float32)

    mask_original = STN_mask(transformed_ones_images_crop.unsqueeze(0), gt_forward_crop).squeeze(0)
    mask_transformed = STN_mask(ones_images_crop.unsqueeze(0), gt_inverse_crop).squeeze(0)

    mask_original = torch.gt(mask_original, 0)
    mask_transformed = torch.gt(mask_transformed, 0)

    return mask_original, mask_transformed
