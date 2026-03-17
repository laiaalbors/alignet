import random
import kornia
import numpy as np
from typing import Tuple

import os
import torch
import torchvision.transforms.functional as F

from utils.transforms import STN, STN_mask


def apply_random_homography(image, H=None, max_displacement=40):
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    B, C, Ht, Wt = image.shape
    if H is None:
        H = random_homography(Ht, Wt, max_displacement=max_displacement)
    H = H.unsqueeze(0).to(image.device)
    warped = kornia.geometry.transform.warp_perspective(
        image, H, dsize=(Ht, Wt), align_corners=True
    )
    return warped.squeeze(0), H.squeeze()


def random_homography(height, width, max_displacement=10):
    """Generate a random homography matrix by perturbing the corners."""
    # Original corners in normalized coordinates [-1, 1]
    src_pts = torch.tensor([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ], dtype=torch.float)

    # Random displacement of each corner
    dst_pts = src_pts + torch.empty_like(src_pts).uniform_(-max_displacement, max_displacement)

    # Compute homography from 4 point correspondences
    H = kornia.geometry.homography.find_homography_dlt(
        src_pts.unsqueeze(0),  # [1, 4, 2]
        dst_pts.unsqueeze(0)   # [1, 4, 2]
    )[0]

    H = H/H[2,2]

    return H


def inverse_affine_matrix(affine_matrix_input, device):
    b = affine_matrix_input.shape[0]

    if affine_matrix_input.shape[-1] == 6:
        idx3 = torch.Tensor([[[0,0,1]]]).to(device).repeat(b, 1, 1)
        affine_matrix = affine_matrix_input.reshape(-1, 2, 3)
        affine_matrix = torch.cat((affine_matrix, idx3), dim=1)
    elif affine_matrix_input.shape[-1] == 8:
        ones = torch.ones(b, 1, device=device, dtype=affine_matrix_input.dtype)
        affine_matrix = torch.cat([affine_matrix_input, ones], dim=1)
        affine_matrix = affine_matrix.reshape(-1, 3, 3)
    else:
        raise ValueError(f"Expected 6 (Affine) or 8 (Projective) prediction parameters, but got {affine_matrix_input.shape[-1]}.")
    
    try:
        affine_matrix_inverse = torch.linalg.inv(affine_matrix)
    except:
        print(f"\n[inverse_affine_matrix] matrix not invertible, returning identity. batchsize={b}", flush=True)
        return torch.eye(2,3, device=device).reshape(1,6).repeat(b, 1)
    
    if affine_matrix_input.shape[-1] == 6:
        return affine_matrix_inverse[:, :2, :].reshape(b, 6)
    else:
        return affine_matrix_inverse.reshape(b, 9)[:, :-1]


def normalize_affine_matrix(matrix_input, width=256, height=256):
    if matrix_input.shape[-1] == 6:
        # Convert 2x3 to 3x3
        matrix = matrix_input.squeeze().view(2, 3)
        matrix_h = torch.cat([matrix, torch.tensor([[0, 0, 1]], device=matrix.device, dtype=matrix.dtype)], dim=0)
    elif matrix_input.shape[-1] == 8:
        ones = torch.ones(1, device=matrix_input.device, dtype=matrix_input.dtype)
        matrix = torch.cat([matrix_input, ones], dim=0)
        matrix_h = matrix.reshape(3, 3)
    else:
        raise ValueError(f"Expected 6 (Affine) or 8 (Projective) prediction parameters, but got {matrix_input.shape[-1]}.")

    T = torch.tensor([[width/2, 0, width/2],
                      [0, height/2, height/2],
                      [0, 0, 1]], device=matrix.device, dtype=matrix.dtype)
    T_inv = torch.tensor([[2/width, 0, -1],
                          [0, 2/height, -1],
                          [0, 0, 1]], device=matrix.device, dtype=matrix.dtype)
    normalized_matrix = T_inv @ matrix_h @ T
    # return normalized_matrix[:2, :]

    if matrix_input.shape[-1] == 6:
        return normalized_matrix[:2, :].reshape(6)
    else:
        return normalized_matrix.reshape(9)[:-1]


def adjust_affine_transformation(homo_params, offset1, offset2):
    #o = (original_size - crop_size) / 2.0
    # 3×3 crop←big and big←crop
    H_uncrop = np.array([[1,0, offset2],
                         [0,1, offset1],
                         [0,0, 1]])
    H_crop   = np.array([[1,0,-offset2],
                         [0,1,-offset1],
                         [0,0, 1]])
    # lift to 3×3
    if homo_params.shape[-1] == 6:
        A = np.vstack([homo_params.reshape(2,3),
                      [0,0,1]])
    elif homo_params.shape[-1] == 8:
        A = np.concatenate([homo_params, [1]], axis=0).reshape(3, 3)
    else:
        raise ValueError(f"Expected 6 (Affine) or 8 (Projective) prediction parameters, but got {homo_params.shape[-1]}.")

    M = H_crop.dot(A.dot(H_uncrop))

    if homo_params.shape[-1] == 6:
        return torch.from_numpy(M[:2, :]).float().view_as(homo_params)
    else:
        return torch.from_numpy(M.reshape(9)[:-1]).float()


def apply_normalized_affine_to_polygon(polygon, affine_matrix, H, W, scale=1):
    polygon = np.array(polygon, dtype=np.float32)

    # ensure it's numpy
    affine_matrix = np.array(affine_matrix, dtype=np.float32)

    if affine_matrix.size == 6:
        affine_matrix = affine_matrix.reshape(2, 3)
        affine_matrix = np.vstack([affine_matrix, [0, 0, 1]])
    elif affine_matrix.size == 8:
        affine_matrix = np.concatenate([affine_matrix, [1]], axis=0).reshape(3, 3)

    # 1) Normalize to [-1,1]
    x = (polygon[:, 0] / (W - 1)) * 2 - 1
    y = (polygon[:, 1] / (H - 1)) * 2 - 1
    homo = np.vstack([x, y, np.ones_like(x)])  # (3, N)

    # 2) Apply normalized affine or homography
    trans_norm = affine_matrix @ homo
    trans_norm /= trans_norm[2:3, :]  # divide by w  (safe for affine too)
    trans_norm = trans_norm[:2, :].T  # (N, 2)

    # 3) Denormalize to pixel coordinates
    px = (trans_norm[:, 0] + 1) / 2 * (W - 1)
    py = (trans_norm[:, 1] + 1) / 2 * (H - 1)
    transformed = np.stack([px, py], axis=1)

    # 4) Optional scaling around center
    if scale != 1:
        cx, cy = (W - 1) / 2, (H - 1) / 2
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
