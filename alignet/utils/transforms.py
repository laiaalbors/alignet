import torch
from typing import Tuple
import torch.nn.functional as F


def STN(img, pre_tps):
    # ensure batch dimension exists
    if pre_tps.ndim == 1:
        pre_tps = pre_tps.unsqueeze(0)
    if img.ndim == 3:
        img = img.unsqueeze(0)

    B = pre_tps.shape[0]

    if pre_tps.shape[-1] == 6:
        aff_mat = pre_tps.reshape(B, 2, 3)
        img_grid = F.affine_grid(aff_mat, img.size(), align_corners=True)
    elif pre_tps.shape[-1] == 8:
        ones = torch.ones((B, 1), device=pre_tps.device, dtype=pre_tps.dtype)
        homo_mat = torch.cat([pre_tps, ones], dim=1).reshape(B, 3, 3)
        img_grid = homography_grid(homo_mat, img.size(), align_corners=True)
    else:
        raise ValueError(f"Expected 6 (Affine) or 8 (Projective) prediction parameters, but got {pre_tps.shape[-1]}.")
    img_reg = F.grid_sample(img, img_grid, align_corners=True)
    return img_reg


def STN_mask(img, pre_tps):
    # ensure batch dimension exists
    if pre_tps.ndim == 1:
        pre_tps = pre_tps.unsqueeze(0)
    if img.ndim == 3:
        img = img.unsqueeze(0)

    B = pre_tps.shape[0]

    if pre_tps.shape[-1] == 6:
        aff_mat = pre_tps.reshape(B, 2, 3)
        img_grid = F.affine_grid(aff_mat, img.size(), align_corners=True)
    elif pre_tps.shape[-1] == 8:
        ones = torch.ones((B, 1), device=pre_tps.device, dtype=pre_tps.dtype)
        homo_mat = torch.cat([pre_tps, ones], dim=1).reshape(B, 3, 3)
        img_grid = homography_grid(homo_mat, img.size(), align_corners=True)
    else:
        raise ValueError(f"Expected 6 (Affine) or 8 (Projective) prediction parameters, but got {pre_tps.shape[-1]}.")
    d_type = img.dtype
    img_reg = F.grid_sample(img.to(dtype=torch.float32), img_grid.to(dtype=torch.float32), mode="nearest", padding_mode="zeros", align_corners=True).to(dtype=d_type)
    return img_reg


def homography_grid(H, size, align_corners=True):
    """
    Args:
        H: [B, 3, 3] batch of homography matrices
        size: output image size as (B, C, H, W)
    Returns:
        grid: [B, H, W, 2] sampling grid for F.grid_sample
    """
    B, _, _ = H.shape
    _, _, Ht, Wt = size

    # create normalized grid coordinates in [-1, 1]
    ys, xs = torch.linspace(-1, 1, Ht, device=H.device), torch.linspace(-1, 1, Wt, device=H.device)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    ones = torch.ones_like(xx)
    grid = torch.stack([xx, yy, ones], dim=-1).view(-1, 3).T  # [3, H*W]

    # apply homography
    grid = H @ grid  # [B, 3, H*W]
    grid = grid / (grid[:, 2:3, :] + 1e-8)  # divide by z

    # reshape to [B, H, W, 2]
    grid = grid[:, :2, :].permute(0, 2, 1).view(B, Ht, Wt, 2)

    return grid