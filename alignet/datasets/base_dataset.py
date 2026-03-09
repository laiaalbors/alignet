import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F

from datasets.utils import percentile_minmax, PairedRandomCrop, PairedCenterCrop
from datasets.augmentations import sensor_like_augmentation, build_traditional_augmentation
from utils.utils import (
    normalize_affine_matrix,
    create_masks,
    adjust_affine_transformation,
    apply_normalized_affine_to_polygon,
    apply_random_homography,
)


class BaseDataset(Dataset):
    """Base class for all ALIGNet datasets.

    Responsibilities:
    - Builds the normalization transform pipelines (transforms, transforms_percentile).
    - Wires up augmentation strategies (none / traditional / sensor-like).
    - Exposes shared helper methods for geometric matrix computation and the
      crop-and-finalize pipeline so subclasses avoid copy-pasting that logic.

    Subclasses only need to implement __getitem__ and __len__, using:
        _build_affine_matrices()
        _build_random_projective_matrices()
        _build_projective_matrices_from_H()
        _crop_and_finalize()
        _apply_augmentation()
    """

    def __init__(self, cfg=None, train=True, img_size=None):
        self.train = train

        # --- Common config attributes ---
        cfg = cfg or {}
        if "mode" not in cfg:
            pass  # subclasses may warn about this if relevant
        self.mode = cfg.get("mode", "mono")
        self.homography = cfg.get("homography", "affine")
        self.use_traditional_aug = cfg.get("traditional_augmentation", False)
        self.use_sensor_like_aug = cfg.get("sensor_like_augmentation", False)

        # Sensor-like augmentation object (the novel contribution)
        self.sensor_like_augmentation = sensor_like_augmentation

        # --- Optional paired crop (set when img_size is passed) ---
        if img_size is not None:
            self.paired_crop = (
                PairedRandomCrop(img_size, contained=True) if train else PairedCenterCrop(img_size)
            )

        # --- Normalization pipelines ---
        norm_type = cfg.get("norm", None)

        if norm_type == "min-max":
            # Base transforms: ToTensor + optional flips; normalization deferred to aug if aug is on
            base_tfs = [transforms.ToTensor()]
            if not self.use_traditional_aug and not self.use_sensor_like_aug:
                base_tfs.append(transforms.Lambda(lambda t: (t - t.min()) / (t.max() - t.min() + 1e-8)))
            if self.use_traditional_aug:
                base_tfs += [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
            self.transforms = transforms.Compose(base_tfs)

            self.transforms_percentile = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda t: percentile_minmax(t, low=1.0, high=99.0)),
            ])

            self.traditional_augmentation = build_traditional_augmentation("min-max")

        elif norm_type == "mean-std":
            mean = cfg.get("mean", None)
            std = cfg.get("std", None)
            assert mean is not None and std is not None, \
                "[ERROR] 'mean' and 'std' must be set when using 'mean-std' normalization."
            if not isinstance(mean, list):
                mean = list(mean) if hasattr(mean, "__iter__") and not isinstance(mean, str) else [mean]
            if not isinstance(std, list):
                std = list(std) if hasattr(std, "__iter__") and not isinstance(std, str) else [std]

            base_tfs = [transforms.ToTensor()]
            if not self.use_traditional_aug and not self.use_sensor_like_aug:
                base_tfs.append(transforms.Normalize(mean=mean, std=std))
            if self.use_traditional_aug:
                base_tfs += [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
            self.transforms = transforms.Compose(base_tfs)

            self.transforms_percentile = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])

            self.traditional_augmentation = build_traditional_augmentation("mean-std", mean=mean, std=std)

        else:
            if norm_type is not None:
                print(f"[WARNING] Unknown norm '{norm_type}'. No normalization applied "
                      f"to the {'train' if self.train else 'validation'} data.", flush=True)
            else:
                print(f"[WARNING] No normalization is being applied to the "
                      f"{'train' if self.train else 'validation'} input data.", flush=True)
            self.transforms = transforms.ToTensor()
            self.transforms_percentile = transforms.ToTensor()
            self.traditional_augmentation = None

    # ------------------------------------------------------------------
    # Augmentation dispatch
    # ------------------------------------------------------------------

    def _apply_augmentation(self, image):
        """Apply the configured augmentation strategy to a single image tensor.

        Returns the image unchanged when training is off or no augmentation is set.
        """
        if not self.train:
            return image
        if self.use_sensor_like_aug:
            return self.sensor_like_augmentation(image)
        if self.use_traditional_aug:
            return self.traditional_augmentation(image)
        return image

    # ------------------------------------------------------------------
    # Geometric matrix helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_affine_matrices(image, angle, translate, scale, shear):
        """Compute forward and inverse 2×3 affine matrices for the given parameters.

        Args:
            image: Tensor (C, H, W) used only to read the image center.
            angle, translate, scale, shear: affine parameters.

        Returns:
            forward_matrix: flat Tensor[6]
            inverse_matrix: flat Tensor[6]
        """
        inv_flat = F._get_inverse_affine_matrix(
            center=(image.shape[2] / 2, image.shape[1] / 2),
            angle=angle,
            translate=translate,
            scale=scale,
            shear=[shear, shear],
        )
        inverse_matrix = torch.tensor(inv_flat, dtype=torch.float32)
        inv_3x3 = torch.tensor([
            [inverse_matrix[0], inverse_matrix[1], inverse_matrix[2]],
            [inverse_matrix[3], inverse_matrix[4], inverse_matrix[5]],
            [0.0, 0.0, 1.0],
        ], dtype=torch.float32)
        forward_matrix = torch.inverse(inv_3x3)[:2, :].reshape(-1)
        return forward_matrix, inverse_matrix

    @staticmethod
    def _build_random_projective_matrices(image, max_displacement=80):
        """Apply a random projective (homography) transformation.

        Retries until no NaN values appear in the result.

        Returns:
            warped_image: Tensor (C, H, W)
            forward_matrix: flat Tensor[8]
            inverse_matrix: flat Tensor[8]
        """
        warped, H = apply_random_homography(image, max_displacement=max_displacement)
        while torch.isnan(H).any():
            warped, H = apply_random_homography(image, max_displacement=max_displacement)
        forward_matrix = H.reshape(9)[:-1]
        inv_H = torch.inverse(H)
        inv_H = inv_H / inv_H[2, 2]
        inverse_matrix = inv_H.reshape(9)[:-1]
        assert not torch.isnan(forward_matrix).any(), "NaNs in forward_matrix!"
        assert not torch.isnan(inverse_matrix).any(), "NaNs in inverse_matrix!"
        return warped, forward_matrix, inverse_matrix

    @staticmethod
    def _build_projective_matrices_from_H(image, H):
        """Apply a given 3×3 homography H to image_src.

        Returns:
            warped_image: Tensor (C, H, W)
            forward_matrix: flat Tensor[8]
            inverse_matrix: flat Tensor[8]
        """
        warped, forward_affine = apply_random_homography(image, H)
        forward_matrix = forward_affine.reshape(9)[:-1]
        inv_H = torch.inverse(forward_affine)
        inv_H = inv_H / inv_H[2, 2]
        inverse_matrix = inv_H.reshape(9)[:-1]
        assert not torch.isnan(forward_matrix).any(), "NaNs in forward_matrix!"
        assert not torch.isnan(inverse_matrix).any(), "NaNs in inverse_matrix!"
        return warped, forward_matrix, inverse_matrix

    # ------------------------------------------------------------------
    # Crop and finalize pipeline
    # ------------------------------------------------------------------

    def _crop_and_finalize(self, image1, image2, forward_matrix, inverse_matrix,
                           original_shape, crop_coords=None):
        """Crop both images and compute all normalized transformation matrices and masks.

        This is the common tail of every dataset's __getitem__, identical across all
        datasets (only the crop coordinate source differs between dynamic and CSV
        datasets).

        Args:
            image1, image2: Tensors (C, H, W).  image2 is the geometrically transformed
                version of image1 (or its paired modality).
            forward_matrix: flat Tensor[6 or 8].
            inverse_matrix: flat Tensor[6 or 8].
            original_shape: (C, H, W) tuple of the *original* (pre-crop) image.
            crop_coords: None  → use self.paired_crop (dynamic crop)
                         (offset1, offset2, h, w, contained) → use pre-computed coords.

        Returns:
            image1_crop, image2_crop: cropped Tensors.
            inverse_matrix_crop_norm, forward_matrix_crop_norm: normalized matrices.
            mask1, mask2: boolean validity masks.
        """
        image_depth, image_height, image_width = original_shape

        forward_matrix_norm = normalize_affine_matrix(forward_matrix.squeeze(), image_width, image_height)
        inverse_matrix_norm = normalize_affine_matrix(inverse_matrix.squeeze(), image_width, image_height)

        if crop_coords is None:
            bbox_orig = np.array([
                [0, 0], [image_width, 0], [image_width, image_height], [0, image_height]
            ])
            bbox_trans = apply_normalized_affine_to_polygon(
                bbox_orig, forward_matrix_norm.numpy(), image_height, image_width
            )
            image1_crop, image2_crop, offset1, offset2, h, w, contained = self.paired_crop(
                image1, image2, bbox=bbox_trans
            )
        else:
            offset1, offset2, h, w, contained = crop_coords
            image1_crop = F.crop(image1, offset1, offset2, h, w)
            image2_crop = F.crop(image2, offset1, offset2, h, w)

        _, crop_h, crop_w = image1_crop.shape

        forward_matrix_crop = adjust_affine_transformation(forward_matrix, offset1, offset2)
        inverse_matrix_crop = adjust_affine_transformation(inverse_matrix, offset1, offset2)

        forward_matrix_crop_norm = normalize_affine_matrix(forward_matrix_crop.squeeze(), crop_w, crop_h)
        inverse_matrix_crop_norm = normalize_affine_matrix(inverse_matrix_crop.squeeze(), crop_w, crop_h)

        mask1, mask2 = create_masks(
            offset1, offset2, h, w,
            (image_depth, image_height, image_width),
            inverse_matrix_norm, forward_matrix_norm,
            inverse_matrix_crop_norm, forward_matrix_crop_norm,
            contained,
        )

        return image1_crop, image2_crop, inverse_matrix_crop_norm, forward_matrix_crop_norm, mask1, mask2
