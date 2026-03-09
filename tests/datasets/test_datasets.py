"""Tests for the refactored datasets package.

These tests verify:
1. Augmentation pipelines produce correct output shape/range.
2. BaseDataset helper methods compute geometrically correct results.
3. The crop-and-finalize pipeline returns correctly structured 6-tuples.
4. Each dataset's __getitem__ returns the expected 8-tuple with correct types/shapes.

All tests use synthetic in-memory data and do NOT require actual dataset files.
"""

import os
import sys
import random
import tempfile

import numpy as np
import pytest
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_cfg(norm="min-max", homography="affine", mode="mono", **extra):
    cfg = {"norm": norm, "homography": homography, "mode": mode}
    cfg.update(extra)
    return cfg


def random_rgb_pil(h=256, w=256):
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def random_tensor(c=1, h=256, w=256, low=0.0, high=1.0):
    return torch.empty(c, h, w).uniform_(low, high)


# ===========================================================================
# 1. Augmentation pipeline tests
# ===========================================================================

class TestAugmentations:
    def test_sensor_like_augmentation_output_shape(self):
        from datasets.augmentations import sensor_like_augmentation
        image = random_tensor(3, 128, 128)
        out = sensor_like_augmentation(image)
        assert isinstance(out, torch.Tensor)
        assert out.ndim == 3
        assert not torch.isnan(out).any(), "NaN in sensor_like_augmentation output"

    def test_sensor_like_augmentation_value_range(self):
        from datasets.augmentations import sensor_like_augmentation
        # Run several times to hit different random branches
        torch.manual_seed(0)
        for _ in range(5):
            image = random_tensor(3, 64, 64)
            out = sensor_like_augmentation(image)
            assert out.min() >= -1e-3, f"Output below 0: {out.min()}"
            assert out.max() <= 1.0 + 1e-3, f"Output above 1: {out.max()}"

    def test_build_traditional_augmentation_min_max(self):
        from datasets.augmentations import build_traditional_augmentation
        aug = build_traditional_augmentation("min-max")
        image = random_tensor(1, 64, 64)
        out = aug(image)
        assert out.shape == image.shape or out.ndim == 3

    def test_build_traditional_augmentation_mean_std(self):
        from datasets.augmentations import build_traditional_augmentation
        aug = build_traditional_augmentation("mean-std", mean=[0.5], std=[0.5])
        image = random_tensor(1, 64, 64)
        out = aug(image)
        assert out.shape == image.shape

    def test_build_traditional_augmentation_requires_mean_std(self):
        from datasets.augmentations import build_traditional_augmentation
        with pytest.raises(AssertionError):
            build_traditional_augmentation("mean-std")


# ===========================================================================
# 2. percentile_minmax tests
# ===========================================================================

class TestPercentileMinmax:
    def test_output_range(self):
        from datasets.utils import percentile_minmax
        t = torch.randn(3, 64, 64)
        out = percentile_minmax(t)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_shape_preserved(self):
        from datasets.utils import percentile_minmax
        t = torch.randn(2, 32, 32)
        out = percentile_minmax(t)
        assert out.shape == t.shape

    def test_constant_channel_no_crash(self):
        from datasets.utils import percentile_minmax
        t = torch.ones(1, 16, 16)
        out = percentile_minmax(t)  # p_high - p_low == 0, should not crash
        assert not torch.isnan(out).any()


# ===========================================================================
# 3. BaseDataset helper method tests
# ===========================================================================

class TestBaseDatasetHelpers:
    def _make_ds(self, **cfg_extra):
        from datasets.base_dataset import BaseDataset
        cfg = make_cfg(**cfg_extra)
        return BaseDataset(cfg=cfg, train=True)

    # --- _build_affine_matrices ---

    def test_affine_matrices_inverse_consistency(self):
        """forward @ inverse should be ~identity."""
        ds = self._make_ds()
        image = random_tensor(1, 256, 256)
        fwd, inv = ds._build_affine_matrices(image, angle=30.0, translate=(10.0, -5.0),
                                              scale=0.9, shear=5.0)
        assert fwd.shape == (6,)
        assert inv.shape == (6,)

        fwd_3x3 = torch.cat([fwd.reshape(2, 3), torch.tensor([[0., 0., 1.]])])
        inv_3x3 = torch.cat([inv.reshape(2, 3), torch.tensor([[0., 0., 1.]])])
        product = fwd_3x3 @ inv_3x3
        assert torch.allclose(product, torch.eye(3), atol=1e-4), \
            f"forward @ inverse != identity:\n{product}"

    def test_affine_matrices_identity_params(self):
        """Identity affine (angle=0, t=(0,0), scale=1, shear=0) should give identity matrix."""
        ds = self._make_ds()
        image = random_tensor(1, 256, 256)
        fwd, inv = ds._build_affine_matrices(image, angle=0.0, translate=(0.0, 0.0),
                                              scale=1.0, shear=0.0)
        fwd_3x3 = torch.cat([fwd.reshape(2, 3), torch.tensor([[0., 0., 1.]])])
        assert torch.allclose(fwd_3x3, torch.eye(3), atol=1e-4), f"fwd not identity:\n{fwd_3x3}"

    def test_affine_no_nan(self):
        ds = self._make_ds()
        image = random_tensor(1, 128, 128)
        fwd, inv = ds._build_affine_matrices(image, angle=-45.0, translate=(-30.0, 30.0),
                                              scale=1.2, shear=-20.0)
        assert not torch.isnan(fwd).any()
        assert not torch.isnan(inv).any()

    # --- _apply_augmentation ---

    def test_apply_augmentation_no_aug(self):
        """Without any aug flag, image should be returned unchanged."""
        ds = self._make_ds()
        image = random_tensor(3, 64, 64)
        out = ds._apply_augmentation(image)
        assert torch.equal(out, image)

    def test_apply_augmentation_sensor_like(self):
        from datasets.base_dataset import BaseDataset
        cfg = make_cfg(sensor_like_augmentation=True)
        ds = BaseDataset(cfg=cfg, train=True)
        image = random_tensor(3, 64, 64)
        out = ds._apply_augmentation(image)
        assert isinstance(out, torch.Tensor)
        assert not torch.isnan(out).any()

    def test_apply_augmentation_traditional(self):
        from datasets.base_dataset import BaseDataset
        cfg = make_cfg(traditional_augmentation=True)
        ds = BaseDataset(cfg=cfg, train=True)
        image = random_tensor(1, 64, 64)
        out = ds._apply_augmentation(image)
        assert out.shape == image.shape

    def test_apply_augmentation_skipped_when_not_training(self):
        from datasets.base_dataset import BaseDataset
        cfg = make_cfg(sensor_like_augmentation=True)
        ds = BaseDataset(cfg=cfg, train=False)
        image = random_tensor(3, 64, 64)
        out = ds._apply_augmentation(image)
        assert torch.equal(out, image), "Augmentation should be skipped in eval mode"

    # --- _crop_and_finalize ---

    def test_crop_and_finalize_output_structure(self):
        """The 6-tuple should have correct types and shapes."""
        from datasets.base_dataset import BaseDataset
        from datasets.utils import PairedRandomCrop
        cfg = make_cfg()
        ds = BaseDataset(cfg=cfg, train=True)
        ds.paired_crop = PairedRandomCrop(128, contained=False)

        image1 = random_tensor(1, 512, 512)
        image2 = random_tensor(1, 512, 512)
        # Simple translation-only affine
        fwd = torch.tensor([1.0, 0.0, 10.0, 0.0, 1.0, 5.0])
        inv = torch.tensor([1.0, 0.0, -10.0, 0.0, 1.0, -5.0])

        result = ds._crop_and_finalize(image1, image2, fwd, inv, (1, 512, 512))
        assert len(result) == 6, "Expected 6-tuple from _crop_and_finalize"

        img1_c, img2_c, inv_norm, fwd_norm, mask1, mask2 = result
        assert img1_c.shape == (1, 128, 128)
        assert img2_c.shape == (1, 128, 128)
        assert inv_norm.shape == (6,)
        assert fwd_norm.shape == (6,)
        assert mask1.dtype == torch.bool
        assert mask2.dtype == torch.bool

    def test_crop_and_finalize_with_precomputed_coords(self):
        from datasets.base_dataset import BaseDataset
        cfg = make_cfg()
        ds = BaseDataset(cfg=cfg, train=True)

        image1 = random_tensor(1, 256, 256)
        image2 = random_tensor(1, 256, 256)
        fwd = torch.tensor([1.0, 0.0, 5.0, 0.0, 1.0, 3.0])
        inv = torch.tensor([1.0, 0.0, -5.0, 0.0, 1.0, -3.0])

        crop_coords = (10, 20, 128, 128, True)
        result = ds._crop_and_finalize(image1, image2, fwd, inv, (1, 256, 256), crop_coords=crop_coords)
        img1_c, img2_c, inv_norm, fwd_norm, mask1, mask2 = result
        assert img1_c.shape == (1, 128, 128)
        assert img2_c.shape == (1, 128, 128)

    def test_crop_and_finalize_no_nan(self):
        from datasets.base_dataset import BaseDataset
        from datasets.utils import PairedRandomCrop
        cfg = make_cfg()
        ds = BaseDataset(cfg=cfg, train=True)
        ds.paired_crop = PairedRandomCrop(64, contained=False)

        image1 = random_tensor(1, 256, 256)
        image2 = random_tensor(1, 256, 256)
        fwd = torch.tensor([0.866, -0.5, 20.0, 0.5, 0.866, -10.0])
        inv = torch.tensor([0.866, 0.5, -20.0, -0.5, 0.866, 10.0])

        result = ds._crop_and_finalize(image1, image2, fwd, inv, (1, 256, 256))
        for item in result[:4]:
            assert not torch.isnan(item).any(), "NaN found in crop_and_finalize output"


# ===========================================================================
# 4. ImageNetDataset __getitem__ integration test (synthetic data)
# ===========================================================================

class TestImageNetDataset:
    @pytest.fixture
    def dataset(self, tmp_path):
        from datasets.imagenet_dataset import ImageNetDataset
        # Create a temporary list file with fake PNG paths
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        paths = []
        for i in range(3):
            p = img_dir / f"img_{i}.png"
            random_rgb_pil(300, 300).save(str(p))
            paths.append(str(p))

        list_file = tmp_path / "paths.txt"
        list_file.write_text("\n".join(paths))

        cfg = make_cfg(sensor_like_augmentation=True)
        return ImageNetDataset(root_path=str(list_file), img_size=128, train=True, cfg=cfg)

    def test_getitem_returns_8_tuple(self, dataset):
        item = dataset[0]
        assert len(item) == 8, f"Expected 8-tuple, got {len(item)}"

    def test_getitem_shapes(self, dataset):
        img1, img2, lbl1, lbl2, inv, fwd, mask1, mask2 = dataset[0]
        assert img1.ndim == 3
        assert img2.ndim == 3
        assert img1.shape[-2:] == img2.shape[-2:]
        assert lbl1.shape == ()
        assert lbl2.shape == ()
        assert inv.shape in [(6,), (8,)]
        assert fwd.shape in [(6,), (8,)]
        assert mask1.dtype == torch.bool
        assert mask2.dtype == torch.bool

    def test_getitem_no_nan(self, dataset):
        img1, img2, lbl1, lbl2, inv, fwd, mask1, mask2 = dataset[0]
        assert not torch.isnan(img1).any()
        assert not torch.isnan(img2).any()
        assert not torch.isnan(inv).any()
        assert not torch.isnan(fwd).any()

    def test_len(self, dataset):
        assert len(dataset) == 3


# ===========================================================================
# 5. Mono/multi mode consistency
# ===========================================================================

class TestModeHandling:
    def test_imagenet_warns_on_multi(self, tmp_path, capsys):
        from datasets.imagenet_dataset import ImageNetDataset
        img = random_rgb_pil(200, 200)
        p = tmp_path / "img.png"
        img.save(str(p))
        list_file = tmp_path / "paths.txt"
        list_file.write_text(str(p))

        cfg = make_cfg(mode="multi")
        ImageNetDataset(root_path=str(list_file), img_size=128, train=True, cfg=cfg)
        captured = capsys.readouterr()
        assert "mono" in captured.out.lower() or "warning" in captured.out.lower()

    def test_base_dataset_mode_defaults_to_mono(self):
        from datasets.base_dataset import BaseDataset
        ds = BaseDataset(cfg={"norm": "min-max"}, train=True)
        assert ds.mode == "mono"

    def test_base_dataset_reads_mode_from_cfg(self):
        from datasets.base_dataset import BaseDataset
        ds = BaseDataset(cfg={"norm": "min-max", "mode": "multi"}, train=True)
        assert ds.mode == "multi"


# ===========================================================================
# 6. Normalization pipeline tests
# ===========================================================================

class TestNormalizationPipelines:
    def test_min_max_transforms_no_aug(self):
        from datasets.base_dataset import BaseDataset
        ds = BaseDataset(cfg={"norm": "min-max"}, train=True)
        pil = random_rgb_pil(64, 64)
        t = ds.transforms(pil)
        assert t.min() >= 0.0
        assert t.max() <= 1.0 + 1e-5

    def test_transforms_percentile_range(self):
        from datasets.base_dataset import BaseDataset
        ds = BaseDataset(cfg={"norm": "min-max"}, train=True)
        arr = np.random.randn(64, 64, 1).astype(np.float32)
        t = ds.transforms_percentile(arr)
        assert t.min() >= 0.0
        assert t.max() <= 1.0 + 1e-5

    def test_mean_std_normalization(self):
        from datasets.base_dataset import BaseDataset
        cfg = {"norm": "mean-std", "mean": [0.5], "std": [0.25]}
        ds = BaseDataset(cfg=cfg, train=True)
        pil = Image.fromarray(np.full((64, 64), 128, dtype=np.uint8))
        t = ds.transforms(pil)
        assert t.shape[0] == 1

    def test_no_norm_warning(self, capsys):
        from datasets.base_dataset import BaseDataset
        BaseDataset(cfg={}, train=True)
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
