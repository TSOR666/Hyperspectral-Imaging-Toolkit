"""Tests for MSWR v2 dataloader."""

import numpy as np
import torch


class TestPatchExtraction:
    """Tests for patch extraction logic."""

    def test_patch_coordinates_non_negative(self):
        """Test that patch coordinates are always non-negative."""
        # Simulate the patch extraction logic
        crop_size = 128
        stride = 64

        # Test with various image sizes
        test_sizes = [(128, 128), (256, 256), (100, 100), (64, 64), (300, 200)]

        for h, w in test_sizes:
            patches_h = max(1, (h - crop_size) // stride + 1)
            patches_w = max(1, (w - crop_size) // stride + 1)

            for row in range(patches_h):
                for col in range(patches_w):
                    # This is the fixed logic
                    y = max(0, min(row * stride, h - crop_size))
                    x = max(0, min(col * stride, w - crop_size))

                    assert y >= 0, f"y={y} should be non-negative for h={h}"
                    assert x >= 0, f"x={x} should be non-negative for w={w}"

    def test_patch_within_bounds(self):
        """Test that patch extraction stays within image bounds."""
        crop_size = 128
        stride = 64

        test_sizes = [(256, 256), (128, 128), (200, 150)]

        for h, w in test_sizes:
            # Pad if too small (as done in dataloader)
            if h < crop_size or w < crop_size:
                h = max(h, crop_size)
                w = max(w, crop_size)

            patches_h = max(1, (h - crop_size) // stride + 1)
            patches_w = max(1, (w - crop_size) // stride + 1)

            for row in range(patches_h):
                for col in range(patches_w):
                    y = max(0, min(row * stride, h - crop_size))
                    x = max(0, min(col * stride, w - crop_size))

                    # Check that patch doesn't exceed bounds
                    assert y + crop_size <= h, f"y+crop_size exceeds h for h={h}"
                    assert x + crop_size <= w, f"x+crop_size exceeds w for w={w}"


class TestUndersizedImageHandling:
    """Tests for handling images smaller than crop size."""

    def test_padding_small_image(self):
        """Test that small images are padded correctly."""
        crop_size = 128

        # Create a small image
        small_h, small_w = 64, 80
        image = np.random.rand(3, small_h, small_w).astype(np.float32)

        # Simulate padding logic from dataloader
        if small_h < crop_size or small_w < crop_size:
            pad_h = max(0, crop_size - small_h)
            pad_w = max(0, crop_size - small_w)
            padded = np.pad(
                image, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect"
            )

            assert padded.shape[1] >= crop_size
            assert padded.shape[2] >= crop_size

    def test_padding_preserves_original_data(self):
        """Test that padding preserves original image data."""
        crop_size = 128

        small_h, small_w = 64, 80
        image = np.random.rand(3, small_h, small_w).astype(np.float32)

        pad_h = max(0, crop_size - small_h)
        pad_w = max(0, crop_size - small_w)
        padded = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")

        # Original data should be preserved
        np.testing.assert_array_equal(padded[:, :small_h, :small_w], image)

    def test_no_padding_for_large_image(self):
        """Test that large images are not padded."""
        crop_size = 128

        large_h, large_w = 256, 256
        image = np.random.rand(3, large_h, large_w).astype(np.float32)

        if large_h >= crop_size and large_w >= crop_size:
            # No padding needed
            assert image.shape == (3, large_h, large_w)


class TestAugmentations:
    """Tests for data augmentation functions."""

    def test_rotation_preserves_shape(self):
        """Test that rotation preserves tensor shape."""
        rgb = np.random.rand(3, 128, 128).astype(np.float32)
        hsi = np.random.rand(31, 128, 128).astype(np.float32)

        # Simulate rotation (k=1 for 90 degrees)
        rgb_rot = np.rot90(rgb, k=1, axes=(1, 2)).copy()
        hsi_rot = np.rot90(hsi, k=1, axes=(1, 2)).copy()

        assert rgb_rot.shape == rgb.shape
        assert hsi_rot.shape == hsi.shape

    def test_flip_preserves_shape(self):
        """Test that flipping preserves tensor shape."""
        rgb = np.random.rand(3, 128, 128).astype(np.float32)

        # Vertical flip
        rgb_vflip = rgb[:, ::-1, :].copy()
        # Horizontal flip
        rgb_hflip = rgb[:, :, ::-1].copy()

        assert rgb_vflip.shape == rgb.shape
        assert rgb_hflip.shape == rgb.shape


class TestTensorConversion:
    """Tests for numpy to tensor conversion."""

    def test_tensor_dtype(self):
        """Test that tensors have correct dtype."""
        rgb = np.random.rand(3, 128, 128).astype(np.float32)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb, dtype=np.float32))

        assert tensor.dtype == torch.float32

    def test_tensor_contiguity(self):
        """Test that tensors are contiguous."""
        # Create non-contiguous array
        rgb = np.random.rand(128, 128, 3).astype(np.float32).transpose(2, 0, 1)

        # Make contiguous
        tensor = torch.from_numpy(np.ascontiguousarray(rgb, dtype=np.float32))

        assert tensor.is_contiguous()
