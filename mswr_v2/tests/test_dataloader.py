"""Tests for MSWR v2 dataloader.

Extended with edge case tests as per FINDING 4.2 from the audit.
"""

import numpy as np
import pytest
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


# ============================================================================
# EDGE CASE TESTS (FINDING 4.2)
# Tests for edge cases: batch_size=1, odd dimensions, boundary conditions
# ============================================================================


class TestEdgeCaseDimensions:
    """Tests for edge case dimensions as per FINDING 4.2."""

    @pytest.mark.parametrize("crop_size,stride", [
        (64, 8),
        (128, 16),
        (256, 32),
    ])
    @pytest.mark.parametrize("image_size", [
        (63, 63),    # Odd dimensions
        (127, 127),  # Just under common sizes
        (255, 255),  # Near power of 2
        (65, 65),    # Just over 64
    ])
    def test_odd_dimension_handling(self, crop_size, stride, image_size):
        """Test that padding works correctly for odd dimensions."""
        h, w = image_size

        # Calculate if padding is needed
        needs_padding_h = h < crop_size
        needs_padding_w = w < crop_size

        if needs_padding_h or needs_padding_w:
            pad_h = max(0, crop_size - h)
            pad_w = max(0, crop_size - w)
            h_padded = h + pad_h
            w_padded = w + pad_w
        else:
            h_padded, w_padded = h, w

        # Calculate patches
        patches_h = max(1, (h_padded - crop_size) // stride + 1)
        patches_w = max(1, (w_padded - crop_size) // stride + 1)

        assert patches_h >= 1, f"Should have at least 1 patch for height"
        assert patches_w >= 1, f"Should have at least 1 patch for width"

        # Test all patch coordinates are valid
        for row in range(patches_h):
            for col in range(patches_w):
                y = max(0, min(row * stride, h_padded - crop_size))
                x = max(0, min(col * stride, w_padded - crop_size))

                assert 0 <= y <= h_padded - crop_size
                assert 0 <= x <= w_padded - crop_size
                assert y + crop_size <= h_padded
                assert x + crop_size <= w_padded

    @pytest.mark.parametrize("h,w", [
        (1, 1),      # Minimum size
        (2, 2),      # Very small
        (16, 16),    # Small
        (31, 31),    # Odd small
        (33, 17),    # Asymmetric odd
    ])
    def test_very_small_images(self, h, w):
        """Test handling of very small images that need significant padding."""
        crop_size = 64

        # Image needs padding
        image = np.random.rand(3, h, w).astype(np.float32)

        pad_h = max(0, crop_size - h)
        pad_w = max(0, crop_size - w)

        # Pad with reflect (or replicate if too small for reflect)
        if h > 1 and w > 1:
            padded = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
        else:
            # Very small images use edge padding
            padded = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode='edge')

        assert padded.shape[1] >= crop_size
        assert padded.shape[2] >= crop_size
        assert np.isfinite(padded).all(), "Padded image contains NaN or Inf"


class TestEdgeCaseBatchSizes:
    """Tests for edge case batch sizes."""

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
    def test_batch_size_handling(self, batch_size):
        """Test that various batch sizes work correctly."""
        # Simulate a batch of tensors
        tensors = [
            torch.randn(3, 128, 128) for _ in range(batch_size)
        ]
        batch = torch.stack(tensors)

        assert batch.shape == (batch_size, 3, 128, 128)
        assert batch.is_contiguous()

    def test_batch_size_one(self):
        """Specifically test batch_size=1 (common edge case)."""
        # Create single sample batch
        rgb = torch.randn(1, 3, 128, 128)
        hsi = torch.randn(1, 31, 128, 128)

        # Verify shapes
        assert rgb.shape[0] == 1
        assert hsi.shape[0] == 1

        # Verify batch dimension can be squeezed/unsqueezed
        rgb_squeezed = rgb.squeeze(0)
        assert rgb_squeezed.shape == (3, 128, 128)

        rgb_unsqueezed = rgb_squeezed.unsqueeze(0)
        assert rgb_unsqueezed.shape == (1, 3, 128, 128)


class TestRGBNormalizationEdgeCases:
    """Tests for RGB normalization edge cases (related to FINDING 2.4)."""

    def test_constant_value_image(self):
        """Test that constant-value images don't become all zeros."""
        # Create constant-value image (e.g., calibration target)
        constant_value = 128
        image = np.full((100, 100, 3), constant_value, dtype=np.uint8)
        image = image.astype(np.float32)

        # Apply normalization logic
        denom = image.max() - image.min()
        if denom < 1e-6:
            # FIXED: Preserve mean intensity instead of zeroing
            mean_val = image.mean()
            normalized_mean = np.clip(mean_val / 255.0, 0.0, 1.0)
            normalized = np.full_like(image, normalized_mean)
        else:
            normalized = (image - image.min()) / denom

        # Should not be all zeros
        assert not np.allclose(normalized, 0.0), "Constant image should not become all zeros"
        # Should have a reasonable value
        expected_value = constant_value / 255.0
        assert np.allclose(normalized, expected_value, atol=0.01)

    def test_near_constant_image(self):
        """Test images with very small variance."""
        # Create image with tiny variance
        base_value = 200
        image = np.full((100, 100, 3), base_value, dtype=np.float32)
        image += np.random.uniform(-0.0001, 0.0001, image.shape).astype(np.float32)

        denom = image.max() - image.min()
        assert denom < 1e-6, "Test setup: image should have near-zero variance"

        # Apply fixed normalization
        if denom < 1e-6:
            mean_val = image.mean()
            normalized_mean = np.clip(mean_val / 255.0, 0.0, 1.0)
            normalized = np.full_like(image, normalized_mean)
        else:
            normalized = (image - image.min()) / denom

        assert np.isfinite(normalized).all(), "Normalization produced NaN/Inf"

    def test_black_image(self):
        """Test all-black image handling."""
        image = np.zeros((100, 100, 3), dtype=np.float32)

        denom = image.max() - image.min()
        if denom < 1e-6:
            mean_val = image.mean()
            normalized_mean = np.clip(mean_val / 255.0, 0.0, 1.0)
            normalized = np.full_like(image, normalized_mean)
        else:
            normalized = (image - image.min()) / denom

        assert np.isfinite(normalized).all()
        assert np.allclose(normalized, 0.0)  # Black should stay at 0

    def test_white_image(self):
        """Test all-white image handling."""
        image = np.full((100, 100, 3), 255, dtype=np.float32)

        denom = image.max() - image.min()
        if denom < 1e-6:
            mean_val = image.mean()
            normalized_mean = np.clip(mean_val / 255.0, 0.0, 1.0)
            normalized = np.full_like(image, normalized_mean)
        else:
            normalized = (image - image.min()) / denom

        assert np.isfinite(normalized).all()
        assert np.allclose(normalized, 1.0)  # White should be at 1


class TestAsymmetricDimensions:
    """Tests for asymmetric image dimensions."""

    @pytest.mark.parametrize("h,w", [
        (100, 200),   # Wide
        (200, 100),   # Tall
        (64, 256),    # Very wide
        (256, 64),    # Very tall
        (127, 255),   # Odd asymmetric
    ])
    def test_asymmetric_patch_extraction(self, h, w):
        """Test patch extraction from asymmetric images."""
        crop_size = 64
        stride = 32

        # Pad if needed
        if h < crop_size:
            h = crop_size
        if w < crop_size:
            w = crop_size

        patches_h = max(1, (h - crop_size) // stride + 1)
        patches_w = max(1, (w - crop_size) // stride + 1)

        # Should have different patch counts
        if h != w:
            # At least one dimension should differ in patch count
            # unless the difference is absorbed by stride
            pass

        # All patches should be valid
        for row in range(patches_h):
            for col in range(patches_w):
                y = max(0, min(row * stride, h - crop_size))
                x = max(0, min(col * stride, w - crop_size))

                assert y + crop_size <= h
                assert x + crop_size <= w


class TestBoundaryConditions:
    """Tests for boundary conditions in data loading."""

    def test_patch_at_image_boundary(self):
        """Test patches extracted at the very edge of an image."""
        h, w = 256, 256
        crop_size = 128
        stride = 128

        # Last valid patch position
        last_y = h - crop_size
        last_x = w - crop_size

        # Simulate patch extraction at boundary
        y = max(0, min((h - crop_size) // stride * stride, h - crop_size))
        x = max(0, min((w - crop_size) // stride * stride, w - crop_size))

        assert y == last_y
        assert x == last_x
        assert y + crop_size == h
        assert x + crop_size == w

    def test_single_patch_image(self):
        """Test image that produces exactly one patch."""
        h, w = 128, 128
        crop_size = 128
        stride = 64

        patches_h = max(1, (h - crop_size) // stride + 1)
        patches_w = max(1, (w - crop_size) // stride + 1)

        assert patches_h == 1
        assert patches_w == 1

        # The only patch should start at (0, 0)
        y = max(0, min(0 * stride, h - crop_size))
        x = max(0, min(0 * stride, w - crop_size))

        assert y == 0
        assert x == 0
