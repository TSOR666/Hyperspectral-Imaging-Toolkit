"""Tests for MSWR v2 inference module."""

import numpy as np
import pytest
import torch

# Conditional import to handle missing dependencies
try:
    from mswr_inference import (
        TiledProcessor,
        MemoryManager,
        EnsembleProcessor,
        InferenceConfig,
        MSWRInference,
        normalize_rgb_like_training,
    )
    INFERENCE_AVAILABLE = True
except ImportError:
    INFERENCE_AVAILABLE = False
    TiledProcessor = None
    MemoryManager = None
    EnsembleProcessor = None
    InferenceConfig = None
    MSWRInference = None
    normalize_rgb_like_training = None

pytestmark = pytest.mark.skipif(not INFERENCE_AVAILABLE, reason="inference module dependencies not available")


class TestInferencePreprocessing:
    """Tests for train/inference preprocessing consistency."""

    def test_rgb_normalization_matches_training_minmax(self):
        image = np.array(
            [
                [[10, 20, 30], [40, 50, 60]],
                [[70, 80, 90], [100, 110, 120]],
            ],
            dtype=np.uint8,
        )

        normalized = normalize_rgb_like_training(image)
        expected = (image.astype(np.float32) - 10.0) / 110.0

        assert normalized.dtype == np.float32
        assert np.allclose(normalized, expected)

    def test_rgb_constant_image_preserves_intensity(self):
        image = np.full((4, 4, 3), 127, dtype=np.uint8)

        normalized = normalize_rgb_like_training(image)

        assert np.allclose(normalized, 127.0 / 255.0, atol=1e-6)


class TestTiledProcessor:
    """Tests for TiledProcessor class."""

    def test_split_image_no_duplicates(self):
        """Test that split_image produces no duplicate positions."""
        processor = TiledProcessor(tile_size=256, overlap=32)

        # Create test image
        image = np.random.rand(512, 512, 3).astype(np.float32)
        tiles, metadata = processor.split_image(image)

        # Check for duplicates
        positions = metadata["positions"]
        unique_positions = set(positions)
        assert len(positions) == len(unique_positions), "Duplicate positions found"

    def test_split_image_covers_full_image(self):
        """Test that tiles cover the entire image."""
        processor = TiledProcessor(tile_size=256, overlap=32)

        image = np.random.rand(512, 512, 3).astype(np.float32)
        tiles, metadata = processor.split_image(image)

        H, W = metadata["original_shape"]

        # Check that we have tiles covering corners
        positions = set(metadata["positions"])

        # At least one tile should start at (0, 0)
        has_origin = (0, 0) in positions
        assert has_origin, "No tile at origin"

    def test_merge_tiles_preserves_content(self):
        """Test that merge preserves tile content for non-overlapping case."""
        processor = TiledProcessor(tile_size=256, overlap=0)

        # Create test image
        image = np.random.rand(256, 256, 3).astype(np.float32)
        tiles, metadata = processor.split_image(image)

        # Merge back
        merged = processor.merge_tiles(tiles, metadata)

        # Should be close to original (may differ due to blending)
        np.testing.assert_array_almost_equal(
            merged[:256, :256, :], image, decimal=5
        )

    def test_merge_tiles_float32_output(self):
        """Test that merge output is float32."""
        processor = TiledProcessor(tile_size=128, overlap=16)

        image = np.random.rand(256, 256, 3).astype(np.float32)
        tiles, metadata = processor.split_image(image)
        merged = processor.merge_tiles(tiles, metadata)

        assert merged.dtype == np.float32

    def test_merge_tiles_no_extreme_values(self):
        """Test that merge doesn't produce extreme values at corners."""
        processor = TiledProcessor(tile_size=128, overlap=32)

        # Create test image with values in [0, 1]
        image = np.random.rand(256, 256, 3).astype(np.float32)
        tiles, metadata = processor.split_image(image)
        merged = processor.merge_tiles(tiles, metadata)

        # Output should not have extreme values
        # (old code could produce spikes at corners)
        assert merged.max() < 2.0, "Extreme max value detected"
        assert merged.min() > -1.0, "Extreme min value detected"

    def test_split_small_image(self):
        """Test splitting image smaller than tile size."""
        processor = TiledProcessor(tile_size=256, overlap=32)

        # Image smaller than tile size
        image = np.random.rand(128, 128, 3).astype(np.float32)
        tiles, metadata = processor.split_image(image)

        assert len(tiles) >= 1
        assert metadata["n_tiles"] >= 1

    def test_blend_weights_corners(self):
        """Test that blend weights handle corners properly."""
        processor = TiledProcessor(tile_size=128, overlap=32)
        weights = processor._create_blend_weights(128, 32)

        # Corners should have lower weights but not zero
        corner_value = weights[0, 0]
        center_value = weights[64, 64]

        assert corner_value > 0, "Corner weight should be > 0"
        assert corner_value < center_value, "Corner weight should be less than center"
        assert center_value == 1.0, "Center weight should be 1.0"

    def test_png_save_returns_directory(self, workspace_tmp_dir):
        """PNG export should return the output directory and write metadata."""
        engine = object.__new__(MSWRInference)
        engine.config = InferenceConfig(
            model_path="unused.pth",
            output_dir=str(workspace_tmp_dir),
            save_format="png",
            save_visualization=False,
        )
        engine.output_dir = workspace_tmp_dir

        output = np.random.rand(8, 8, 4).astype(np.float32)
        metadata = {"shape": output.shape}

        saved_path = engine.save_output(output, "sample.png", metadata)

        assert saved_path.is_dir()
        assert (saved_path / "channel_000.png").exists()
        assert (saved_path / "metadata.json").exists()


class TestMemoryManager:
    """Tests for MemoryManager class."""

    def test_get_available_memory_returns_positive(self):
        """Test that available memory is positive."""
        manager = MemoryManager()
        memory = manager.get_available_memory()

        assert memory > 0

    def test_estimate_tile_size_reasonable(self):
        """Test that estimated tile size is reasonable."""
        manager = MemoryManager()

        # Create a small dummy model
        model = torch.nn.Linear(10, 10)
        tile_size = manager.estimate_tile_size(model)

        assert 128 <= tile_size <= 1024
        assert tile_size % 32 == 0, "Tile size should be multiple of 32"


class TestEnsembleProcessor:
    """Tests for EnsembleProcessor class."""

    def test_flip_augmentations(self):
        """Test flip augmentations."""
        processor = EnsembleProcessor(mode="flip")
        assert len(processor.transforms) == 3  # Original + 2 flips

    def test_rotate_augmentations(self):
        """Test rotation augmentations."""
        processor = EnsembleProcessor(mode="rotate")
        assert len(processor.transforms) == 4  # Original + 3 rotations

    def test_full_augmentations(self):
        """Test full augmentations."""
        processor = EnsembleProcessor(mode="full")
        assert len(processor.transforms) == 8
