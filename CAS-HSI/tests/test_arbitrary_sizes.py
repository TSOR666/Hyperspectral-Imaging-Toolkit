"""Arbitrary image sizes, padding, and tiled inference (specification section 8)."""

from __future__ import annotations

import pytest
import torch

from cas_hsi import tiled_inference
from cas_hsi.layers import crop_to_original, pad_to_multiple, required_padding

# Spec 8.6's own matrix. The last two are large on a CPU-only box, so they are marked
# slow rather than dropped -- 482x512 is the actual ARAD-1K scene size and 513x769 is
# the odd-and-larger case, both of which are precisely what a resolution-agnostic claim
# has to survive.
SIZES = [(1, 1), (7, 9), (31, 47), (63, 65), (127, 193)]
LARGE_SIZES = [(482, 512), (513, 769)]


@pytest.mark.parametrize("height,width", SIZES)
def test_arbitrary_sizes(small_model, height, width):
    with torch.no_grad():
        out = small_model(torch.randn(1, 3, height, width))
    assert out.shape == (1, 31, height, width)


@pytest.mark.slow
@pytest.mark.parametrize("height,width", LARGE_SIZES)
def test_arbitrary_sizes_large(small_model, height, width):
    with torch.no_grad():
        out = small_model(torch.randn(1, 3, height, width))
    assert out.shape == (1, 31, height, width)


def test_same_instance_handles_changing_sizes(small_model):
    """Spec 8.8: no module may cache a spatial size between calls."""
    torch.manual_seed(0)
    square = torch.randn(1, 3, 64, 64)
    odd = torch.randn(1, 3, 33, 97)

    with torch.no_grad():
        first = small_model(square)
        other = small_model(odd)
        again = small_model(square)   # the SAME tensor, after an intervening odd size

    assert first.shape == (1, 31, 64, 64)
    assert other.shape == (1, 31, 33, 97)
    # Any cached spatial state would make this differ.
    assert torch.equal(first, again)


def test_batch_dimension_is_free(small_model):
    with torch.no_grad():
        out = small_model(torch.randn(3, 3, 20, 28))
    assert out.shape == (3, 31, 20, 28)


# ------------------------------------------------------------------ padding ----


@pytest.mark.parametrize(
    "size,multiple,expected",
    [(1, 4, 3), (4, 4, 0), (5, 4, 3), (7, 4, 1), (482, 4, 2), (512, 4, 0)],
)
def test_required_padding(size, multiple, expected):
    assert required_padding(size, multiple) == expected


@pytest.mark.parametrize("height,width", [(1, 1), (2, 3), (7, 9), (31, 47), (128, 128)])
def test_pad_crop_roundtrip_is_identity(height, width):
    x = torch.randn(2, 5, height, width)
    padded, info = pad_to_multiple(x, multiple=4)

    assert padded.shape[-2] % 4 == 0 and padded.shape[-1] % 4 == 0
    assert padded.shape[-2] >= height and padded.shape[-1] >= width

    restored = crop_to_original(padded, info)
    assert restored.shape == x.shape
    assert torch.equal(restored, x), "crop did not exactly undo the pad"


def test_pad_falls_back_to_replicate_on_tiny_inputs():
    """reflect needs pad < extent; a 1x1 input cannot reflect and must not crash."""
    x = torch.randn(1, 3, 1, 1)
    padded, info = pad_to_multiple(x, multiple=4, mode="reflect")
    assert padded.shape[-2:] == (4, 4)
    assert torch.isfinite(padded).all()
    assert torch.equal(crop_to_original(padded, info), x)


def test_crop_is_a_slice_not_an_interpolation():
    """Spec 8.4: 'Do not use interpolation to recover the original size'."""
    x = torch.arange(2 * 1 * 3 * 5, dtype=torch.float32).reshape(2, 1, 3, 5)
    padded, info = pad_to_multiple(x, multiple=4)
    restored = crop_to_original(padded, info)
    # Exact bit equality is only possible if the values were sliced, never resampled.
    assert torch.equal(restored, x)


# ------------------------------------------------------------ tiled inference --


def test_tiled_matches_direct_away_from_seams(small_model):
    """Spec 8.7's validation requirement."""
    torch.manual_seed(0)
    image = torch.randn(1, 3, 160, 192)

    with torch.no_grad():
        direct = small_model(image)
        tiled = tiled_inference(small_model, image, tile_size=96, overlap=32)

    assert tiled.shape == direct.shape
    assert torch.isfinite(tiled).all()

    interior = (slice(None), slice(None), slice(40, -40), slice(40, -40))
    error = (tiled[interior] - direct[interior]).abs().max()
    assert error < 1e-3, f"tiled and direct disagree in the interior by {error:.2e}"


def test_tiled_falls_back_to_direct_when_tile_exceeds_image(small_model):
    image = torch.randn(1, 3, 40, 40)
    with torch.no_grad():
        direct = small_model(image)
        tiled = tiled_inference(small_model, image, tile_size=256, overlap=32)
    assert torch.allclose(direct, tiled, atol=1e-6)


def test_tiled_rejects_overlap_at_least_tile_size(small_model):
    with pytest.raises(ValueError, match="must exceed overlap"):
        tiled_inference(small_model, torch.randn(1, 3, 64, 64), tile_size=32, overlap=32)


def test_tiled_covers_every_pixel(small_model):
    """A tile grid that does not reach the right/bottom edge would silently leave
    those pixels at zero. Non-divisible sizes are exactly where that happens."""
    image = torch.randn(1, 3, 100, 130)  # neither dim is a multiple of the stride
    with torch.no_grad():
        tiled = tiled_inference(small_model, image, tile_size=64, overlap=16)
    assert tiled.shape == (1, 31, 100, 130)
    assert torch.isfinite(tiled).all()
    # Every pixel must have been written by at least one tile: an untouched pixel would
    # be exactly 0.0 across all 31 bands.
    untouched = (tiled.abs().sum(dim=1) == 0).sum()
    assert untouched == 0, f"{int(untouched)} pixels were never covered by a tile"


def test_edge_backend_handles_arbitrary_sizes(small_edge_model):
    for height, width in [(1, 1), (31, 47), (127, 193)]:
        with torch.no_grad():
            out = small_edge_model(torch.randn(1, 3, height, width))
        assert out.shape == (1, 31, height, width)
