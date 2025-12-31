import pytest
import torch

from hsi_model.models.attention import CSWinAttentionBlock, EfficientSpectralAttention


def test_cswin_attention_padding_shape():
    block = CSWinAttentionBlock(dim=8, num_heads=2, split_size=3)
    x = torch.randn(1, 8, 5, 7)
    out = block(x)
    assert out.shape == x.shape


def test_efficient_spectral_attention_preserves_dtype():
    attn = EfficientSpectralAttention(8, num_heads=2, config={"norm_groups": 2})
    x = torch.randn(1, 8, 4, 4, dtype=torch.float16)
    try:
        out = attn(x)
    except RuntimeError as exc:
        message = str(exc).lower()
        if "float16" in message or "half" in message:
            pytest.skip("float16 not supported on this device")
        raise
    assert out.dtype == x.dtype


# GATE 4.2: Edge Case Tests
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("height,width", [(7, 7), (8, 8), (17, 19)])
def test_cswin_attention_edge_cases(batch_size, height, width):
    """Test CSWin attention with various edge case dimensions (Finding 4.2)."""
    block = CSWinAttentionBlock(dim=8, num_heads=2, split_size=7)
    x = torch.randn(batch_size, 8, height, width)
    out = block(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all(), "Output contains NaN/Inf"


def test_cswin_attention_rejects_nan():
    """Test that NaN inputs are properly handled (Finding 4.2)."""
    block = CSWinAttentionBlock(dim=8, num_heads=2, split_size=7)
    x = torch.full((1, 8, 14, 14), float('nan'))

    # The model may either raise an error or propagate NaN
    # We test that it doesn't crash silently
    try:
        out = block(x)
        # If no error, check that NaN is at least detected in output
        assert torch.isnan(out).any(), "NaN input should produce NaN output or error"
    except (RuntimeError, ValueError):
        # This is acceptable - model detected invalid input
        pass


def test_cswin_attention_minimum_size():
    """Test attention with minimum viable input size (Finding 4.2)."""
    split_size = 7
    block = CSWinAttentionBlock(dim=8, num_heads=2, split_size=split_size)
    # Minimum size that's divisible by split_size
    x = torch.randn(1, 8, split_size, split_size)
    out = block(x)
    assert out.shape == x.shape


def test_cswin_attention_rejects_wrong_channels():
    """Test that channel mismatch is caught (Finding 5.1)."""
    block = CSWinAttentionBlock(dim=8, num_heads=2, split_size=7)
    x = torch.randn(1, 16, 14, 14)  # Wrong channel count (16 instead of 8)

    with pytest.raises(ValueError, match="Channel mismatch"):
        out = block(x)


def test_cswin_attention_rejects_wrong_dims():
    """Test that incorrect tensor dimensions are caught (Finding 5.1)."""
    block = CSWinAttentionBlock(dim=8, num_heads=2, split_size=7)
    x = torch.randn(8, 14, 14)  # 3D instead of 4D

    with pytest.raises(ValueError, match="Expected 4D input"):
        out = block(x)


# GATE 4.5: FP16 Stability Test
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA for FP16")
def test_efficient_spectral_attention_fp16():
    """Test that FP16 is stable and doesn't produce NaN/Inf (Finding 4.5)."""
    attn = EfficientSpectralAttention(8, num_heads=2, config={"norm_groups": 2}).cuda()
    x = torch.randn(1, 8, 4, 4, dtype=torch.float16, device='cuda')
    out = attn(x)
    assert out.dtype == torch.float16, "Output dtype mismatch"
    assert torch.isfinite(out).all(), "FP16 produced NaN/Inf"
