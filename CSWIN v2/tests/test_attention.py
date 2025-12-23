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
