from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from hsiformer.attention import CSWinCrossAttention, _scaled_cosine_attention


def _reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    logits = (
        F.normalize(query, dim=-1)
        @ F.normalize(key, dim=-1).transpose(-2, -1)
    ) * scale
    if bias is not None:
        logits = logits + bias
    return logits.softmax(dim=-1) @ value


def test_scaled_cosine_attention_matches_reference_on_cpu() -> None:
    torch.manual_seed(0)
    query = torch.randn(2, 3, 7, 8)
    key = torch.randn(2, 3, 5, 8)
    value = torch.randn(2, 3, 5, 6)
    scale = torch.rand(3, 1, 1)
    bias = torch.randn(1, 3, 7, 5)

    expected = _reference_attention(query, key, value, scale, bias)
    actual = _scaled_cosine_attention(
        query,
        key,
        value,
        scale,
        attention_bias=bias,
    )
    torch.testing.assert_close(actual, expected)


def test_scaled_cosine_attention_clamps_runaway_scale() -> None:
    """An exploded temperature must not produce NaNs; it saturates at the ceiling."""
    torch.manual_seed(0)
    query = torch.randn(2, 3, 7, 8)
    key = torch.randn(2, 3, 5, 8)
    value = torch.randn(2, 3, 5, 6)
    runaway = torch.full((3, 1, 1), 1e6)
    ceiling = torch.full((3, 1, 1), 100.0)

    clamped = _scaled_cosine_attention(query, key, value, runaway)
    saturated = _scaled_cosine_attention(query, key, value, ceiling)

    assert torch.isfinite(clamped).all()
    torch.testing.assert_close(clamped, saturated)


def test_scaled_cosine_attention_is_bit_exact_below_ceiling() -> None:
    """Normal temperatures (<100) pass through unclamped, preserving checkpoints."""
    torch.manual_seed(0)
    query = torch.randn(2, 3, 7, 8)
    key = torch.randn(2, 3, 5, 8)
    value = torch.randn(2, 3, 5, 6)
    scale = torch.full((3, 1, 1), 50.0)

    expected = _reference_attention(query, key, value, scale)
    actual = _scaled_cosine_attention(query, key, value, scale)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("stripe_index", [0, 1])
def test_cross_attention_supports_rectangular_features(
    stripe_index: int,
) -> None:
    torch.manual_seed(0)
    attention = CSWinCrossAttention(
        dim=8,
        resolution=(6, 8),
        idx=stripe_index,
        split_size=2,
        num_heads=2,
    )
    first = torch.randn(2, 48, 8, requires_grad=True)
    second = torch.randn(2, 48, 8, requires_grad=True)

    output = attention(first, second)
    output.square().mean().backward()

    assert output.shape == first.shape
    assert torch.isfinite(output).all()
    assert first.grad is not None and torch.isfinite(first.grad).all()
    assert second.grad is not None and torch.isfinite(second.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_sdpa_matches_reference() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    query = torch.randn(4, 2, 64, 16, device=device)
    key = torch.randn(4, 2, 64, 16, device=device)
    value = torch.randn(4, 2, 64, 16, device=device)
    scale = torch.rand(2, 1, 1, device=device)

    expected = _reference_attention(query, key, value, scale)
    actual = _scaled_cosine_attention(query, key, value, scale)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
