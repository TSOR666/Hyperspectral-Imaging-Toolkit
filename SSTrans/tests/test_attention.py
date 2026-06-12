from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from hsiformer.attention import _scaled_cosine_attention


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
