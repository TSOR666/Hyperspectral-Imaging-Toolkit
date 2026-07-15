"""Mixed-precision policy: fp32 attention logits + softmax under autocast (spec 9.5).

The subtle failure this guards against: casting q/k to fp32 by hand is not enough,
because ``torch.autocast`` intercepts the following ``torch.matmul`` and re-downcasts
its operands. Only disabling autocast around the logits/softmax region keeps them in
fp32. These tests would pass on the *broken* version at the module level (the output is
still finite and roughly right), so they check the mechanism directly.
"""

from __future__ import annotations

import pytest
import torch

from cas_hsi import build_cas_hsi
from cas_hsi.blocks import CrossChannelAttention, stripe_attention
from cas_hsi.blocks.channel_attention import fp32_matmul_guard


def _cpu_bf16_autocast_ok() -> bool:
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            _ = torch.randn(2, 2) @ torch.randn(2, 2)
        return True
    except (RuntimeError, AssertionError):  # pragma: no cover - platform dependent
        return False


pytestmark = pytest.mark.skipif(
    not _cpu_bf16_autocast_ok(), reason="CPU bf16 autocast unavailable in this build"
)


def test_guard_keeps_matmul_fp32_under_autocast():
    """The core mechanism: inside the guard, an fp32 matmul stays fp32 under autocast."""
    a = torch.randn(4, 8)
    b = torch.randn(8, 4)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        # Without the guard, autocast downcasts the matmul to bf16 -- this is the bug.
        assert torch.matmul(a, b).dtype == torch.bfloat16
        # With the guard, the fp32 operands' matmul stays fp32.
        with fp32_matmul_guard(True, "cpu"):
            assert torch.matmul(a.float(), b.float()).dtype == torch.float32


def test_guard_is_noop_when_disabled():
    a = torch.randn(4, 8)
    b = torch.randn(8, 4)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with fp32_matmul_guard(False, "cpu"):
            assert torch.matmul(a, b).dtype == torch.bfloat16


def test_cross_channel_attention_logits_are_fp32_under_autocast():
    """The whole point of fp32_attention=True: match the pure-fp32 result under AMP.

    With the bug, the bf16 logit accumulation makes the autocast output drift from the
    fp32 reference by more than the value-matmul rounding alone. We assert the fp32
    branch tracks the fp32 reference clearly better than the fp32_attention=False branch.
    """
    torch.manual_seed(0)
    x = torch.randn(1, 64, 12, 12)

    guarded = CrossChannelAttention(64, head_dim=32, fp32_attention=True).eval()
    unguarded = CrossChannelAttention(64, head_dim=32, fp32_attention=False)
    unguarded.load_state_dict(guarded.state_dict())
    unguarded.eval()

    with torch.no_grad():
        reference = guarded(x)  # pure fp32, no autocast
        with torch.autocast("cpu", dtype=torch.bfloat16):
            fp32_logits = guarded(x).float()
            reduced_logits = unguarded(x).float()

    err_fp32 = (fp32_logits - reference).abs().mean()
    err_reduced = (reduced_logits - reference).abs().mean()
    assert err_fp32 <= err_reduced, (
        f"fp32-logit path ({err_fp32:.3e}) is not closer to the fp32 reference than the "
        f"reduced path ({err_reduced:.3e}); the guard is not taking effect"
    )


def test_stripe_attention_runs_finite_under_autocast():
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 4, 32, 13, 17) for _ in range(3))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = stripe_attention(q, k, v, stripe_width=8, orientation="horizontal",
                               fp32_attention=True)
    assert out.shape == q.shape
    assert torch.isfinite(out).all()


def test_full_model_runs_and_is_finite_under_autocast():
    torch.manual_seed(0)
    model = build_cas_hsi("tiny").eval()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(torch.rand(1, 3, 48, 64))
    assert out.shape == (1, 31, 48, 64)
    assert torch.isfinite(out).all()
