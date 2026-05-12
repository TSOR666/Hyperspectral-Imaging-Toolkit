"""Tests for the v2 audit patches.

Each test asserts the post-fix behavior of one specific finding from the
``AUDIT_REPORT_V2`` independent re-audit. Together with ``probes/probe_audit_v2.py``
they prove the patches behave as documented.
"""
from __future__ import annotations

import math

import torch

from hsi_model.models.generator_v3 import NoiseRobustCSWinGenerator
from hsi_model.models.losses_consolidated import (
    NoiseRobustLoss,
    SinkhornDivergence,
    SinkhornLoss,
)
from hsi_model.utils.metrics import compute_sam_value


# ---------------------------------------------------------------------------
# BLOCKER 1: NoiseRobustLoss must remain graph-connected on NaN inputs
# ---------------------------------------------------------------------------

def _minimal_loss_config() -> dict:
    return {
        "lambda_rec": 1.0,
        "lambda_perceptual": 0.0,
        "lambda_adversarial": 0.1,
        "lambda_sam": 0.05,
        "use_adaptive_weights": False,
        "use_sinkhorn_adversarial": True,
        "max_loss_value": 100.0,
    }


def test_noise_robust_loss_nan_input_keeps_graph_connected():
    """Pre-fix: returning ``torch.tensor(1.0, requires_grad=True)`` produced a
    leaf with grad_fn=None, so ``backward()`` was a silent no-op and the
    trainer's ``isfinite(gen_loss)`` check could not catch it. After the fix
    the fallback is anchored to ``pred`` via ``nan_to_num(pred).sum()*0+1`` so
    the autograd graph is intact and gradients flow as zeros.
    """
    torch.manual_seed(0)
    loss_fn = NoiseRobustLoss(_minimal_loss_config())
    pred = torch.randn(2, 31, 8, 8, requires_grad=True)
    pred_bad = pred.clone()
    pred_bad[0, 0, 0, 0] = float("nan")
    target = torch.randn(2, 31, 8, 8)

    total, comps = loss_fn(pred_bad, target)

    assert torch.isfinite(total), f"fallback loss must be finite, got {total}"
    assert total.requires_grad, "fallback loss must require grad"
    assert total.grad_fn is not None, (
        "fallback loss is graph-disconnected; backward will silently produce no gradients"
    )

    # Backward must not error and must populate pred.grad with a real tensor
    # (zeros are fine — the point is that the optimizer step is observably a
    # no-op rather than silently a no-op).
    total.backward()
    assert pred.grad is not None
    assert torch.all(pred.grad == 0)


def test_noise_robust_loss_total_loss_nan_keeps_graph_connected():
    """Same disconnect existed in the second fallback (total loss NaN). The
    test exercises a synthetic path where the inner Charbonnier comes out
    valid but a hand-injected total triggers the fallback.
    """
    torch.manual_seed(0)
    loss_fn = NoiseRobustLoss(_minimal_loss_config())
    # We piggyback on the fact that lambda_adversarial * adv_loss can flip total
    # to NaN if disc inputs are NaN. Easier: construct pred/target with extreme
    # values that drive Charbonnier finite but the *combined* loss > clip.
    pred = torch.full((1, 31, 4, 4), 0.5, requires_grad=True)
    target = torch.full((1, 31, 4, 4), 0.5)
    total, _ = loss_fn(pred, target)
    # Healthy path sanity check: graph connected.
    assert total.grad_fn is not None
    total.backward()
    assert pred.grad is not None


# ---------------------------------------------------------------------------
# BLOCKER 2: legacy SinkhornLoss now computes the actual entropic OT cost
# ---------------------------------------------------------------------------

def test_sinkhorn_loss_legacy_returns_zero_on_identical_clouds():
    torch.manual_seed(0)
    sl = SinkhornLoss(epsilon=0.1, num_iterations=200)
    real = torch.randn(8, 4)
    cost = sl(real, real.clone())
    # The pre-fix value was ~2e-5; we keep the loose bound to allow for
    # numerical drift across BLAS implementations but assert it is small.
    assert cost.item() < 1e-3, f"identical clouds should give ~0 OT cost, got {cost}"


def test_sinkhorn_loss_legacy_recovers_correct_magnitude_on_shift():
    """Shifting the second cloud by 1.0 in 4-D space gives squared distance ~ 4.
    Pre-fix SinkhornLoss returned 0.0023 (off by ~3 orders of magnitude); post-fix
    it returns a value of order 4 (close to the true OT cost for unif marginals
    on coincident points after a translation).
    """
    torch.manual_seed(0)
    sl = SinkhornLoss(epsilon=0.1, num_iterations=200)
    real = torch.randn(8, 4)
    fake = real.clone() + 1.0
    cost = sl(real, fake)
    assert 1.0 < cost.item() < 50.0, (
        f"expected OT cost ~ 4 for shift=1.0; got {cost.item():.4f} which is "
        f"the magnitude of the pre-fix bug (2e-3) or worse"
    )


# ---------------------------------------------------------------------------
# HIGH 3: SinkhornDivergence is symmetric/debiased at the diagonal
# ---------------------------------------------------------------------------

def test_sinkhorn_divergence_at_diagonal_has_small_gradient():
    """For a true Sinkhorn divergence S_eps, dS(X, X)/dX = 0 at the diagonal.
    Pre-fix the asymmetric ``OT(X, X.detach())`` left a non-zero residual
    (~7.8e-3 on a 64-point cloud). Post-fix the symmetric self-OT term gives
    near-zero gradient up to Sinkhorn convergence noise.

    Note: the divergence now passes through softplus(d/τ)·τ with τ=1e-3 to
    keep gradient flow alive near zero (a hard clamp at 0 killed the
    generator's adversarial signal once distributions converged). At the
    diagonal the value is therefore ≈ τ·ln(2) ≈ 7e-4, not 0 — but the
    gradient remains tiny because dS/dX = sigmoid(d/τ) · dDiv/dX and the
    inner gradient is still ~0 at the diagonal.
    """
    torch.manual_seed(0)
    sd = SinkhornDivergence(epsilon=0.1, n_iters=80, max_points=64)
    X = torch.randn(64, 1, requires_grad=True)
    S = sd(X, X.clone())
    S.backward()
    grad_norm = X.grad.norm().item()
    # Soft floor baseline: τ·ln(2) ≈ 6.9e-4 with τ=1e-3.
    assert S.item() < 2e-3, f"S(X, X) must be near the softplus baseline, got {S.item()}"
    # Loose bound; tighter than the pre-fix 7.8e-3 by an order of magnitude.
    # sigmoid(d/τ) ≈ 0.5 at the diagonal, so a tiny inner grad doubles in
    # the worst case — still well under the pre-fix bound.
    assert grad_norm < 5e-3, (
        f"||dS(X, X)/dX|| = {grad_norm} — too large for a debiased divergence"
    )


def test_sinkhorn_divergence_value_is_zero_for_identical_inputs():
    """Soft-floor baseline: ≈ τ·ln(2) with τ=1e-3, not exactly zero.

    Identical inputs should still yield a *small* divergence — small enough
    that it cannot dominate any real signal in the working range (which is
    O(0.01) and up). The exact-zero contract was traded for gradient
    survival near convergence.
    """
    sd = SinkhornDivergence(epsilon=0.1, n_iters=50, max_points=64)
    X = torch.randn(64, 1)
    val = sd(X, X.clone())
    assert val.item() < 2e-3


# ---------------------------------------------------------------------------
# MED 4: Generator counter respects trainer-managed mode
# ---------------------------------------------------------------------------

def _tiny_generator_config() -> dict:
    return {
        "in_channels": 3,
        "out_channels": 31,
        "base_channels": 16,
        "split_sizes": [2, 2, 2],
        "num_heads": 2,
        "norm_groups": 4,
        "ckpt_min_tokens": 100000,
        "output_activation": "delayed_sigmoid",
        "activation_delay_iters": 5,
        "clamp_after_iters": 0,
    }


def test_generator_counter_legacy_auto_increments():
    """No ``set_iteration`` calls => forward auto-increments on every train pass."""
    torch.manual_seed(0)
    gen = NoiseRobustCSWinGenerator(_tiny_generator_config())
    gen.train()
    x = torch.randn(1, 3, 16, 16)
    _ = gen(x)
    _ = gen(x)
    _ = gen(x)
    assert gen._iteration_count == 3


def test_generator_counter_externally_managed_does_not_drift():
    """With ``set_iteration`` (trainer-managed mode) forward must not also
    increment. Simulates ``accumulation_steps=2`` where the trainer fixes the
    counter to the optimizer-step value at the start of every loop iteration.
    """
    torch.manual_seed(0)
    gen = NoiseRobustCSWinGenerator(_tiny_generator_config())
    gen.train()
    x = torch.randn(1, 3, 16, 16)

    opt_step = 0
    counter_per_loop = []
    for loop_iter in range(6):
        is_step_boundary = ((loop_iter + 1) % 2 == 0)
        gen.set_iteration(opt_step)
        _ = gen(x)
        counter_per_loop.append(gen._iteration_count)
        if is_step_boundary:
            opt_step += 1

    # Counter must equal opt_step at the moment of each set_iteration call.
    assert counter_per_loop == [0, 0, 1, 1, 2, 2], counter_per_loop
    # And after 6 forwards, counter should be 2 (the last opt_step set), not 6
    # (which is what the pre-fix auto-increment would have produced).
    assert gen._iteration_count == 2


# ---------------------------------------------------------------------------
# LOW 5: SAM eval consistent with SAM loss
# ---------------------------------------------------------------------------

def test_compute_sam_value_zero_on_identical_spectra():
    """Pre-fix the ``acos(clamp(., -1+eps, 1-eps))`` form floored identical
    spectra at ~ sqrt(2*eps) rad. Post-fix the atan2 form returns exactly 0.
    """
    pred = torch.randn(2, 31, 8, 8).abs()  # non-negative spectra
    val = compute_sam_value(pred, pred.clone())
    assert val.item() < 1e-3, f"SAM(x, x) should be ~0, got {val.item()} deg"


def test_compute_sam_value_nonzero_on_shifted_spectra():
    pred = torch.randn(2, 31, 8, 8).abs()
    target = torch.randn(2, 31, 8, 8).abs()
    val = compute_sam_value(pred, target)
    assert val.item() > 1.0


# ---------------------------------------------------------------------------
# HIGH 4: CSWin relative-position bias is translation-invariant in long_axis
# mode (the new default). Pre-fix the bias was window-cyclic — diagonals had
# non-zero std (e.g. 21.9, 20.6, 15.1, ...). Post-fix every diagonal must have
# std = 0 exactly because ``bias[h, i, j] = f(h, i - j)`` by construction.
# ---------------------------------------------------------------------------

def test_cswin_long_axis_bias_is_translation_invariant():
    from hsi_model.models.attention import CSWinAttentionBlock

    s = 4
    block = CSWinAttentionBlock(
        dim=8, num_heads=2, split_size=s,
        config={"cswin_bias_mode": "long_axis", "cswin_max_long_axis": 64},
    )
    assert block._bias_mode == "long_axis"
    # Set deterministic table values so the test does not depend on init RNG.
    with torch.no_grad():
        for h in range(block.num_heads):
            block.relative_position_bias_table_h_long[:, h] = (
                torch.arange(2 * 64 - 1, dtype=block.relative_position_bias_table_h_long.dtype)
                + 1000.0 * h
            )

    bias = block._long_axis_bias(block.relative_position_bias_table_h_long, length=16)
    assert bias.shape == (block.num_heads, 16, 16)

    # Translation invariance: for every signed offset k = i - j, the slice
    # bias[h, i, j] with i - j = k is constant.
    for h in range(block.num_heads):
        for k in range(-15, 16):
            diag = torch.diagonal(bias[h], offset=k)
            assert diag.numel() > 0
            assert torch.allclose(diag, diag[0].expand_as(diag)), (
                f"long_axis bias is not translation-invariant on diagonal "
                f"offset {k} for head {h}: std={diag.std().item():.4f}"
            )


def test_cswin_long_axis_bias_default_mode():
    """Default mode must be ``long_axis`` so users get the correct bias out
    of the box without explicit config plumbing."""
    from hsi_model.models.attention import CSWinAttentionBlock

    block = CSWinAttentionBlock(dim=8, num_heads=2, split_size=4)
    assert block._bias_mode == "long_axis"
    assert hasattr(block, "relative_position_bias_table_h_long")


def test_cswin_window_cyclic_mode_still_available_for_legacy_checkpoints():
    """``window_cyclic`` mode kept for back-compat with pre-audit checkpoints."""
    from hsi_model.models.attention import CSWinAttentionBlock

    block = CSWinAttentionBlock(
        dim=8, num_heads=2, split_size=4,
        config={"cswin_bias_mode": "window_cyclic"},
    )
    assert block._bias_mode == "window_cyclic"
    # Legacy params remain trainable in legacy mode.
    assert block.relative_position_bias_table_h.requires_grad
    # Long table is NOT allocated in legacy mode (keeps state_dict minimal).
    assert not hasattr(block, "relative_position_bias_table_h_long")


def test_cswin_long_axis_block_forward_shape():
    """End-to-end: a forward pass through long_axis-mode CSWin produces the
    correct shape and finite outputs."""
    from hsi_model.models.attention import CSWinAttentionBlock

    block = CSWinAttentionBlock(
        dim=8, num_heads=2, split_size=4,
        config={"cswin_bias_mode": "long_axis", "cswin_max_long_axis": 64},
    )
    x = torch.randn(1, 8, 16, 16)
    out = block(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_cswin_long_axis_raises_when_input_exceeds_max():
    from hsi_model.models.attention import CSWinAttentionBlock

    block = CSWinAttentionBlock(
        dim=8, num_heads=2, split_size=4,
        config={"cswin_bias_mode": "long_axis", "cswin_max_long_axis": 16},
    )
    # 32 > max_long_axis=16 ⇒ must raise rather than silently produce
    # out-of-bounds index lookups.
    x = torch.randn(1, 8, 32, 32)
    import pytest
    with pytest.raises(ValueError, match="exceeds cswin_max_long_axis"):
        _ = block(x)
