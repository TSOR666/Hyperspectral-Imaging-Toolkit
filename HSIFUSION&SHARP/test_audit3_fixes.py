"""Regression tests for the 2026-06 (audit pass 3) bottleneck-audit fixes.

Run (single-threaded to avoid a preview-torch multithreaded-BLAS segfault on large allocs):

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
        python -m pytest -q test_audit3_fixes.py

Each test pins one confirmed finding so the fix cannot silently regress.
"""

import copy
import logging
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# SHARP output parameterization: sigmoid default -> [0,1]; tanh still available
# ---------------------------------------------------------------------------
def test_sharp_output_activation_default_is_bounded_nonnegative():
    from sharp_v322_hardened import create_sharp_v32

    m = create_sharp_v32("tiny", compile_model=False, verbose=False).eval()
    assert m.config.output_activation == "sigmoid"
    with torch.no_grad():
        y = m(torch.rand(1, 3, 48, 48))
    assert y.min().item() >= 0.0, "sigmoid output must be nonnegative (reflectance)"
    assert y.max().item() <= 1.0, "sigmoid output must be <= 1"


def test_sharp_output_activation_dispatch_all_modes():
    from sharp_v322_hardened import create_sharp_v32

    x = torch.rand(1, 3, 32, 32)
    ranges = {}
    for act in ("sigmoid", "tanh", "relu", "softplus", "none"):
        m = create_sharp_v32("tiny", compile_model=False, verbose=False,
                             output_activation=act).eval()
        with torch.no_grad():
            y = m(x)
        assert torch.isfinite(y).all()
        ranges[act] = (y.min().item(), y.max().item())
    assert ranges["tanh"][0] < 0.0, "legacy tanh must still be able to emit negatives"
    assert ranges["sigmoid"][0] >= 0.0 and ranges["sigmoid"][1] <= 1.0
    assert ranges["relu"][0] >= 0.0


def test_sharp_invalid_output_activation_rejected():
    from sharp_v322_hardened import SHARPv32Config

    with pytest.raises(ValueError):
        SHARPv32Config(output_activation="elu")


def test_sharp_perfect_prediction_zero_loss_with_sigmoid():
    # compute_loss(x, x) must still be exactly 0 (the prior-audit invariant) under the
    # new default output activation.
    from sharp_v322_hardened import create_sharp_v32

    m = create_sharp_v32("tiny", compile_model=False, verbose=False)
    t = torch.rand(2, 31, 16, 16)
    assert m.compute_loss(t, t).item() == 0.0


# ---------------------------------------------------------------------------
# common_utils.sparse_attention_topk: .view on non-contiguous tensor -> .reshape
# ---------------------------------------------------------------------------
def test_sparse_attention_topk_noncontiguous_inputs():
    from common_utils_v32 import sparse_attention_topk

    # permute makes q/k/v non-contiguous -> the old .view() raised RuntimeError.
    base = torch.randn(2, 16, 4, 8)
    q, k, v = (base.permute(0, 2, 1, 3) for _ in range(3))
    assert not v.is_contiguous()
    out = sparse_attention_topk(q, k, v, sparsity_ratio=0.5)
    assert out.shape == q.shape
    assert torch.isfinite(out).all()


def test_hsifusion_sparse_attention_path_forward_backward():
    # OptimizedDynamicSparseAttention (use_sparse_attention=True) crashed on first call.
    from hsifusion_v252_complete import create_hsifusion_lightning_pro

    m = create_hsifusion_lightning_pro(
        "tiny", compile_mode=None, force_compile=False,
        use_sparse_attention=True, use_moe=False, estimate_uncertainty=False, min_input_size=32,
    )
    x = torch.rand(1, 3, 64, 64, requires_grad=True)
    out = m(x)
    out = out[0] if isinstance(out, tuple) else out
    out.mean().backward()
    assert torch.isfinite(out).all() and torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# VectorizedWindowedSparsemax: preserve caller dtype (no silent fp32 upcast)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seq_len", [50, 150])  # <=window (short path) and >window (padded path)
def test_vectorized_windowed_sparsemax_preserves_dtype(seq_len):
    from sharp_v322_hardened import VectorizedWindowedSparsemax

    vsm = VectorizedWindowedSparsemax(window_size=64)
    out = vsm(torch.randn(2, seq_len, dtype=torch.float16))
    assert out.dtype == torch.float16


# ---------------------------------------------------------------------------
# _factor_pair: always returns an exact factorization (h*w == n)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [13, 17, 31, 97, 101, 4096, 4097, 256, 1000])
def test_factor_pair_is_exact(n, recwarn):
    from hsifusion_v252_complete import _factor_pair

    h, w = _factor_pair(n)
    assert h * w == n, f"_factor_pair({n}) -> {h}x{w} (product {h*w} != {n})"


# ---------------------------------------------------------------------------
# SHARP trainer criterion routing (MRAE vs model.compute_loss)
# ---------------------------------------------------------------------------
def test_sharp_trainer_criterion_routing():
    from sharp_v322_hardened import SHARPv32Trainer, create_sharp_v32
    from optimized_dataloader import MSTPlusPlusLoss

    m = create_sharp_v32("tiny", compile_model=False, verbose=False)
    x, t = torch.rand(1, 3, 32, 32), torch.rand(1, 31, 32, 32)
    pred = m(x)

    tr_mrae = SHARPv32Trainer(m, total_steps=10, ema_decay=0.0, use_amp=False,
                              criterion=MSTPlusPlusLoss())
    manual = torch.mean(torch.abs(pred - t) / torch.clamp_min(torch.abs(t), 1e-6)).item()
    assert abs(tr_mrae._loss(pred, t).item() - manual) < 1e-4

    tr_default = SHARPv32Trainer(m, total_steps=10, ema_decay=0.0, use_amp=False)
    assert abs(tr_default._loss(pred, t).item() - m.compute_loss(pred, t).item()) < 1e-6


def test_sharp_training_config_loss_type_default_is_mrae():
    from sharp_training_script_fixed import SHARPTrainingConfig

    cfg = SHARPTrainingConfig()
    assert cfg.loss_type == "mrae"
    assert cfg.output_activation == "sigmoid"
    with pytest.raises(ValueError):
        SHARPTrainingConfig(loss_type="huber")


# ---------------------------------------------------------------------------
# HSIFusion config defaults realigned with audit/training
# ---------------------------------------------------------------------------
def test_hsifusion_config_defaults():
    from hsifusion_v252_complete import LightningProConfig

    cfg = LightningProConfig()
    assert cfg.cross_attention_max_tokens == 1024
    assert cfg.estimate_uncertainty is False


# ---------------------------------------------------------------------------
# Dataloader HSI range warning (warn, never auto-normalize)
# ---------------------------------------------------------------------------
def test_hsi_range_warning_fires_and_does_not_mutate():
    import optimized_dataloader as D

    D._HSI_RANGE_WARNED = False
    arr = np.full((31, 4, 4), 5000.0, dtype=np.float32)
    snapshot = arr.copy()
    D._warn_if_hsi_out_of_range(arr, "bad.mat")
    assert D._HSI_RANGE_WARNED is True
    assert np.array_equal(arr, snapshot), "range check must NOT modify the cube"


def test_hsi_range_warning_quiet_for_normalized():
    import optimized_dataloader as D

    D._HSI_RANGE_WARNED = False
    D._warn_if_hsi_out_of_range(np.random.rand(31, 4, 4).astype(np.float32), "ok.mat")
    assert D._HSI_RANGE_WARNED is False


# ---------------------------------------------------------------------------
# Gradient-accumulation loss scaling: each optimizer step is a proper average
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("total_steps,accum", [(7, 3), (10, 4), (9, 3), (5, 1)])
def test_accumulation_group_scaling_sums_to_one(total_steps, accum):
    from hsifusion_training import should_optimizer_step

    n_full = (total_steps // accum) * accum
    group_weight = 0.0
    seen_groups = 0
    for batch_idx in range(total_steps):
        group_size = accum if batch_idx < n_full else (total_steps - n_full)
        group_weight += 1.0 / max(1, group_size)
        if should_optimizer_step(batch_idx, total_steps, accum):
            # Every completed accumulation group must contribute weight exactly 1.0
            assert abs(group_weight - 1.0) < 1e-6, (
                f"group ending at {batch_idx} summed to {group_weight} (total={total_steps}, accum={accum})"
            )
            group_weight = 0.0
            seen_groups += 1
    assert seen_groups >= 1


# ---------------------------------------------------------------------------
# Manual-path EMA validation: evaluates EMA weights, then restores originals
# ---------------------------------------------------------------------------
def test_manual_validate_uses_ema_and_restores():
    import sharp_training_script_fixed as S
    from sharp_v322_hardened import create_sharp_v32

    class Mini(S.DedicatedSHARPTrainer):
        def __init__(self):
            self.config = SimpleNamespace(use_amp=False, val_crop=False, val_crop_size=(8, 8),
                                          psnr_data_range=1.0, min_mrae_denom=1e-6)
            self.device = torch.device("cpu")
            self.amp_dtype = None
            self.model = create_sharp_v32("tiny", compile_model=False, verbose=False).eval()
            self.ema_state = {n: p.detach().clone() + 0.5
                              for n, p in self.model.named_parameters() if p.requires_grad}
            self.val_loader = [(torch.rand(1, 3, 32, 32), torch.rand(1, 31, 32, 32))]

    mini = Mini()
    before = {n: p.detach().clone() for n, p in mini.model.named_parameters()}
    metrics = mini._manual_validate()
    after = {n: p.detach().clone() for n, p in mini.model.named_parameters()}
    assert all(torch.equal(before[n], after[n]) for n in before), "EMA weights not restored"
    assert np.isfinite(metrics["mrae"])
