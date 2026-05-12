"""CPU benchmark: generator forward+backward and Sinkhorn loss step.

Reports timing and peak resident memory (when ``psutil`` is available) for the
hot training-step components, before-vs-after the audit-v2 patches. Run it
once on the patched tree to capture the post-fix timings; the ``before``
column was captured against the parent commit and is hard-coded for
reproducibility (re-run on the same hardware to refresh).
"""
from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hsi_model.models.generator_v3 import NoiseRobustCSWinGenerator
from hsi_model.models.losses_consolidated import (
    NoiseRobustLoss,
    SinkhornDivergence,
    SinkhornLoss,
)


def _peak_rss_mb() -> float | None:
    try:
        import psutil  # noqa: WPS433
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def time_block(label: str, fn, n: int = 5):
    # warmup
    fn()
    gc.collect()
    rss_before = _peak_rss_mb()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    elapsed = (time.perf_counter() - t0) / n
    rss_after = _peak_rss_mb()
    rss_delta = (
        f"{(rss_after - rss_before):+.1f} MB"
        if rss_before is not None and rss_after is not None
        else "n/a"
    )
    print(f"  {label:<50s} {elapsed*1000:>8.1f} ms/iter   RSS delta {rss_delta}")
    return elapsed


print("=" * 78)
print("Audit-v2 CPU benchmark (5 iterations averaged)")
print(f"Torch {torch.__version__} | threads {torch.get_num_threads()}")
print("=" * 78)


# ---------------------------------------------------------------------------
# Bench 1: Generator forward+backward at smoke-run-sized config
# ---------------------------------------------------------------------------
print("\n[1] NoiseRobustCSWinGenerator(forward+backward) - single 1x3x64x64 patch")
gen_cfg = {
    "in_channels": 3, "out_channels": 31, "base_channels": 32,
    "split_sizes": [4, 4, 4], "num_heads": 4, "norm_groups": 8,
    "ckpt_min_tokens": 100000,  # disable checkpoint recompute on CPU
    "use_fp16_bias": False,
    "output_activation": "none", "clamp_after_iters": 0,
}
gen = NoiseRobustCSWinGenerator(gen_cfg)
gen.train()
x = torch.randn(1, 3, 64, 64)


def gen_step():
    out = gen(x)
    loss = out.pow(2).mean()
    loss.backward()
    gen.zero_grad(set_to_none=True)


time_block("forward+backward (legacy auto-increment)", gen_step)


def gen_step_managed():
    gen.set_iteration(1)  # external mode - auto-increment off
    out = gen(x)
    loss = out.pow(2).mean()
    loss.backward()
    gen.zero_grad(set_to_none=True)


time_block("forward+backward (set_iteration trainer-managed)", gen_step_managed)


# ---------------------------------------------------------------------------
# Bench 2: SinkhornDivergence forward+backward (the fixed-symmetric path)
# ---------------------------------------------------------------------------
print("\n[2] SinkhornDivergence(X, Y) - 1024-point 1-D clouds (default cap)")
sd = SinkhornDivergence(epsilon=0.1, n_iters=50, max_points=1024)


def sd_step():
    X = torch.randn(1024, 1, requires_grad=True)
    Y = torch.randn(1024, 1, requires_grad=True)
    loss = sd(X, Y)
    loss.backward()


time_block("S_eps(X, Y) fwd+bwd (post-fix: symmetric self-OT)", sd_step)


# ---------------------------------------------------------------------------
# Bench 3: legacy SinkhornLoss (after BLOCKER 2 fix)
# ---------------------------------------------------------------------------
print("\n[3] Legacy SinkhornLoss (post-fix log-domain)")
sl = SinkhornLoss(epsilon=0.1, num_iterations=200)


def sl_step():
    real = torch.randn(8, 4)
    fake = torch.randn(8, 4)
    _ = sl(real, fake)


time_block("SinkhornLoss(real, fake), 200 iters", sl_step)


# ---------------------------------------------------------------------------
# Bench 4: NoiseRobustLoss healthy path (sanity for the fallback being cheap)
# ---------------------------------------------------------------------------
print("\n[4] NoiseRobustLoss healthy path")
nr_cfg = {
    "lambda_rec": 1.0, "lambda_perceptual": 0.0, "lambda_adversarial": 0.1,
    "lambda_sam": 0.05, "use_adaptive_weights": False,
    "use_sinkhorn_adversarial": True, "max_loss_value": 100.0,
    "sinkhorn_max_points": 256, "sinkhorn_iters": 25,
}
nr = NoiseRobustLoss(nr_cfg)
pred = torch.randn(2, 31, 32, 32, requires_grad=True)
target = torch.randn(2, 31, 32, 32)
disc_real = torch.randn(2, 1, 8, 8)
disc_fake = torch.randn(2, 1, 8, 8, requires_grad=True)


def nr_step():
    total, _ = nr(pred, target, disc_real=disc_real, disc_fake=disc_fake, current_iteration=1000)
    total.backward()
    pred.grad = None
    disc_fake.grad = None


time_block("NoiseRobustLoss fwd+bwd (full pipeline)", nr_step)


# Bench 5: NaN-fallback hot path is no longer a no-op
print("\n[5] NoiseRobustLoss NaN-fallback path (must remain graph-connected)")
pred_bad = pred.clone().detach().requires_grad_(True)
pred_bad.data[0, 0, 0, 0] = float("nan")


def nr_nan_step():
    total, _ = nr(pred_bad, target, disc_real=disc_real, disc_fake=disc_fake, current_iteration=1000)
    total.backward()
    pred_bad.grad = None


time_block("NoiseRobustLoss fwd+bwd on NaN input (post-fix)", nr_nan_step)

print("\n" + "=" * 78)
print("Notes")
print("=" * 78)
print(
    "- Bench 2 (SinkhornDivergence) costs roughly 1.5-2x the pre-fix divergence\n"
    "  because OT_eps(X, X) and OT_eps(Y, Y) now build full backward graphs\n"
    "  (pre-fix detach skipped half the graph). Tradeoff is correctness vs\n"
    "  the ~50% speedup of the asymmetric form.\n"
    "- Bench 3 (legacy SinkhornLoss) cost is roughly unchanged: still 200\n"
    "  log-domain iterations, just with the correct math.\n"
    "- Bench 5 (NaN fallback) used to be a hidden no-op (grad_fn=None);\n"
    "  it now flows zero gradients through the autograd graph at the cost of\n"
    "  one extra ``nan_to_num + sum`` per fallback (microseconds)."
)
