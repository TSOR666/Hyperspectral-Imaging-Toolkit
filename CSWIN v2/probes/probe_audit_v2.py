"""Independent probes for audit v2.

Each function below verifies (or refutes) a specific suspected issue.
Run with: python probes/probe_audit_v2.py
"""
from __future__ import annotations

import sys
from pathlib import Path
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hsi_model.models.losses_consolidated import (
    SinkhornDivergence,
    SinkhornLoss,
    NoiseRobustLoss,
)
from hsi_model.models.attention import CSWinAttentionBlock


def banner(msg):
    print("\n" + "=" * 72)
    print(msg)
    print("=" * 72)


# ----------------------------------------------------------------------
# Probe 1: SinkhornDivergence(X, X) — should be ~0
# ----------------------------------------------------------------------
banner("Probe 1: SinkhornDivergence on identical clouds (should be ~0)")
torch.manual_seed(0)
sd = SinkhornDivergence(epsilon=0.1, n_iters=50, max_points=64)
X = torch.randn(64, 1) * 0.5  # 1-D scalar point cloud (matches discriminator path)
v_xx = sd(X, X.clone())
print(f"S_eps(X, X)        = {v_xx.item():.6f}")
v_xy = sd(X, X.clone() + 0.5)
print(f"S_eps(X, X+0.5)    = {v_xy.item():.6f}")

# Probe the divergence's behavior at the diagonal: dS(X, X)/dX should be 0.
# Pre-fix: detach asymmetry leaked self-term gradient into X.
# Post-fix: symmetric self-OT with full debiasing, gradient at the diagonal
# vanishes (up to Sinkhorn convergence noise).
X_g = torch.randn(64, 1, requires_grad=True)
S_xx = sd(X_g, X_g.clone())
S_xx.backward()
print(f"||dS(X, X)/dX||_post_fix      = {X_g.grad.norm().item():.6e} "
      f"(should be near 0 for a true Sinkhorn divergence at the diagonal)")


# ----------------------------------------------------------------------
# Probe 2: Legacy SinkhornLoss math
# ----------------------------------------------------------------------
banner("Probe 2: Legacy SinkhornLoss — log-space update math")
torch.manual_seed(0)
sl = SinkhornLoss(epsilon=0.1, num_iterations=200)
real = torch.randn(8, 4)
fake_same = real.clone()
fake_diff = real.clone() + 1.0
v_same = sl(real, fake_same)
v_diff = sl(real, fake_diff)
print(f"SinkhornLoss(real, real)       = {v_same.item():.6f}  (should be ≈ 0)")
print(f"SinkhornLoss(real, real+1.0)   = {v_diff.item():.6f}  (should be > 0)")
# Compare to a reference torch-only Sinkhorn cost for the same inputs
def reference_sinkhorn_cost(x, y, eps=0.1, n_iters=200):
    n = x.shape[0]
    a = torch.full((n,), 1.0 / n)
    b = torch.full((n,), 1.0 / n)
    C = torch.cdist(x, y) ** 2
    K = torch.exp(-C / eps)
    K = K.clamp_min(1e-30)
    u = torch.ones(n)
    v = torch.ones(n)
    for _ in range(n_iters):
        u = a / (K @ v).clamp_min(1e-30)
        v = b / (K.T @ u).clamp_min(1e-30)
    pi = u[:, None] * K * v[None, :]
    return torch.sum(pi * C)
ref_same = reference_sinkhorn_cost(real, fake_same)
ref_diff = reference_sinkhorn_cost(real, fake_diff)
print(f"Reference Sinkhorn(real, real)     = {ref_same.item():.6f}")
print(f"Reference Sinkhorn(real, real+1.0) = {ref_diff.item():.6f}")


# ----------------------------------------------------------------------
# Probe 3: CSWin _expand_bias — does it encode correct relative positions?
# ----------------------------------------------------------------------
banner("Probe 3: CSWin relative-position bias along the long axis")
config = {"ckpt_min_tokens": 1, "use_fp16_bias": False, "norm_groups": 8}
torch.manual_seed(0)
block = CSWinAttentionBlock(dim=8, num_heads=2, split_size=4, config=config)

# Set the bias table to a deterministic pattern: table[i, j, h] = 100*i + 10*j + h
table = torch.zeros(2 * 4 - 1, 2 * 4 - 1, 2)
for i in range(2 * 4 - 1):
    for j in range(2 * 4 - 1):
        for h in range(2):
            table[i, j, h] = 100 * i + 10 * j + h
with torch.no_grad():
    block.relative_position_bias_table_h.copy_(table)

# Read what _expand_bias produces along the long axis for a bias_ss derived
# from a "row=center" slice (mimics how horizontal attention uses it).
rel_cols = block._relative_position_index  # (s, s)
bias_ss = block.relative_position_bias_table_h[
    block._relative_center_index, rel_cols, :
]  # (s, s, num_heads)
expanded = block._expand_bias(bias_ss, tiles_long=4)  # (num_heads, 16, 16)
print(f"bias_ss (head 0):\n{bias_ss[..., 0]}")
print(f"\nexpanded[0] first row (along W axis, len=16):\n{expanded[0, 0]}")
print(f"\nexpanded[0] diagonal (i==j relative position 0 expected):\n"
      f"{torch.diagonal(expanded[0])}")
# True relative bias along W with (2W-1) table entries would have:
#   - constant value along each diagonal (i-j = const)
# Whether _expand_bias has this property:
diag_consistency = []
for k in range(-15, 16):
    diag = torch.diagonal(expanded[0], offset=k)
    if diag.numel() > 1:
        diag_consistency.append((k, diag.std().item()))
print("\nPer-diagonal std (non-zero ⇒ position bias varies along the same "
      "relative offset, which a true relative-pos bias would not):")
for k, s in diag_consistency[:8]:
    print(f"  diag offset {k}: std = {s:.4f}")
print(f"  ... (showing first 8 of {len(diag_consistency)})")
print("Reference: a true (W,W) relative-position bias has std=0 on every "
      "diagonal (translation-invariance).")

# Probe 3b: long_axis mode (the post-fix default) MUST be translation-invariant.
print("\n--- Post-fix: cswin_bias_mode='long_axis' (default) ---")
torch.manual_seed(0)
block_long = CSWinAttentionBlock(
    dim=8, num_heads=2, split_size=4,
    config={"cswin_bias_mode": "long_axis", "cswin_max_long_axis": 64,
            "ckpt_min_tokens": 1, "use_fp16_bias": False, "norm_groups": 8},
)
# Set deterministic table values
with torch.no_grad():
    for h in range(block_long.num_heads):
        block_long.relative_position_bias_table_h_long[:, h] = (
            torch.arange(2 * 64 - 1).float() + 1000.0 * h
        )
bias_long = block_long._long_axis_bias(
    block_long.relative_position_bias_table_h_long, length=16
)
print(f"long_axis bias[0] first row : {bias_long[0, 0]}")
print("Per-diagonal std (must all be 0 for translation invariance):")
worst = 0.0
for k in range(-15, 16):
    d = torch.diagonal(bias_long[0], offset=k)
    if d.numel() > 1:
        s_ = d.std().item()
        worst = max(worst, s_)
print(f"  worst std across all diagonals: {worst:.4e}  "
      f"(pre-fix worst was ~21.9)")


# ----------------------------------------------------------------------
# Probe 4: NoiseRobustLoss failure-mode disconnection
# ----------------------------------------------------------------------
banner("Probe 4: NoiseRobustLoss returns disconnected tensor on NaN input")
config = {
    "lambda_rec": 1.0, "lambda_perceptual": 0.0, "lambda_adversarial": 0.1,
    "lambda_sam": 0.05, "use_adaptive_weights": False,
    "use_sinkhorn_adversarial": True, "max_loss_value": 100.0,
}
loss_fn = NoiseRobustLoss(config)

# Build a tiny "model" to test gradient flow: pred is a leaf with grad
pred = torch.randn(2, 31, 8, 8, requires_grad=True)
pred_bad = pred.clone()
pred_bad[0, 0, 0, 0] = float("nan")
target = torch.randn(2, 31, 8, 8)

total, comps = loss_fn(pred_bad, target)
print(f"Total loss when pred has NaN: {total.item():.4f}")
print(f"Total loss requires_grad     : {total.requires_grad}")
print(f"Total loss grad_fn           : {total.grad_fn}")
print(f"pred.grad before backward    : {pred.grad}")
try:
    total.backward()
    print(f"pred.grad after backward     : {pred.grad}")
except Exception as e:
    print(f"backward raised              : {type(e).__name__}: {e}")
# Compare to a healthy run:
total2, _ = loss_fn(pred, target)
print(f"\nHealthy: total.requires_grad = {total2.requires_grad}, "
      f"grad_fn = {total2.grad_fn.__class__.__name__ if total2.grad_fn else None}")


# ----------------------------------------------------------------------
# Probe 5: Generator iteration drift under gradient accumulation
# ----------------------------------------------------------------------
banner("Probe 5: Generator auto-increment behavior under multi-forward")
from hsi_model.models.generator_v3 import NoiseRobustCSWinGenerator
gen_cfg = {
    "in_channels": 3, "out_channels": 31, "base_channels": 16,
    "split_sizes": [2, 2, 2], "num_heads": 2, "norm_groups": 4,
    "ckpt_min_tokens": 100000,  # disable ckpt to keep it fast
    "output_activation": "delayed_sigmoid", "activation_delay_iters": 5,
    "clamp_after_iters": 0,
}
gen = NoiseRobustCSWinGenerator(gen_cfg)
gen.train()
x = torch.randn(1, 3, 16, 16)
print(f"Initial _iteration_count: {gen._iteration_count}")
print("LEGACY (no set_iteration): forward auto-increments per call")
for step in range(3):
    _ = gen(x); _ = gen(x)
    print(f"  After step {step+1} (2 forwards): _iteration_count = {gen._iteration_count}")

print("\nTRAINER-MANAGED (set_iteration each step, accumulation_steps=2):")
gen2 = NoiseRobustCSWinGenerator(gen_cfg)
gen2.train()
opt_step = 0
for loop_iter in range(6):
    is_step_boundary = ((loop_iter + 1) % 2 == 0)
    gen2.set_iteration(opt_step)
    _ = gen2(x)
    if is_step_boundary:
        opt_step += 1
    print(f"  loop_iter={loop_iter} opt_step_after={opt_step} "
          f"_iteration_count={gen2._iteration_count}")
print("  Expected after 6 forwards / 3 optimizer steps: _iteration_count = 3 "
      "(opt_step value at last set_iteration call)")
