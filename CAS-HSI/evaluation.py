"""Shared evaluation protocol for CAS-HSI (used by the trainer and the test tool).

Two protocols are reported for every scene, exactly as the sibling projects do:

* ``full``  -- metrics over the whole native-resolution frame (NTIRE-style).
* ``crop``  -- metrics over the MST++ centre region, i.e. after discarding a
  128-pixel border (482x512 -> 226x256).  MST++ selects checkpoints on this, so
  it is the **selection metric** here too; reporting only the full-frame number
  while selecting on the crop would be comparing two different quantities.

Predictions are **not clamped** by default.  That is the NTIRE convention and what
the rest of this repo does.  ``clamp`` is available (spec 7.4 permits it for
evaluation) but it flatters MRAE by silently repairing out-of-range predictions,
so it is off unless you ask.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import torch
import torch.nn as nn

METRIC_KEYS = ("mrae", "rmse", "psnr", "sam", "ssim", "mae")
MST_CROP_BORDER = 128

# The repo-wide metric implementations live at the toolkit root. Using them (rather
# than a private copy) is what makes a CAS-HSI number comparable to an MSWR or
# SHARP number.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from hsi_benchmark.metrics import compute_hsi_metrics  # type: ignore

    METRICS_SOURCE = "hsi_benchmark"
except ImportError:  # pragma: no cover - standalone checkout
    from metrics import batch_metrics as _batch_metrics

    METRICS_SOURCE = "cas_hsi.metrics (fallback; hsi_benchmark not importable)"

    def compute_hsi_metrics(  # type: ignore[misc]
        prediction, target, *, epsilon: float = 1e-6, crop_border: int = 0
    ):
        pred = torch.as_tensor(prediction, dtype=torch.float32)
        truth = torch.as_tensor(target, dtype=torch.float32)
        if pred.ndim == 3:
            pred, truth = pred.unsqueeze(0), truth.unsqueeze(0)
        if crop_border:
            pred = pred[..., crop_border:-crop_border, crop_border:-crop_border]
            truth = truth[..., crop_border:-crop_border, crop_border:-crop_border]
        return _batch_metrics(pred, truth, epsilon=epsilon), {}


__all__ = [
    "METRIC_KEYS",
    "MST_CROP_BORDER",
    "METRICS_SOURCE",
    "evaluate_scene",
    "forward_scene",
    "format_metric_table",
    "average_rows",
]


@torch.no_grad()
def forward_scene(
    model: nn.Module,
    rgb: torch.Tensor,
    *,
    tile_size: int = 0,
    overlap: int = 32,
) -> torch.Tensor:
    """Reconstruct one full scene.

    The model pads and crops internally (spec 8), so no padding is needed here.
    ``tile_size > 0`` routes through overlapping tiles for scenes too large to fit
    in memory (spec 8.7).
    """
    if tile_size and tile_size > 0:
        from cas_hsi.inference import tiled_inference

        prediction = tiled_inference(model, rgb, tile_size=tile_size, overlap=overlap)
    else:
        prediction = model(rgb)

    if isinstance(prediction, tuple):
        prediction = prediction[0]
    return prediction


def evaluate_scene(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    crop_border: int = MST_CROP_BORDER,
    epsilon: float = 1e-6,
    clamp: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Full-frame and MST++-crop metrics for a single scene, in fp32 on CPU."""
    pred = prediction.detach().float().cpu()
    truth = target.detach().float().cpu()

    if clamp:
        pred = pred.clamp(0.0, 1.0)

    rows: Dict[str, Dict[str, float]] = {}
    full, _ = compute_hsi_metrics(pred, truth, epsilon=epsilon)
    rows["full"] = {key: float(full[key]) for key in METRIC_KEYS}

    height, width = truth.shape[-2:]
    if crop_border > 0 and height > 2 * crop_border and width > 2 * crop_border:
        cropped, _ = compute_hsi_metrics(pred, truth, epsilon=epsilon, crop_border=crop_border)
        rows["crop"] = {key: float(cropped[key]) for key in METRIC_KEYS}

    return rows


def average_rows(rows: List[Mapping[str, float]]) -> Dict[str, float]:
    """Mean of each metric over scenes (the scene is the unit, as in NTIRE)."""
    if not rows:
        return {}
    return {
        key: float(sum(row[key] for row in rows) / len(rows))
        for key in METRIC_KEYS
        if key in rows[0]
    }


def format_metric_table(
    protocols: Mapping[str, Mapping[str, float]],
    indent: str = "  ",
    extra: Optional[Mapping[str, Any]] = None,
) -> str:
    header = f"{indent}{'protocol':<10}" + "".join(f"{key.upper():>10}" for key in METRIC_KEYS)
    lines = [header, indent + "-" * (10 + 10 * len(METRIC_KEYS))]
    for name, row in protocols.items():
        lines.append(
            f"{indent}{name:<10}"
            + "".join(f"{row[key]:>10.4f}" if key in row else f"{'-':>10}" for key in METRIC_KEYS)
        )
    if extra:
        lines.append(indent + "  ".join(f"{k}={v}" for k, v in extra.items()))
    return "\n".join(lines)
