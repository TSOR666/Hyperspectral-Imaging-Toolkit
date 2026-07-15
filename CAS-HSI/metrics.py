"""Losses and metrics for CAS-HSI.

Two consumers, one set of definitions:

* **Training** uses the on-device torch modules below (fast, per-batch, autograd
  for the loss).
* **Validation / test** uses ``hsi_benchmark.metrics.compute_hsi_metrics``, the
  repo-wide implementation the other projects report against.

The torch definitions here are written to match ``hsi_benchmark`` numerically, so
a train-set MRAE and a val-set MRAE are the same quantity.  In particular PSNR is
``-10*log10(mse)`` on [0, 1] reflectance (data_range = 1) -- *not* the 8-bit
``data_range=255`` variant, and the prediction is **not** clamped.

Clamping policy (spec 7.4): the loss NEVER sees a clamped prediction. ``clamp`` has
zero gradient outside its range, so clamping before the loss would silently freeze
learning on exactly the elements that are furthest from the target. Evaluation may
clamp (``--clamp_eval``), but the NTIRE/repo default is unclamped.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "MRAE_EPSILON",
    "METRIC_KEYS",
    "AverageMeter",
    "Loss_MRAE",
    "Loss_RMSE",
    "Loss_PSNR",
    "Loss_SSIM",
    "Loss_SAM",
    "batch_metrics",
    "build_criterion",
]

MRAE_EPSILON = 1e-6
METRIC_KEYS = ("mrae", "rmse", "psnr", "sam", "ssim", "mae")


def _pair(prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if prediction.shape != target.shape:
        raise ValueError(f"Shape mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}")
    # Metric arithmetic in fp32 at minimum: under bf16 autocast a small absolute
    # error would otherwise round away entirely and flatter the number.
    dtype = torch.float64 if prediction.dtype == torch.float64 else torch.float32
    return prediction.to(dtype), target.to(dtype)


class AverageMeter:
    """Running mean of a scalar."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.val = float(value)
        self.sum += float(value) * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0.0


class Loss_MRAE(nn.Module):
    """Mean Relative Absolute Error -- the ARAD-1K / NTIRE objective.

    ``mean(|pred - gt| / max(|gt|, eps))``.  The denominator is clamped rather than
    offset so that a target of exactly zero yields a finite, bounded term instead of
    a division by zero.
    """

    def __init__(self, epsilon: float = MRAE_EPSILON) -> None:
        super().__init__()
        self.epsilon = float(epsilon)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction, target = _pair(prediction, target)
        denominator = torch.clamp_min(target.abs(), self.epsilon)
        return ((prediction - target).abs() / denominator).mean()


class Loss_RMSE(nn.Module):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction, target = _pair(prediction, target)
        return torch.sqrt((prediction - target).pow(2).mean())


class Loss_PSNR(nn.Module):
    """PSNR on [0, 1] reflectance: ``-10 log10(MSE)``.

    Matches ``hsi_benchmark.metrics``. Unclamped, data_range = 1.
    """

    def __init__(self, data_range: float = 1.0) -> None:
        super().__init__()
        self.data_range = float(data_range)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction, target = _pair(prediction, target)
        mse = (prediction - target).pow(2).mean()
        if float(mse) <= 1e-12:
            return torch.full((), 100.0, device=prediction.device, dtype=prediction.dtype)
        return 10.0 * torch.log10(
            torch.tensor(self.data_range ** 2, device=mse.device, dtype=mse.dtype) / mse
        )


class Loss_SSIM(nn.Module):
    """Per-channel SSIM with an 11x11 box window, averaged over bands and pixels."""

    def __init__(self, window_size: int = 11, data_range: float = 1.0) -> None:
        super().__init__()
        self.window_size = int(window_size)
        self.data_range = float(data_range)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction, target = _pair(prediction, target)

        window = self.window_size
        smallest = min(prediction.shape[-2], prediction.shape[-1])
        if smallest < window:
            # Shrink to the largest odd window that fits rather than returning a
            # fake 1.0, which would silently inflate the metric on small patches.
            window = max(3, smallest | 1)
            if smallest < 3:
                return torch.ones((), device=prediction.device, dtype=prediction.dtype)

        padding = window // 2
        mu_x = F.avg_pool2d(prediction, window, 1, padding)
        mu_y = F.avg_pool2d(target, window, 1, padding)

        sigma_x = F.avg_pool2d(prediction.square(), window, 1, padding) - mu_x.square()
        sigma_y = F.avg_pool2d(target.square(), window, 1, padding) - mu_y.square()
        sigma_xy = F.avg_pool2d(prediction * target, window, 1, padding) - mu_x * mu_y

        c1 = (0.01 * self.data_range) ** 2
        c2 = (0.03 * self.data_range) ** 2

        numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
        return (numerator / denominator.clamp_min(1e-12)).mean()


class Loss_SAM(nn.Module):
    """Spectral Angle Mapper, in degrees.

    Uses ``atan2(||orthogonal component||, cosine)`` rather than ``acos(cosine)``:
    acos loses all precision as the angle approaches 0, which is exactly the regime
    a good reconstruction lives in.
    """

    def __init__(self, epsilon: float = MRAE_EPSILON) -> None:
        super().__init__()
        self.epsilon = float(epsilon)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction, target = _pair(prediction, target)
        pred_unit = F.normalize(prediction, dim=1, eps=self.epsilon)
        target_unit = F.normalize(target, dim=1, eps=self.epsilon)

        cosine = (pred_unit * target_unit).sum(dim=1).clamp(-1.0, 1.0)
        orthogonal = pred_unit - target_unit * cosine.unsqueeze(1)
        sine = torch.linalg.vector_norm(orthogonal, dim=1)
        angle = torch.atan2(sine, cosine).clamp(0.0, float(torch.pi))
        return (angle * 180.0 / torch.pi).mean()


@torch.no_grad()
def batch_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = MRAE_EPSILON,
) -> Dict[str, float]:
    """All five reported metrics for one training batch, on-device.

    Definitions match ``hsi_benchmark.metrics`` so train and val numbers are
    directly comparable.
    """
    prediction, target = _pair(prediction.detach(), target.detach())

    error = prediction - target
    abs_error = error.abs()
    mse = error.square().mean()
    rmse = mse.sqrt()
    psnr = (
        torch.full((), 100.0, device=mse.device, dtype=mse.dtype)
        if float(mse) <= 1e-12
        else -10.0 * torch.log10(mse)
    )
    mrae = (abs_error / target.abs().clamp_min(epsilon)).mean()

    return {
        "mrae": float(mrae),
        "rmse": float(rmse),
        "psnr": float(psnr),
        "sam": float(Loss_SAM(epsilon)(prediction, target)),
        "ssim": float(Loss_SSIM()(prediction, target)),
        "mae": float(abs_error.mean()),
    }


def build_criterion(name: str, epsilon: float = MRAE_EPSILON) -> nn.Module:
    """Training objective.

    MRAE is the default and the only one comparable to the ARAD-1K leaderboard --
    MST++ and every number in the NTIRE table optimize it. L1 is offered because
    some reconstruction papers report it, but a run trained on L1 is NOT
    loss-comparable to MST++ even if its MRAE is reported.
    """
    key = str(name).strip().lower()
    if key == "mrae":
        return Loss_MRAE(epsilon)
    if key in {"l1", "mae"}:
        return nn.L1Loss()
    if key in {"l2", "mse"}:
        return nn.MSELoss()
    raise ValueError(f"Unknown loss {name!r}; expected 'mrae', 'l1' or 'l2'.")
