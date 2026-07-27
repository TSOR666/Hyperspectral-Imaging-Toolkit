from __future__ import annotations

from typing import Literal

import torch
from torch.nn import functional as F

MRAEDenominator = Literal["clamp_abs", "source_additive"]


def mean_relative_absolute_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-6,
    denominator: MRAEDenominator = "clamp_abs",
) -> torch.Tensor:
    _check_shapes(prediction, target)
    if denominator == "clamp_abs":
        divisor = target.abs().clamp_min(eps)
    elif denominator == "source_additive":
        # The source SSTrans evaluator adds 1e-5 to both prediction and
        # target whenever a scored tensor contains a zero. The numerator is
        # unchanged, so this is equivalent to dividing by target + epsilon.
        divisor = target if bool(torch.all(target != 0)) else target + eps
    else:
        raise ValueError(f"Unknown MRAE denominator: {denominator}")
    return ((prediction - target).abs() / divisor).mean()


def root_mean_squared_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    _check_shapes(prediction, target)
    return torch.sqrt(F.mse_loss(prediction, target))


def peak_signal_to_noise_ratio(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Mean per-image PSNR."""
    _check_shapes(prediction, target)
    if prediction.ndim < 2:
        raise ValueError("PSNR expects a batch dimension.")
    mse = (prediction - target).square().flatten(1).mean(dim=1)
    peak = prediction.new_tensor(data_range).square()
    return (10.0 * torch.log10(peak / mse.clamp_min(eps))).mean()


def spectral_angle_mapper(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    channel_dim: int = 1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean per-pixel spectral angle in radians."""
    _check_shapes(prediction, target)
    cosine = F.cosine_similarity(
        prediction,
        target,
        dim=channel_dim,
        eps=eps,
    )
    return torch.acos(cosine.clamp(-1.0, 1.0)).mean()


def spectral_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float = 1.0,
    eps: float = 1e-6,
    mrae_denominator: MRAEDenominator = "clamp_abs",
) -> dict[str, torch.Tensor]:
    return {
        "mrae": mean_relative_absolute_error(
            prediction,
            target,
            eps=eps,
            denominator=mrae_denominator,
        ),
        "rmse": root_mean_squared_error(prediction, target),
        "psnr": peak_signal_to_noise_ratio(
            prediction,
            target,
            data_range=data_range,
        ),
        "sam": spectral_angle_mapper(
            prediction,
            target,
            eps=eps,
        ),
    }


def _check_shapes(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> None:
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction shape {prediction.shape} does not match target "
            f"shape {target.shape}."
        )
