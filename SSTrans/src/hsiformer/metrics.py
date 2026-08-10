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


def structural_similarity_index(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float = 1.0,
    window_size: int = 11,
) -> torch.Tensor:
    """Mean spectral-band SSIM using the NTIRE/MSWR box-filter convention."""
    _check_shapes(prediction, target)
    if prediction.ndim != 4:
        raise ValueError("SSIM expects BCHW tensors.")
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer.")

    padding = window_size // 2
    prediction = prediction.float()
    target = target.float()
    mu_prediction = F.avg_pool2d(prediction, window_size, 1, padding)
    mu_target = F.avg_pool2d(target, window_size, 1, padding)
    mu_prediction_sq = mu_prediction.square()
    mu_target_sq = mu_target.square()
    mu_product = mu_prediction * mu_target

    variance_prediction = (
        F.avg_pool2d(prediction.square(), window_size, 1, padding)
        - mu_prediction_sq
    )
    variance_target = (
        F.avg_pool2d(target.square(), window_size, 1, padding)
        - mu_target_sq
    )
    covariance = (
        F.avg_pool2d(prediction * target, window_size, 1, padding)
        - mu_product
    )
    epsilon = torch.finfo(prediction.dtype).eps * 10
    variance_prediction = variance_prediction.clamp_min(epsilon)
    variance_target = variance_target.clamp_min(epsilon)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2 * mu_product + c1) * (2 * covariance + c2)
    denominator = (
        (mu_prediction_sq + mu_target_sq + c1)
        * (variance_prediction + variance_target + c2)
    )
    return (numerator / denominator.clamp_min(epsilon)).mean()


def spectral_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float = 1.0,
    eps: float = 1e-6,
    mrae_denominator: MRAEDenominator = "clamp_abs",
    include_ssim: bool = True,
) -> dict[str, torch.Tensor]:
    metrics = {
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
    if include_ssim:
        metrics["ssim"] = structural_similarity_index(
            prediction,
            target,
            data_range=data_range,
        )
    return metrics


def _check_shapes(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> None:
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction shape {prediction.shape} does not match target "
            f"shape {target.shape}."
        )
