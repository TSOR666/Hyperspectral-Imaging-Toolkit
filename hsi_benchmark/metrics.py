from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _as_bchw(value: np.ndarray | torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError(f"Expected CHW or BCHW data, got shape {tuple(tensor.shape)}")
    return tensor


def relative_absolute_error(
    prediction: torch.Tensor, target: torch.Tensor, *, epsilon: float = 0.0
) -> torch.Tensor:
    """FP32 relative errors; epsilon=0 is the positive-target reference policy.

    Exact MRAE is undefined for zero targets. Reject them explicitly rather than
    silently changing the objective, dropping pixels, or replacing infinities.
    """
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("MRAE epsilon must be finite and non-negative")
    if prediction.shape != target.shape or target.numel() == 0:
        raise ValueError("MRAE requires aligned, nonempty prediction and target tensors")
    prediction, target = prediction.float(), target.float()
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise FloatingPointError("MRAE requires finite predictions and targets")
    if epsilon == 0 and (target <= 0).any():
        raise ValueError("Exact MRAE requires strictly positive targets; choose an explicit floor for zeros")
    return (prediction - target).abs() / target.abs().clamp_min(epsilon)


def mrae_breakdown(
    prediction: torch.Tensor, target: torch.Tensor, *, epsilon: float = 0.0
) -> Dict[str, object]:
    """Raw/clamped MRAE, per-scene/band errors, and additive intensity contributions."""
    pred, truth = _as_bchw(prediction), _as_bchw(target)
    error = relative_absolute_error(pred, truth, epsilon=epsilon)
    buckets = []
    limits = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, float("inf"))
    for low, high in zip(limits[:-1], limits[1:]):
        mask = (truth.abs() >= low) & (truth.abs() < high)
        count = int(mask.sum().item())
        selected = error[mask]
        buckets.append({
            "lower": low, "upper": high if np.isfinite(high) else None,
            "count": count, "fraction": count / error.numel(),
            "mean_mrae": float(selected.mean().item()) if count else 0.0,
            "contribution": float(selected.double().sum().item() / error.numel()),
        })
    return {
        "epsilon": epsilon,
        "raw_mrae": float(error.mean().item()),
        "clamped_mrae": float(relative_absolute_error(pred.clamp(0, 1), truth, epsilon=epsilon).mean().item()),
        "out_of_range_fraction": float(((pred < 0) | (pred > 1)).float().mean().item()),
        "per_band_mrae": error.mean(dim=(0, 2, 3)).detach().cpu().tolist(),
        "per_scene_mrae": error.mean(dim=(1, 2, 3)).detach().cpu().tolist(),
        "intensity_buckets": buckets,
    }


def _ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    if min(pred.shape[-2:]) < window_size:
        window_size = max(3, min(pred.shape[-2:]) | 1)
    padding = window_size // 2
    mu_x = F.avg_pool2d(pred, window_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(target, window_size, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(pred.square(), window_size, 1, padding) - mu_x.square()
    sigma_y = F.avg_pool2d(target.square(), window_size, 1, padding) - mu_y.square()
    sigma_xy = F.avg_pool2d(pred * target, window_size, 1, padding) - mu_x * mu_y
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x.square() + mu_y.square() + c1) * (
        sigma_x + sigma_y + c2
    )
    return (numerator / denominator.clamp_min(1e-12)).mean()


def _sam_map(pred: torch.Tensor, target: torch.Tensor, epsilon: float) -> torch.Tensor:
    pred_norm = F.normalize(pred, dim=1, eps=epsilon)
    target_norm = F.normalize(target, dim=1, eps=epsilon)
    cosine = (pred_norm * target_norm).sum(dim=1).clamp(-1.0, 1.0)
    orthogonal = pred_norm - target_norm * cosine.unsqueeze(1)
    sine = torch.linalg.vector_norm(orthogonal, dim=1)
    return torch.atan2(sine, cosine).clamp(0.0, float(torch.pi)) * 180.0 / torch.pi


def compute_hsi_metrics(
    prediction: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    *,
    epsilon: float = 1e-6,
    crop_border: int = 0,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Compute paper-standard full-reference HSI metrics.

    Inputs must be aligned CHW or BCHW reflectance arrays in [0, 1]. SAM is
    reported in degrees. The same optional border crop is applied to both.
    """
    pred = _as_bchw(prediction)
    truth = _as_bchw(target)
    if pred.shape != truth.shape:
        raise ValueError(
            f"Prediction/target mismatch: {tuple(pred.shape)} vs {tuple(truth.shape)}"
        )
    if crop_border:
        height, width = pred.shape[-2:]
        if height <= 2 * crop_border or width <= 2 * crop_border:
            raise ValueError(
                f"crop_border={crop_border} is too large for {height}x{width}"
            )
        pred = pred[..., crop_border:-crop_border, crop_border:-crop_border]
        truth = truth[..., crop_border:-crop_border, crop_border:-crop_border]

    pred = pred.float()
    truth = truth.float()
    error = pred - truth
    abs_error = error.abs()
    mse = error.square().mean()
    rmse = mse.sqrt()
    psnr = torch.tensor(100.0) if mse <= 1e-12 else -10.0 * torch.log10(mse)
    relative_error = relative_absolute_error(pred, truth, epsilon=epsilon)
    mrae = relative_error.mean()
    sam_map = _sam_map(pred, truth, max(epsilon, 1e-12))

    metrics = {
        "mrae": float(mrae),
        "rmse": float(rmse),
        "psnr": float(psnr),
        "sam": float(sam_map.mean()),
        "ssim": float(_ssim(pred, truth)),
        "mae": float(abs_error.mean()),
    }

    band_mse = error.square().mean(dim=(0, 2, 3))
    band_psnr = torch.where(
        band_mse <= 1e-12,
        torch.full_like(band_mse, 100.0),
        -10.0 * torch.log10(band_mse.clamp_min(1e-12)),
    )
    per_band = {
        "mrae": relative_error
        .mean(dim=(0, 2, 3))
        .numpy(),
        "rmse": band_mse.sqrt().numpy(),
        "psnr": band_psnr.numpy(),
        "mae": abs_error.mean(dim=(0, 2, 3)).numpy(),
    }
    maps = {
        "mae": abs_error.mean(dim=1).squeeze(0).numpy(),
        "rmse": error.square().mean(dim=1).sqrt().squeeze(0).numpy(),
        "sam": sam_map.squeeze(0).numpy(),
    }
    return metrics, {**per_band, **{f"map_{k}": v for k, v in maps.items()}}


def bootstrap_confidence_interval(
    values: Iterable[float],
    *,
    confidence: float = 0.95,
    samples: int = 4000,
    seed: int = 0,
) -> Tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan")
    if array.size == 1:
        value = float(array[0])
        return value, value
    rng = np.random.default_rng(seed)
    chunk = min(samples, 512)
    means: List[np.ndarray] = []
    remaining = samples
    while remaining:
        current = min(chunk, remaining)
        indices = rng.integers(0, array.size, size=(current, array.size))
        means.append(array[indices].mean(axis=1))
        remaining -= current
    bootstrap_means = np.concatenate(means)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(bootstrap_means, [alpha, 1.0 - alpha])
    return float(low), float(high)


def summarize_metric_rows(
    rows: Iterable[Mapping[str, float]],
    *,
    bootstrap_samples: int = 4000,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    rows = list(rows)
    metric_names = sorted({key for row in rows for key in row})
    summary: Dict[str, Dict[str, float]] = {}
    for offset, metric in enumerate(metric_names):
        values = np.asarray(
            [float(row[metric]) for row in rows if metric in row], dtype=np.float64
        )
        if values.size == 0:
            continue
        ci_low, ci_high = bootstrap_confidence_interval(
            values, samples=bootstrap_samples, seed=seed + offset
        )
        summary[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "median": float(np.median(values)),
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "count": int(values.size),
        }
    return summary
