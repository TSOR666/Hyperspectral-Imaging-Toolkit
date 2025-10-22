# src/hsi_model/utils/data/transforms.py
"""
Data transformation and conversion utilities.
"""

import logging
from typing import Dict, Union, Tuple

import numpy as np
import torch

from ...constants import (
    VALIDATION_CENTER_CROP_START_H,
    VALIDATION_CENTER_CROP_START_W,
    VALIDATION_CENTER_CROP_END_H,
    VALIDATION_CENTER_CROP_END_W,
)

logger = logging.getLogger(__name__)


def mst_to_gan_batch(
    bgr_batch: Union[np.ndarray, torch.Tensor],
    hyper_batch: Union[np.ndarray, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert MST++ batch format to GAN training format.
    """
    if isinstance(bgr_batch, torch.Tensor) and isinstance(
        hyper_batch, torch.Tensor
    ):
        rgb_tensor = bgr_batch.float()
        hsi_tensor = hyper_batch.float()
    else:
        rgb_tensor = torch.from_numpy(bgr_batch).float()
        hsi_tensor = torch.from_numpy(hyper_batch).float()

    return rgb_tensor, hsi_tensor


def compute_mst_center_crop_metrics(
    pred_hsi: torch.Tensor,
    target_hsi: torch.Tensor,
    criterion: torch.nn.Module = None,
) -> Dict[str, float]:
    """
    Compute metrics using the MST++ center-crop evaluation protocol.
    """
    # Import locally to avoid circular dependencies.
    from ...utils.metrics import compute_metrics_arad1k, compute_metrics

    pred_crop = pred_hsi[
        :,
        :,
        VALIDATION_CENTER_CROP_START_H:VALIDATION_CENTER_CROP_END_H,
        VALIDATION_CENTER_CROP_START_W:VALIDATION_CENTER_CROP_END_W,
    ]
    target_crop = target_hsi[
        :,
        :,
        VALIDATION_CENTER_CROP_START_H:VALIDATION_CENTER_CROP_END_H,
        VALIDATION_CENTER_CROP_START_W:VALIDATION_CENTER_CROP_END_W,
    ]

    if pred_crop.numel() == 0 or target_crop.numel() == 0:
        logger.error(
            "Center crop resulted in empty tensor! Input shape: %s, Crop shape: %s",
            pred_hsi.shape,
            pred_crop.shape,
        )
        return {
            "mrae": 999.0,
            "psnr": 0.0,
            "ssim": 0.0,
            "sam": 999.0,
        }

    logger.debug(
        "MST++ center crop: %s -> %s (cropped %spx from each side)",
        pred_hsi.shape,
        pred_crop.shape,
        VALIDATION_CENTER_CROP_START_H,
    )

    try:
        metrics = compute_metrics_arad1k(pred_crop, target_crop)
    except Exception:
        logger.exception("Falling back to generic metrics for MST center crop.")
        metrics = compute_metrics(pred_crop, target_crop, compute_all=True)

    if criterion is not None:
        try:
            loss_val = criterion(pred_crop, target_crop)[0]
            metrics["loss"] = (
                loss_val.item() if torch.is_tensor(loss_val) else float(loss_val)
            )
        except Exception:
            logger.exception("Failed to compute criterion on MST center crop.")

    return metrics


def normalize_batch(
    batch: torch.Tensor,
    mean: Union[float, torch.Tensor],
    std: Union[float, torch.Tensor],
) -> torch.Tensor:
    """
    Normalize a batch of tensors.
    """
    if isinstance(mean, (int, float)):
        mean = torch.tensor(mean, device=batch.device)
    if isinstance(std, (int, float)):
        std = torch.tensor(std, device=batch.device)

    if mean.dim() == 1:
        mean = mean.view(1, -1, 1, 1)
    if std.dim() == 1:
        std = std.view(1, -1, 1, 1)

    return (batch - mean) / std


def denormalize_batch(
    batch: torch.Tensor,
    mean: Union[float, torch.Tensor],
    std: Union[float, torch.Tensor],
) -> torch.Tensor:
    """
    Revert normalization applied by `normalize_batch`.
    """
    if isinstance(mean, (int, float)):
        mean = torch.tensor(mean, device=batch.device)
    if isinstance(std, (int, float)):
        std = torch.tensor(std, device=batch.device)

    if mean.dim() == 1:
        mean = mean.view(1, -1, 1, 1)
    if std.dim() == 1:
        std = std.view(1, -1, 1, 1)

    return batch * std + mean


def resize_batch(
    batch: torch.Tensor,
    target_size: Tuple[int, int],
    mode: str = "bilinear",
    align_corners: bool = False,
) -> torch.Tensor:
    """
    Resize a batch of images.
    """
    return torch.nn.functional.interpolate(
        batch,
        size=target_size,
        mode=mode,
        align_corners=align_corners if mode != "nearest" else None,
    )


def random_crop_batch(
    batch: torch.Tensor,
    crop_size: int,
    num_crops: int = 1,
) -> torch.Tensor:
    """
    Extract random crops from a batch.
    """
    _, _, height, width = batch.shape

    if height < crop_size or width < crop_size:
        raise ValueError(
            f"Crop size {crop_size} larger than image size ({height}, {width})"
        )

    crops = []
    for _ in range(num_crops):
        for b_item in batch:
            top = torch.randint(0, height - crop_size + 1, (1,)).item()
            left = torch.randint(0, width - crop_size + 1, (1,)).item()
            crop = b_item[
                :,
                top : top + crop_size,
                left : left + crop_size,
            ]
            crops.append(crop.unsqueeze(0))

    return torch.cat(crops, dim=0)


def center_crop_batch(
    batch: torch.Tensor,
    crop_size: Union[int, Tuple[int, int]],
) -> torch.Tensor:
    """
    Extract center crops from a batch.
    """
    if isinstance(crop_size, int):
        crop_h, crop_w = crop_size, crop_size
    else:
        crop_h, crop_w = crop_size

    _, _, height, width = batch.shape

    if height < crop_h or width < crop_w:
        raise ValueError(
            f"Crop size ({crop_h}, {crop_w}) larger than image size ({height}, {width})"
        )

    top = (height - crop_h) // 2
    left = (width - crop_w) // 2

    return batch[:, :, top : top + crop_h, left : left + crop_w]
