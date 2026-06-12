from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MRAELoss(nn.Module):
    """Mean relative absolute error with a stable denominator."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        denominator = target.abs().clamp_min(self.eps)
        return ((prediction - target).abs() / denominator).mean()


class SAMLoss(nn.Module):
    """Mean spectral angle mapper loss in radians."""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = prediction.flatten(2).transpose(1, 2)
        target = target.flatten(2).transpose(1, 2)
        cosine = F.cosine_similarity(prediction, target, dim=-1, eps=self.eps)
        return torch.acos(cosine.clamp(-1.0 + self.eps, 1.0 - self.eps)).mean()


class SpectralReconstructionLoss(nn.Module):
    """Configurable objective for retraining experiments."""

    def __init__(
        self,
        *,
        l1_weight: float = 1.0,
        mrae_weight: float = 0.0,
        sam_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.l1_weight = l1_weight
        self.mrae_weight = mrae_weight
        self.sam_weight = sam_weight
        self.mrae = MRAELoss()
        self.sam = SAMLoss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = prediction.new_zeros(())
        if self.l1_weight:
            loss = loss + self.l1_weight * F.l1_loss(prediction, target)
        if self.mrae_weight:
            loss = loss + self.mrae_weight * self.mrae(prediction, target)
        if self.sam_weight:
            loss = loss + self.sam_weight * self.sam(prediction, target)
        return loss

