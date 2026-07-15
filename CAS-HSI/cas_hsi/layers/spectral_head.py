"""Spectral prediction heads (specification section 7).

The prediction is ``Y = P_rgb(RGB) + dY(features)``:

* ``P_rgb`` is a 1x1 conv 3 -> 31, a learned scene-independent *linear* colour-to-
  spectrum baseline.  It is the part of the mapping that does not need context.
* ``dY`` is the deep residual: metamer disambiguation, material-dependent
  corrections, spatial context, high-frequency detail.

The residual head is initialized near zero so the network starts equal to its learned
linear branch. Xavier initialization makes that branch well-scaled for optimization,
but it is not a calibrated physical RGB-to-spectrum mapping before training.

No activation is applied after the 31-band projection (spec 7.4).  Clamping is an
*evaluation* concern only: clamp() has zero gradient outside its range, so
clamping before the loss would silently freeze learning on every out-of-range
element.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["RGBPrior", "ResidualSpectralHead", "LowRankSpectralHead", "build_spectral_head"]


class RGBPrior(nn.Module):
    """Learned linear RGB -> HSI prior: ``Conv1x1(3 -> bands)`` with bias."""

    def __init__(self, input_channels: int = 3, output_bands: int = 31) -> None:
        super().__init__()
        self.projection = nn.Conv2d(
            int(input_channels),
            int(output_bands),
            kernel_size=1,
            bias=True,
        )
        # Xavier gives the trainable linear branch a well-scaled gain. It is not a
        # calibrated spectral response; physical calibration requires the camera's
        # sensitivity functions and a dataset-specific fitting procedure.
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.projection(rgb)


class ResidualSpectralHead(nn.Module):
    """Default deep head: ``Conv3x3(C -> bands)`` initialized near zero."""

    def __init__(
        self,
        feature_channels: int,
        output_bands: int = 31,
        init_std: float = 1e-3,
    ) -> None:
        super().__init__()
        self.projection = nn.Conv2d(
            int(feature_channels),
            int(output_bands),
            kernel_size=3,
            padding=1,
        )
        nn.init.normal_(self.projection.weight, mean=0.0, std=float(init_std))
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features)


class LowRankSpectralHead(nn.Module):
    """Ablation head (spec 7.5): learned spectral basis + scaled dense residual.

    ``Y = B A + eta * dY`` where ``A`` are per-pixel coefficients over a learned
    ``31 x K`` basis.  The low-rank term encodes the fact that natural reflectance
    spectra live near a low-dimensional manifold; the residual term restores what
    the basis cannot express.

    ``basis_weight()`` exposes ``B`` so a caller can add the optional second-order
    smoothness penalty of spec 7.6 *to the loss* -- smoothness is deliberately not
    hard-coded into the forward pass.
    """

    def __init__(
        self,
        feature_channels: int,
        output_bands: int = 31,
        rank: int = 10,
        residual_scale: float = 0.1,
        init_std: float = 1e-3,
    ) -> None:
        super().__init__()
        if not 1 <= int(rank) <= int(output_bands):
            raise ValueError(
                f"rank must be in [1, output_bands={output_bands}], got {rank}"
            )

        self.rank = int(rank)
        self.output_bands = int(output_bands)

        self.coefficient_head = nn.Conv2d(
            int(feature_channels), self.rank, kernel_size=3, padding=1
        )
        self.basis_projection = nn.Conv2d(
            self.rank, self.output_bands, kernel_size=1, bias=False
        )
        self.residual_head = nn.Conv2d(
            int(feature_channels), self.output_bands, kernel_size=3, padding=1
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))

        nn.init.normal_(self.coefficient_head.weight, mean=0.0, std=float(init_std))
        if self.coefficient_head.bias is not None:
            nn.init.zeros_(self.coefficient_head.bias)
        nn.init.xavier_uniform_(self.basis_projection.weight)
        nn.init.normal_(self.residual_head.weight, mean=0.0, std=float(init_std))
        if self.residual_head.bias is not None:
            nn.init.zeros_(self.residual_head.bias)

    def basis_weight(self) -> torch.Tensor:
        """The learned basis ``B`` as a ``(bands, rank)`` matrix."""
        return self.basis_projection.weight.reshape(self.output_bands, self.rank)

    def spectral_smoothness_penalty(self) -> torch.Tensor:
        """``|| D2_lambda B ||_1`` -- the optional regularizer of spec 7.6."""
        basis = self.basis_weight()
        if basis.shape[0] < 3:
            return basis.new_zeros(())
        second_difference = basis[2:] - 2.0 * basis[1:-1] + basis[:-2]
        return second_difference.abs().mean()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        coefficients = self.coefficient_head(features)
        basis_prediction = self.basis_projection(coefficients)
        residual = self.residual_head(features)
        return basis_prediction + self.residual_scale.to(residual.dtype) * residual


def build_spectral_head(
    name: str,
    feature_channels: int,
    output_bands: int = 31,
    rank: int = 10,
    residual_scale: float = 0.1,
    init_std: float = 1e-3,
) -> nn.Module:
    """Factory for the deep (residual) branch of the spectral prediction."""
    key = str(name).strip().lower()
    if key in {"residual", "conv", "default"}:
        return ResidualSpectralHead(
            feature_channels, output_bands=output_bands, init_std=init_std
        )
    if key in {"low_rank", "lowrank", "basis"}:
        return LowRankSpectralHead(
            feature_channels,
            output_bands=output_bands,
            rank=rank,
            residual_scale=residual_scale,
            init_std=init_std,
        )
    raise ValueError(f"Unknown spectral head {name!r}; expected 'residual' or 'low_rank'.")
