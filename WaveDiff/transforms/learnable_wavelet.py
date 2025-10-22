import torch
import torch.nn as nn


def _haar_filters():
    """Return canonical 2x2 Haar filters ordered as LL, LH, HL, HH."""
    return torch.tensor(
        [
            [[0.5, 0.5], [0.5, 0.5]],    # LL
            [[0.5, 0.5], [-0.5, -0.5]],  # LH
            [[0.5, -0.5], [0.5, -0.5]],  # HL
            [[0.5, -0.5], [-0.5, 0.5]],  # HH
        ],
        dtype=torch.float32,
    )


class LearnableWaveletTransform(nn.Module):
    """
    Learnable wavelet transform implemented as grouped convolutions.

    The layer is initialised with Haar filters but allows gradients to adapt
    the filters during training for data-driven decompositions.
    """

    def __init__(self, in_channels, kernel_size=2, trainable=True):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        self.proj = nn.Conv2d(
            in_channels,
            in_channels * 4,
            kernel_size=kernel_size,
            stride=2,
            padding=0,
            groups=in_channels,
            bias=False,
        )

        self.reset_parameters()
        self.proj.weight.requires_grad = trainable

    def reset_parameters(self):
        """Initialise filters with Haar wavelet weights."""
        base_filters = _haar_filters().unsqueeze(1)  # [4, 1, 2, 2]
        weight = base_filters.repeat(self.in_channels, 1, 1, 1)
        with torch.no_grad():
            self.proj.weight.copy_(weight)

    def forward(self, x):
        coeffs = self.proj(x)
        b, _, h, w = coeffs.shape
        return coeffs.view(b, self.in_channels, 4, h, w)


class LearnableInverseWaveletTransform(nn.Module):
    """
    Learnable inverse wavelet transform implemented with transposed convolutions.

    The layer is initialised to be the exact inverse of the Haar transform, but
    can be fine-tuned jointly with the forward transform for improved fidelity.
    """

    def __init__(self, channels, kernel_size=2, trainable=True):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size

        self.reconstruct = nn.ConvTranspose2d(
            channels * 4,
            channels,
            kernel_size=kernel_size,
            stride=2,
            padding=0,
            groups=channels,
            bias=False,
        )

        self.reset_parameters()
        self.reconstruct.weight.requires_grad = trainable

    def reset_parameters(self):
        """Initialise filters with the inverse Haar basis."""
        base_filters = _haar_filters().unsqueeze(1)  # [4, 1, 2, 2]
        weight = base_filters.repeat(self.channels, 1, 1, 1)
        with torch.no_grad():
            self.reconstruct.weight.copy_(weight)

    def forward(self, coeffs):
        b, c, k, h, w = coeffs.shape
        coeffs = coeffs.view(b, c * k, h, w)
        return self.reconstruct(coeffs)
