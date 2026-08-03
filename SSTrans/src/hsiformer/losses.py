from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


MRAEDenominatorMode = Literal["clamp_abs", "source_additive"]
SAMMode = Literal["stable", "source"]
DeltaEMode = Literal["stable", "source"]


class MRAELoss(nn.Module):
    """Mean relative absolute error with a stable denominator."""

    def __init__(
        self,
        eps: float | None = None,
        *,
        denominator: MRAEDenominatorMode = "clamp_abs",
    ) -> None:
        super().__init__()
        if denominator not in {"clamp_abs", "source_additive"}:
            raise ValueError(
                "MRAE denominator mode must be 'clamp_abs' or "
                f"'source_additive', got {denominator!r}"
            )
        if eps is None:
            eps = 1e-5 if denominator == "source_additive" else 1e-6
        if eps <= 0:
            raise ValueError(f"MRAE epsilon must be positive, got {eps}")
        self.eps = float(eps)
        self.denominator = denominator

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.denominator == "source_additive":
            denominator = (
                target
                if bool(torch.all(target != 0))
                else target + self.eps
            )
        else:
            denominator = target.abs().clamp_min(self.eps)
        return ((prediction - target).abs() / denominator).mean()


class SAMLoss(nn.Module):
    """Mean spectral angle mapper loss in radians."""

    def __init__(
        self,
        eps: float = 1e-8,
        *,
        mode: SAMMode = "stable",
    ) -> None:
        super().__init__()
        if mode not in {"stable", "source"}:
            raise ValueError(f"Unknown SAM mode: {mode}")
        self.eps = eps
        self.mode = mode

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = prediction.flatten(2).transpose(1, 2)
        target = target.flatten(2).transpose(1, 2)
        if self.mode == "source":
            numerator = (prediction * target).sum(dim=-1)
            # Intentional parity with the published implementation: it divides
            # directly by the norm product, including its zero/NaN behavior.
            denominator = prediction.norm(dim=-1) * target.norm(dim=-1)
            cosine = numerator / denominator
            return torch.acos(cosine.clamp(-1.0, 1.0)).mean()
        cosine = F.cosine_similarity(prediction, target, dim=-1, eps=self.eps)
        return torch.acos(
            cosine.clamp(-1.0 + self.eps, 1.0 - self.eps)
        ).mean()


# RGB_Camera_QE.csv sampled at 400:10:700 nm. The R and B columns include
# the upstream camera's analogue channel gains (2.2933984, 1, 1.62308182).
_SOURCE_RGB_RESPONSE = (
    (0.15029723284578586, 0.12434685980195616, 0.4784533968196803),
    (0.11212308917987727, 0.11173671099171585, 0.6339493406067507),
    (0.0813341649604948, 0.09727830520840855, 0.6983374116478445),
    (0.06234998083508065, 0.08081584786110915, 0.7737358358363743),
    (0.05203064308787022, 0.10016763185034031, 0.8402584138463991),
    (0.03969131783365103, 0.100905099838001, 0.848494037948465),
    (0.03279154260417424, 0.10166361949218541, 0.8383005270363986),
    (0.03914199703126622, 0.20833059348845856, 0.837821506129702),
    (0.04912669607924436, 0.40721358087883514, 0.7738178108322628),
    (0.04639789109758555, 0.5302973660074621, 0.6752511363108258),
    (0.04764758863647775, 0.5637148651950068, 0.5280440179026482),
    (0.0588105197340916, 0.5998190422659071, 0.3862022898406276),
    (0.09127681591424987, 0.6046750535604876, 0.2548824561894333),
    (0.11334106962023702, 0.6031092202900223, 0.17638635101627634),
    (0.07344523668176263, 0.583566454518704, 0.13498409794037997),
    (0.04421275110248845, 0.5579011270495311, 0.09221754304898398),
    (0.04106656072740637, 0.5342747278636683, 0.057122517651076114),
    (0.05922374546247801, 0.49741243920811923, 0.039709281570231694),
    (0.42734710022076805, 0.46406257904953163, 0.032076517766917716),
    (0.9462817137954608, 0.41643809173282637, 0.026198012817290305),
    (1.048692528765117, 0.3489330158900801, 0.02098614149137024),
    (1.0370983432129837, 0.27663306682621114, 0.017333640264954944),
    (1.0377662737037983, 0.21996050547376947, 0.0170245879483164),
    (0.9997264473293894, 0.17815657889975792, 0.01924103621857437),
    (0.9379943094515231, 0.1496140373937456, 0.024672981557455915),
    (0.7132744608397654, 0.10630337368880272, 0.026383161546908947),
    (0.1959677831478706, 0.028700541417776306, 0.009722542107527338),
    (0.05987816690222926, 0.010075915803722328, 0.0037308619585006545),
    (0.022122415094429127, 0.004738768744570687, 0.0015841941658463518),
    (0.007358296485027827, 0.0019370727659804223, 0.0005694419059162488),
    (0.004094901956576675, 0.0011485723806567, 0.0003142888302681546),
)


def _per_sample_minmax(image: torch.Tensor, eps: float) -> torch.Tensor:
    flat = image.flatten(1)
    minimum = flat.amin(dim=1).view(-1, 1, 1, 1)
    maximum = flat.amax(dim=1).view(-1, 1, 1, 1)
    value_range = maximum - minimum
    normalized = (image - minimum) / value_range.clamp_min(eps)
    return torch.where(
        value_range > eps, normalized, torch.zeros_like(normalized)
    )


def _rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """Convert normalized sRGB to conventional CIE Lab (D65)."""

    linear = torch.where(
        rgb > 0.0405,
        ((rgb + 0.055) / 1.055).pow(2.4),
        rgb / 12.92,
    )
    matrix = rgb.new_tensor(
        (
            (0.4124, 0.3576, 0.1805),
            (0.2126, 0.7152, 0.0722),
            (0.0193, 0.1192, 0.9504),
        )
    )
    xyz = torch.einsum("ij,bjhw->bihw", matrix, linear)
    white = rgb.new_tensor((0.950489, 1.0, 1.08884)).view(1, 3, 1, 1)
    normalized_xyz = xyz / white
    lab_f = torch.where(
        normalized_xyz > 0.008856,
        normalized_xyz.clamp_min(0.008856).pow(1.0 / 3.0),
        7.787 * normalized_xyz + 16.0 / 116.0,
    )
    x, y, z = lab_f.unbind(dim=1)
    return torch.stack(
        (116.0 * y - 16.0, 500.0 * (x - y), 200.0 * (y - z)),
        dim=1,
    )


def _source_xyz_lab(value: torch.Tensor) -> torch.Tensor:
    """Reproduce SSTrans' zero-masked XYZ transfer function."""

    is_zero = value == 0
    safe_value = value + 0.0001 * is_zero.to(value.dtype)
    transformed = torch.where(
        safe_value > 0.008856,
        safe_value.clamp_min(0.008856).pow(1.0 / 3.0),
        7.787 * safe_value + 16.0 / 116.0,
    )
    return transformed * (~is_zero).to(value.dtype)


def _rgb_to_lab_source(rgb: torch.Tensor) -> torch.Tensor:
    """Match the published SSTrans RGB-to-Lab implementation exactly."""

    above_threshold = (rgb > 0.0405).to(rgb.dtype)
    linear = above_threshold * (((rgb + 0.055) / 1.055).pow(2.4))
    linear = linear + (1.0 - above_threshold) * (rgb / 12.92)
    linear = 100.0 * linear
    # Upstream constructs this as float32 and only then casts with type_as,
    # so preserve the rounded float32 coefficients even for float64 inputs.
    matrix = torch.tensor(
        (
            (0.4124, 0.3576, 0.1805),
            (0.2126, 0.7152, 0.0722),
            (0.0193, 0.1192, 0.9504),
        ),
        dtype=torch.float32,
        device=rgb.device,
    ).to(dtype=rgb.dtype)
    batch, _, height, width = rgb.shape
    xyz = torch.matmul(
        matrix,
        linear.permute(1, 0, 2, 3).contiguous().view(3, -1),
    )
    xyz = xyz.view(3, batch, height, width).permute(1, 0, 2, 3)
    x = _source_xyz_lab(xyz[:, 0] / 95.0489)
    y = _source_xyz_lab(xyz[:, 1] / 100.0)
    z = _source_xyz_lab(xyz[:, 2] / 108.8840)
    return torch.stack(
        (116.0 * y - 16.0, 500.0 * (x - y), 200.0 * (y - z)),
        dim=1,
    )


def _stable_chroma(a: torch.Tensor, b: torch.Tensor, eps: float) -> torch.Tensor:
    square = a.square() + b.square()
    return _stable_sqrt(square, eps * eps)


def _stable_sqrt(value: torch.Tensor, threshold: float) -> torch.Tensor:
    return torch.where(
        value > threshold,
        value.clamp_min(threshold).sqrt(),
        torch.zeros_like(value),
    )


def _ciede2000(lab1: torch.Tensor, lab2: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return the pixelwise CIEDE2000 distance between two BCHW Lab tensors."""

    l1, a1, b1 = lab1.unbind(dim=1)
    l2, a2, b2 = lab2.unbind(dim=1)

    c1 = _stable_chroma(a1, b1, eps)
    c2 = _stable_chroma(a2, b2, eps)
    c_bar = 0.5 * (c1 + c2)
    c_bar_7 = c_bar.pow(7)
    g = 0.5 * (
        1.0 - _stable_sqrt(c_bar_7 / (c_bar_7 + 25.0**7), eps)
    )
    a1_prime = (1.0 + g) * a1
    a2_prime = (1.0 + g) * a2
    c1_prime = _stable_chroma(a1_prime, b1, eps)
    c2_prime = _stable_chroma(a2_prime, b2, eps)

    active1 = c1_prime > eps
    active2 = c2_prime > eps
    h1_prime = torch.atan2(
        torch.where(active1, b1, torch.zeros_like(b1)),
        torch.where(active1, a1_prime, torch.ones_like(a1_prime)),
    ).remainder(2.0 * math.pi)
    h2_prime = torch.atan2(
        torch.where(active2, b2, torch.zeros_like(b2)),
        torch.where(active2, a2_prime, torch.ones_like(a2_prime)),
    ).remainder(2.0 * math.pi)

    delta_l_prime = l2 - l1
    delta_c_prime = c2_prime - c1_prime
    delta_h_prime = h2_prime - h1_prime
    chromatic_pair = active1 & active2
    delta_h_prime = torch.where(
        chromatic_pair & (delta_h_prime > math.pi),
        delta_h_prime - 2.0 * math.pi,
        delta_h_prime,
    )
    delta_h_prime = torch.where(
        chromatic_pair & (delta_h_prime < -math.pi),
        delta_h_prime + 2.0 * math.pi,
        delta_h_prime,
    )
    delta_h_prime = torch.where(
        chromatic_pair, delta_h_prime, torch.zeros_like(delta_h_prime)
    )
    delta_big_h_prime = (
        2.0
        * _stable_sqrt((c1_prime * c2_prime).clamp_min(0), eps * eps)
        * torch.sin(0.5 * delta_h_prime)
    )

    l_bar_prime = 0.5 * (l1 + l2)
    c_bar_prime = 0.5 * (c1_prime + c2_prime)
    hue_sum = h1_prime + h2_prime
    hue_difference = (h1_prime - h2_prime).abs()
    h_bar_prime = torch.where(
        ~chromatic_pair,
        hue_sum,
        torch.where(
            hue_difference <= math.pi,
            0.5 * hue_sum,
            torch.where(
                hue_sum < 2.0 * math.pi,
                0.5 * (hue_sum + 2.0 * math.pi),
                0.5 * (hue_sum - 2.0 * math.pi),
            ),
        ),
    )

    t = (
        1.0
        - 0.17 * torch.cos(h_bar_prime - math.radians(30.0))
        + 0.24 * torch.cos(2.0 * h_bar_prime)
        + 0.32 * torch.cos(3.0 * h_bar_prime + math.radians(6.0))
        - 0.20 * torch.cos(4.0 * h_bar_prime - math.radians(63.0))
    )
    delta_theta = math.radians(30.0) * torch.exp(
        -((torch.rad2deg(h_bar_prime) - 275.0) / 25.0).square()
    )
    c_bar_prime_7 = c_bar_prime.pow(7)
    r_c = 2.0 * _stable_sqrt(
        (c_bar_prime_7 / (c_bar_prime_7 + 25.0**7)).clamp_min(0),
        eps,
    )
    s_l = 1.0 + (
        0.015 * (l_bar_prime - 50.0).square()
        / (20.0 + (l_bar_prime - 50.0).square()).sqrt()
    )
    s_c = 1.0 + 0.045 * c_bar_prime
    s_h = 1.0 + 0.015 * c_bar_prime * t
    r_t = -r_c * torch.sin(2.0 * delta_theta)

    lightness = delta_l_prime / s_l
    chroma = delta_c_prime / s_c
    hue = delta_big_h_prime / s_h
    squared_distance = (
        lightness.square()
        + chroma.square()
        + hue.square()
        + r_t * chroma * hue
    ).clamp_min(0)
    # Subtracting sqrt(eps) makes equal colors exactly zero while avoiding the
    # undefined derivative of sqrt at the origin.
    return (squared_distance + eps).sqrt() - math.sqrt(eps)


def _source_hue_degrees(b: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    is_zero = (b == 0) & (a == 0)
    active = (~is_zero).to(b.dtype)
    hue = torch.rad2deg(torch.atan2(b * active, a * active))
    return torch.where(hue >= 0, hue, 360.0 + hue)


def _source_delta_hue(
    c1: torch.Tensor,
    c2: torch.Tensor,
    h1: torch.Tensor,
    h2: torch.Tensor,
) -> torch.Tensor:
    has_zero_chroma = c1 * c2 == 0
    active = (~has_zero_chroma).to(c1.dtype)
    difference = h2 - h1
    result1 = difference * active * (difference.abs() <= 180).to(c1.dtype)
    result2 = (
        (difference - 360.0)
        * (difference > 180).to(c1.dtype)
        * active
    )
    result3 = (
        (difference + 360.0)
        * (difference < -180).to(c1.dtype)
        * active
    )
    return result1 + result2 + result3


def _source_average_hue(
    c1: torch.Tensor,
    c2: torch.Tensor,
    h1: torch.Tensor,
    h2: torch.Tensor,
) -> torch.Tensor:
    has_zero_chroma = c1 * c2 == 0
    active = (~has_zero_chroma).to(c1.dtype)
    close = ((h2 - h1).abs() <= 180).to(c1.dtype)
    far = 1.0 - close
    below_turn = ((h2 + h1).abs() < 360).to(c1.dtype)
    above_turn = 1.0 - below_turn
    result1 = (h1 + h2) * active * close
    result2 = (h1 + h2 + 360.0) * active * far * below_turn
    result3 = (h1 + h2 - 360.0) * active * far * above_turn
    result = result1 + result2 + result3
    # This redundant-looking term is present in the source implementation.
    return 0.5 * (
        result + result * has_zero_chroma.to(result.dtype)
    )


def _ciede2000_source(
    lab1: torch.Tensor,
    lab2: torch.Tensor,
) -> torch.Tensor:
    """Literal differentiable CIEDE2000 variant used by published SSTrans."""

    l1, a1, b1 = lab1.unbind(dim=1)
    l2, a2, b2 = lab2.unbind(dim=1)
    achromatic1 = (a1 == 0) & (b1 == 0)
    achromatic2 = (a2 == 0) & (b2 == 0)
    active1 = (~achromatic1).to(lab1.dtype)
    active2 = (~achromatic2).to(lab2.dtype)
    # The upstream routine injects 1e-4 into B for exactly achromatic colors,
    # then later masks all chroma and hue contributions if either color was
    # achromatic.
    b1_adjusted = b1 + 0.0001 * achromatic1.to(lab1.dtype)
    b2_adjusted = b2 + 0.0001 * achromatic2.to(lab2.dtype)

    c1 = (a1.square() + b1_adjusted.square()).sqrt()
    c2 = (a2.square() + b2_adjusted.square()).sqrt()
    average_c = 0.5 * (c1 + c2)
    average_c_7 = average_c.pow(7)
    g = 0.5 * (
        1.0
        - (average_c_7 / (average_c_7 + 25.0**7)).sqrt()
    )
    a1_prime = (1.0 + g) * a1
    a2_prime = (1.0 + g) * a2
    c1_prime = (a1_prime.square() + b1_adjusted.square()).sqrt()
    c2_prime = (a2_prime.square() + b2_adjusted.square()).sqrt()

    h1_prime = _source_hue_degrees(b1_adjusted, a1_prime) * active1
    h2_prime = _source_hue_degrees(b2_adjusted, a2_prime) * active2
    delta_l_prime = l2 - l1
    delta_c_prime = c2_prime - c1_prime
    delta_h_prime = _source_delta_hue(c1, c2, h1_prime, h2_prime)
    delta_big_h_prime = (
        2.0
        * (c1_prime * c2_prime).sqrt()
        * torch.sin(torch.deg2rad(delta_h_prime) / 2.0)
    )
    chromatic_pair = 1.0 - torch.maximum(
        achromatic1.to(lab1.dtype),
        achromatic2.to(lab2.dtype),
    )
    delta_big_h_prime = delta_big_h_prime * chromatic_pair

    average_l = 0.5 * (l1 + l2)
    average_c_prime = 0.5 * (c1_prime + c2_prime)
    average_h_prime = _source_average_hue(
        c1, c2, h1_prime, h2_prime
    )
    # 39 degrees is intentional. It is the most visible deviation from the
    # standard CIEDE2000 equation's 30-degree term.
    t = (
        1.0
        - 0.17 * torch.cos(torch.deg2rad(average_h_prime - 39.0))
        + 0.24 * torch.cos(torch.deg2rad(2.0 * average_h_prime))
        + 0.32
        * torch.cos(torch.deg2rad(3.0 * average_h_prime + 6.0))
        - 0.20
        * torch.cos(torch.deg2rad(4.0 * average_h_prime - 63.0))
    )
    delta_rotation = 30.0 * torch.exp(
        -((average_h_prime - 275.0) / 25.0).square()
    )
    average_c_prime_7 = average_c_prime.pow(7)
    r_c = (
        average_c_prime_7 / (average_c_prime_7 + 25.0**7)
    ).sqrt()
    s_l = 1.0 + (
        0.015 * (average_l - 50.0).square()
        / (20.0 + (average_l - 50.0).square()).sqrt()
    )
    s_c = 1.0 + 0.045 * average_c_prime
    s_h = 1.0 + 0.015 * average_c_prime * t
    r_t = -2.0 * r_c * torch.sin(torch.deg2rad(2.0 * delta_rotation))

    lightness = delta_l_prime / s_l
    chroma = delta_c_prime / s_c
    hue = delta_big_h_prime / s_h
    squared_distance = (
        lightness.square()
        + chroma.square() * chromatic_pair
        + hue.square() * chromatic_pair
        + r_t * chroma * hue * chromatic_pair
    )
    is_zero = squared_distance <= 0
    safe_distance = squared_distance + 0.0001 * is_zero.to(lab1.dtype)
    return safe_distance.sqrt() * (~is_zero).to(lab1.dtype)


class DeltaE2000Loss(nn.Module):
    """Color loss from 31-band HSI via the fixed SSTrans RGB response."""

    def __init__(
        self,
        eps: float = 1e-12,
        *,
        mode: DeltaEMode = "stable",
    ) -> None:
        super().__init__()
        if eps <= 0:
            raise ValueError(f"DeltaE epsilon must be positive, got {eps}")
        if mode not in {"stable", "source"}:
            raise ValueError(f"Unknown DeltaE mode: {mode}")
        self.eps = float(eps)
        self.mode = mode
        response = torch.tensor(_SOURCE_RGB_RESPONSE, dtype=torch.float32)
        self.register_buffer(
            "rgb_response",
            response.transpose(0, 1).unsqueeze(-1).unsqueeze(-1),
        )

    def _hsi_to_normalized_rgb(self, hsi: torch.Tensor) -> torch.Tensor:
        if hsi.ndim != 4 or hsi.shape[1] != 31:
            raise ValueError(
                "DeltaE2000Loss expects BCHW tensors with 31 spectral bands, "
                f"got shape {tuple(hsi.shape)}"
            )
        working_dtype = (
            torch.float32
            if hsi.dtype in {torch.float16, torch.bfloat16}
            else hsi.dtype
        )
        hsi = hsi.to(dtype=working_dtype)
        response = self.rgb_response.to(device=hsi.device, dtype=working_dtype)
        rgb = F.conv2d(hsi, response)
        return _per_sample_minmax(rgb, self.eps)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                "DeltaE2000Loss prediction and target must have the same shape, "
                f"got {tuple(prediction.shape)} and {tuple(target.shape)}"
            )
        prediction_rgb = self._hsi_to_normalized_rgb(prediction)
        target_rgb = self._hsi_to_normalized_rgb(target)
        if self.mode == "source":
            prediction_lab = _rgb_to_lab_source(prediction_rgb)
            target_lab = _rgb_to_lab_source(target_rgb)
            return _ciede2000_source(target_lab, prediction_lab).mean()
        prediction_lab = _rgb_to_lab(prediction_rgb)
        target_lab = _rgb_to_lab(target_rgb)
        return _ciede2000(target_lab, prediction_lab, self.eps).mean()


class SpectralReconstructionLoss(nn.Module):
    """Configurable objective for retraining experiments."""

    def __init__(
        self,
        *,
        l1_weight: float = 1.0,
        mrae_weight: float = 0.0,
        sam_weight: float = 0.0,
        delta_e_weight: float = 0.0,
        mrae_denominator: MRAEDenominatorMode = "clamp_abs",
        mrae_epsilon: float | None = None,
        sam_mode: SAMMode = "stable",
        delta_e_mode: DeltaEMode | None = None,
    ) -> None:
        super().__init__()
        self.l1_weight = l1_weight
        self.mrae_weight = mrae_weight
        self.sam_weight = sam_weight
        self.delta_e_weight = delta_e_weight
        self.mrae = MRAELoss(
            eps=mrae_epsilon, denominator=mrae_denominator
        )
        self.sam = SAMLoss(mode=sam_mode)
        if delta_e_mode is None:
            delta_e_mode = (
                "source"
                if (
                    sam_mode == "source"
                    or mrae_denominator == "source_additive"
                )
                else "stable"
            )
        self.delta_e = DeltaE2000Loss(mode=delta_e_mode)

    def compute(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return the weighted total alongside each *unweighted* term.

        The terms are reported unweighted so that a logged value can be compared
        directly against the matching validation metric. Without this, the total
        is uninterpretable: DeltaE2000 is expressed in CIE units of order 1-10,
        so a nominal 0.1 weight can still make it the largest contributor, and
        the training MRAE cannot be recovered from the total.
        """
        terms: dict[str, torch.Tensor] = {}
        loss = prediction.new_zeros(())
        if self.l1_weight:
            terms["l1"] = F.l1_loss(prediction, target)
            loss = loss + self.l1_weight * terms["l1"]
        if self.mrae_weight:
            terms["mrae"] = self.mrae(prediction, target)
            loss = loss + self.mrae_weight * terms["mrae"]
        if self.sam_weight:
            terms["sam"] = self.sam(prediction, target)
            loss = loss + self.sam_weight * terms["sam"]
        if self.delta_e_weight:
            terms["delta_e"] = self.delta_e(prediction, target)
            loss = loss + self.delta_e_weight * terms["delta_e"]
        return loss, terms

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.compute(prediction, target)[0]
