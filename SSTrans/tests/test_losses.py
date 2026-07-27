from __future__ import annotations

import pytest
import torch

from hsiformer.losses import (
    DeltaE2000Loss,
    MRAELoss,
    SAMLoss,
    SpectralReconstructionLoss,
    _ciede2000,
    _ciede2000_source,
    _rgb_to_lab_source,
)


def test_mrae_default_preserves_clamped_absolute_denominator() -> None:
    prediction = torch.tensor([[[[0.25, 0.4]]]])
    target = torch.tensor([[[[0.0, -0.2]]]])

    expected = (
        (prediction - target).abs() / target.abs().clamp_min(1e-6)
    ).mean()

    assert MRAELoss()(prediction, target) == pytest.approx(expected.item())


def test_source_mrae_uses_additive_denominator_and_source_epsilon() -> None:
    prediction = torch.tensor([[[[0.1, 0.3]]]], dtype=torch.float64)
    target = torch.tensor([[[[0.0, 0.2]]]], dtype=torch.float64)

    expected = (
        (prediction - target).abs() / (target + 1e-5)
    ).mean()

    actual = MRAELoss(denominator="source_additive")(prediction, target)
    assert actual == pytest.approx(expected.item())


def test_mrae_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="denominator mode"):
        MRAELoss(denominator="unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="epsilon must be positive"):
        MRAELoss(eps=0.0)


def test_source_sam_is_zero_for_identical_nonzero_spectra() -> None:
    spectrum = torch.zeros(1, 31, 1, 1)
    spectrum[:, 0] = 1.0
    assert SAMLoss(mode="source")(spectrum, spectrum).item() == pytest.approx(
        0.0,
        abs=1e-7,
    )


def test_source_sam_matches_unregularized_norm_division() -> None:
    prediction = torch.zeros(1, 31, 1, 1, dtype=torch.float64)
    target = torch.zeros_like(prediction)
    prediction[:, :2, 0, 0] = torch.tensor([3.0, 4.0])
    target[:, :2, 0, 0] = torch.tensor([4.0, 3.0])

    expected = torch.acos(torch.tensor(24.0 / 25.0, dtype=torch.float64))

    assert SAMLoss(mode="source")(prediction, target) == pytest.approx(
        expected.item()
    )


def test_source_sam_preserves_zero_norm_nan_behavior() -> None:
    prediction = torch.zeros(1, 31, 1, 1)
    target = torch.ones_like(prediction)

    assert torch.isnan(SAMLoss(mode="source")(prediction, target))
    assert torch.isfinite(SAMLoss(mode="stable")(prediction, target))


def test_ciede2000_matches_published_reference_pair() -> None:
    lab1 = torch.tensor([50.0, 2.6772, -79.7751]).view(1, 3, 1, 1)
    lab2 = torch.tensor([50.0, 0.0, -82.7485]).view(1, 3, 1, 1)

    delta_e = _ciede2000(lab1, lab2)

    assert delta_e.item() == pytest.approx(2.0425, abs=5e-4)


def test_source_rgb_to_lab_matches_upstream_zero_masking() -> None:
    rgb = torch.tensor(
        [[[[0.0, 0.1]], [[0.0, 0.2]], [[0.0, 0.3]]]],
        dtype=torch.float64,
    )

    lab = _rgb_to_lab_source(rgb)

    assert lab[:, :, 0, 0].flatten().tolist() == pytest.approx(
        [-16.0, 0.0, 0.0],
        abs=1e-12,
    )
    assert lab[:, :, 0, 1].flatten().tolist() == pytest.approx(
        [20.477293000413667, -0.6487942054374529, -18.632550933888915],
        abs=1e-10,
    )


def test_source_ciede2000_matches_upstream_nonstandard_reference_batch() -> None:
    # Reference values were independently generated from the upstream
    # differential_color_functions_no_device.py implementation. They cover
    # the 39-degree hue term, hue wrapping, and its achromatic masking.
    lab1 = torch.tensor(
        [
            [50.0, 2.6772, -79.7751],
            [50.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [50.0, 40.0, 1.0],
            [60.0, 20.0, 30.0],
        ],
        dtype=torch.float64,
    ).view(5, 3, 1, 1)
    lab2 = torch.tensor(
        [
            [50.0, 0.0, -82.7485],
            [50.0, 20.0, 30.0],
            [80.0, 20.0, 30.0],
            [50.0, 40.0, -1.0],
            [40.0, -25.0, 15.0],
        ],
        dtype=torch.float64,
    ).view(5, 3, 1, 1)

    actual = _ciede2000_source(lab1, lab2).flatten()

    assert actual.tolist() == pytest.approx(
        [
            2.021277836361823,
            0.0,
            60.0,
            1.1057710548894575,
            40.840788733508546,
        ],
        abs=1e-10,
    )


def test_source_delta_e_full_hsi_pipeline_matches_upstream_reference() -> None:
    values = torch.arange(31 * 2 * 3, dtype=torch.float32).view(1, 31, 2, 3)
    target = values / (values.numel() - 1)
    prediction = 0.97 * target + 0.02 * torch.sin(0.37 * values)

    delta_e = DeltaE2000Loss(mode="source")(prediction, target)

    assert delta_e.item() == pytest.approx(0.059746790677309036, abs=1e-6)


def test_delta_e_response_is_fixed_buffer_and_equal_inputs_are_zero() -> None:
    criterion = DeltaE2000Loss()
    cube = torch.linspace(0.0, 1.0, 31).view(1, 31, 1, 1).expand(1, 31, 3, 4)

    assert dict(criterion.named_buffers())["rgb_response"].shape == (3, 31, 1, 1)
    assert "rgb_response" not in dict(criterion.named_parameters())
    assert criterion(cube, cube).item() == pytest.approx(0.0, abs=1e-7)


@pytest.mark.parametrize("mode", ["stable", "source"])
def test_delta_e_loss_is_finite_and_differentiable(mode: str) -> None:
    generator = torch.Generator().manual_seed(13)
    target = torch.rand((2, 31, 4, 5), generator=generator)
    prediction = (
        target + 0.02 * torch.randn(target.shape, generator=generator)
    ).requires_grad_()

    loss = DeltaE2000Loss(mode=mode)(prediction, target)  # type: ignore[arg-type]
    loss.backward()

    assert torch.isfinite(loss)
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum() > 0


@pytest.mark.parametrize("mode", ["stable", "source"])
def test_delta_e_constant_cube_has_finite_loss_and_gradient(mode: str) -> None:
    prediction = torch.zeros((1, 31, 2, 2), requires_grad=True)
    target = torch.full_like(prediction, 0.5)

    loss = DeltaE2000Loss(mode=mode)(  # type: ignore[arg-type]
        prediction, target
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_spectral_loss_wires_source_mrae_and_delta_e_weights() -> None:
    generator = torch.Generator().manual_seed(29)
    target = torch.rand((1, 31, 3, 3), generator=generator) + 0.1
    prediction = target + 0.01 * torch.randn(target.shape, generator=generator)
    criterion = SpectralReconstructionLoss(
        l1_weight=0.0,
        mrae_weight=1.0,
        sam_weight=0.0,
        delta_e_weight=0.1,
        mrae_denominator="source_additive",
        mrae_epsilon=1e-5,
    )

    assert criterion.delta_e.mode == "source"
    expected = MRAELoss(
        eps=1e-5, denominator="source_additive"
    )(prediction, target) + 0.1 * DeltaE2000Loss(mode="source")(
        prediction, target
    )

    assert criterion(prediction, target) == pytest.approx(expected.item(), rel=1e-6)
