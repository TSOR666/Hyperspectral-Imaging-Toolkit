from __future__ import annotations

import torch
from torch.nn import functional as F

from hsiformer import build_model, load_checkpoint
from hsiformer.attention import Spectral_MSA
from hsiformer.model import SSTLayer


def _tiny_model(preset: str = "legacy"):
    return build_model(
        preset,
        hidden_dim=8,
        input_resolution=(16, 16),
        n_blocks=(1,),
        bottle_depth=1,
        n_refine=1,
        patch_size=2,
        use_checkpoint=False,
    )


def test_forward_preserves_requested_shape() -> None:
    model = _tiny_model().eval()
    inputs = torch.rand(1, 3, 13, 15)
    with torch.inference_mode():
        outputs = model(inputs)
    assert outputs.shape == (1, 31, 13, 15)


def test_constructor_does_not_mutate_default_sequences() -> None:
    first = _tiny_model()
    second = _tiny_model()
    assert len(first.downblocks) == len(second.downblocks) == 1
    assert len(first.upblocks) == len(second.upblocks) == 1


def test_no_rpe_preset_removes_spectral_bias_parameters() -> None:
    legacy_keys = set(_tiny_model("legacy").state_dict())
    no_rpe_keys = set(_tiny_model("ablation_no_rpe").state_dict())
    assert any(key.endswith("s_msa.relative_bias") for key in legacy_keys)
    assert not any(key.endswith("s_msa.relative_bias") for key in no_rpe_keys)


def test_recommended_retrain_preset_runs() -> None:
    model = _tiny_model("recommended_retrain").eval()
    with torch.inference_mode():
        outputs = model(torch.rand(1, 3, 16, 16))
    assert torch.isfinite(outputs).all()


def test_rectangular_candidate_avoids_outer_square_padding() -> None:
    model = _tiny_model("rectangular_candidate").eval()
    embedded_shapes: list[tuple[int, int]] = []
    handle = model.embed.register_forward_pre_hook(
        lambda _module, inputs: embedded_shapes.append(
            tuple(inputs[0].shape[-2:])
        )
    )
    with torch.inference_mode():
        outputs = model(torch.rand(1, 3, 13, 15))
    handle.remove()

    assert embedded_shapes == [(14, 16)]
    assert outputs.shape == (1, 31, 13, 15)
    assert torch.isfinite(outputs).all()


def test_rectangular_mode_preserves_square_model_output() -> None:
    torch.manual_seed(5)
    square = _tiny_model("recommended_retrain").eval()
    rectangular = _tiny_model("rectangular_candidate").eval()
    rectangular.load_state_dict(square.state_dict())
    inputs = torch.rand(1, 3, 16, 16)

    with torch.inference_mode():
        expected = square(inputs)
        actual = rectangular(inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_recommended_retrain_uses_stage_spectral_heads() -> None:
    model = build_model(
        "recommended_retrain",
        bottle_depth=1,
        n_refine=1,
        use_checkpoint=False,
    )
    layouts = {
        (module.qkv.in_channels, module.num_heads)
        for module in model.modules()
        if isinstance(module, Spectral_MSA)
    }
    assert layouts == {(32, 2), (64, 4), (128, 8), (256, 16)}
    assert all(channels // heads == 16 for channels, heads in layouts)


def test_recommended_retrain_keeps_cat_rpe_only() -> None:
    keys = set(_tiny_model("recommended_retrain").state_dict())
    assert not any(key.endswith("s_msa.relative_bias") for key in keys)
    assert any("relative_position_bias_table" in key for key in keys)


def test_branch_delta_is_identity_for_a_zero_transform_branch() -> None:
    layer = SSTLayer(
        dim=8,
        head=2,
        resolution=(8, 8),
        split_size=1,
        num_blocks=3,
        spectral_rpe="none",
        spectral_head_mode="stage",
        residual_mode="branch_delta",
    ).eval()
    for parameter in layer.parameters():
        torch.nn.init.zeros_(parameter)

    inputs = torch.randn(2, 8, 8, 8)
    with torch.inference_mode():
        outputs = layer(inputs)
    torch.testing.assert_close(outputs, inputs)


def test_tiny_batch_l1_loss_decreases() -> None:
    torch.manual_seed(7)
    model = build_model(
        "ablation_no_rpe",
        hidden_dim=8,
        input_resolution=(8, 8),
        n_blocks=(1,),
        bottle_depth=1,
        n_refine=1,
        patch_size=2,
        use_checkpoint=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    inputs = torch.rand(1, 3, 8, 8)
    target = torch.rand(1, 31, 8, 8)

    with torch.no_grad():
        initial_loss = F.l1_loss(model(inputs), target)
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        loss = F.l1_loss(model(inputs), target)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final_loss = F.l1_loss(model(inputs), target)

    assert final_loss < initial_loss


def test_checkpoint_loader_accepts_wrapped_model_prefix(tmp_path) -> None:
    source = _tiny_model()
    prefixed = {
        f"model.{key}": value.clone()
        for key, value in source.state_dict().items()
    }
    path = tmp_path / "checkpoint.pt"
    torch.save({"state_dict": prefixed}, path)

    target = _tiny_model()
    incompatible = load_checkpoint(target, path)
    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys
