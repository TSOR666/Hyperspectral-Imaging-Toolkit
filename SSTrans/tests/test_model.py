from __future__ import annotations

import torch

from hsiformer import build_model, load_checkpoint


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


def test_paper_residual_candidate_runs() -> None:
    model = _tiny_model("optimized_candidate").eval()
    with torch.inference_mode():
        outputs = model(torch.rand(1, 3, 16, 16))
    assert torch.isfinite(outputs).all()


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
