"""Tests for architecture recovery and ONNX export of historical checkpoints."""

import json

import pytest
import torch

from hsi_model.models.generator_v3 import NoiseRobustCSWinGenerator
from hsi_model.utils.onnx_export import (
    ARCHITECTURE_CONFIG_KEYS,
    export_checkpoint_to_onnx,
    load_checkpoint_payload,
    load_generator_from_any_checkpoint,
    rebuild_generator,
    recover_architecture,
    recover_structural_config,
)

onnx = pytest.importorskip("onnx", reason="ONNX export tests need onnx")
ort = pytest.importorskip("onnxruntime", reason="ONNX export tests need onnxruntime")


# Small but structurally complete: two stages deep, split_size 4 so the
# attention stripe padding is exercised at 32x32.
_BASE_CFG = {
    "in_channels": 3,
    "out_channels": 31,
    "base_channels": 16,
    "split_sizes": [4, 4, 4],
    "num_heads": 2,
    "norm_groups": 4,
    "cswin_max_long_axis": 128,
    "blocks_per_stage": 1,
}

# Every variant here must be recoverable from the tensors alone.
_VARIANTS = {
    "sstb_default": {},
    "sstb_pixelshuffle_thick": {
        "sampling": "pixelshuffle",
        "thick_output_head": True,
        "use_feature_norm": False,
        "use_input_denoising": False,
    },
    "sstb_efficient_no_outnorm": {
        "spectral_attention_type": "efficient",
        "smsa_output_norm": False,
        "decoder1_compress_first": True,
        "stage_depths": [1, 2, 2, 2, 1],
    },
    "sstb_cross_shaped": {"cswin_attention_mode": "cswin"},
    "sstb_window_cyclic": {"cswin_bias_mode": "window_cyclic"},
    "sstb_all_extras": {
        "use_spectral_input_skip": True,
        "spectral_input_skip_hidden": 12,
        "refinement_blocks": 2,
        "refinement_channels": 10,
        "cascade_stages": 2,
        "ffn_expansion": 4.0,
        "cbam_reduction": 2,
        "stage_num_heads": [2, 4, 8, 4, 2],
    },
    "legacy_with_noise_block": {
        "block_variant": "legacy_dtb",
        "use_noise_block": True,
        "spectral_attention_type": "efficient",
    },
    "legacy_smsa_no_noise": {
        "block_variant": "legacy_dtb",
        "use_noise_block": False,
        "spectral_attention_type": "s_msa",
        "legacy_ffn_expansion_factor": 2,
    },
}


def _build(extra=None, seed=0):
    config = {**_BASE_CFG, **(extra or {})}
    torch.manual_seed(seed)
    return NoiseRobustCSWinGenerator(config).eval(), config


@pytest.mark.parametrize("name", sorted(_VARIANTS))
def test_recovery_reproduces_every_variant_bit_exactly(name):
    """Rebuilding from the tensors alone must give an identical function."""
    reference, config = _build(_VARIANTS[name])
    state = reference.state_dict()

    # No embedded config at all: only norm_groups (GroupNorm group count) and
    # the cascade loop count are genuinely unrecoverable, so supply those.
    overrides = {"norm_groups": config["norm_groups"]}
    if config.get("cascade_stages", 1) > 1:
        overrides["cascade_stages"] = config["cascade_stages"]

    rebuilt, recovery, report = rebuild_generator(state, None, overrides=overrides)

    assert report.exact, report.describe()
    x = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(rebuilt(x), reference(x))
    assert recovery.config["block_variant"] == config.get("block_variant", "sstb")


def test_recovered_config_only_contains_architecture_keys():
    reference, config = _build()
    recovery = recover_architecture(
        reference.state_dict(),
        embedded_config={**config, "learning_rate": 1e-4, "data_dir": "/tmp/x"},
    )
    assert set(recovery.config).issubset(ARCHITECTURE_CONFIG_KEYS)
    assert "learning_rate" not in recovery.config
    assert "data_dir" not in recovery.config


def test_tensors_outrank_a_stale_embedded_config():
    """Older configs carry defaults that no longer describe the weights."""
    reference, config = _build({"spectral_attention_type": "efficient"})
    stale = {**config, "spectral_attention_type": "s_msa", "base_channels": 999}

    recovery = recover_architecture(reference.state_dict(), embedded_config=stale)

    assert recovery.config["spectral_attention_type"] == "efficient"
    assert recovery.config["base_channels"] == 16
    assert recovery.conflicts["spectral_attention_type"] == ("s_msa", "efficient")
    assert recovery.conflicts["base_channels"] == (999, 16)


def test_explicit_override_retracts_the_assumption_warning():
    reference, _ = _build()
    assumed = recover_architecture(reference.state_dict(), None)
    assert any(line.startswith("norm_groups=") for line in assumed.assumptions)

    overridden = recover_architecture(
        reference.state_dict(), None, overrides={"norm_groups": 4}
    )
    assert not any(line.startswith("norm_groups=") for line in overridden.assumptions)
    assert overridden.config["norm_groups"] == 4


def test_legacy_gan_checkpoint_layout(tmp_path):
    """Prefixed keys, a discriminator, a 0-d buffer and no embedded config."""
    reference, config = _build(_VARIANTS["legacy_with_noise_block"])
    state = {f"generator.{k}": v for k, v in reference.state_dict().items()}
    state["generator.iteration_count"] = torch.tensor(0)  # pre-(1,) buffer
    state["discriminator.head.weight"] = torch.zeros(4, 4)

    checkpoint = tmp_path / "gan_era.pth"
    torch.save({"epoch": 7, "state_dict": state}, checkpoint)

    generator, recovery, report, metadata = load_generator_from_any_checkpoint(
        checkpoint, overrides={"norm_groups": config["norm_groups"]}
    )

    assert report.exact, report.describe()
    assert any("iteration_count" in line for line in report.adapted)
    assert recovery.config["block_variant"] == "legacy_dtb"
    assert recovery.config["use_noise_block"] is True
    assert metadata["epoch"] == 7
    x = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(generator(x), reference(x))


def test_complete_ema_shadow_is_applied():
    reference, _ = _build()
    shadow = {k: torch.zeros_like(v) for k, v in reference.state_dict().items()}
    payload = {
        "state_dict": reference.state_dict(),
        "ema": {"shadow": shadow},
    }
    return_value = payload

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(torch, "load", lambda *a, **k: return_value)
        state, _config, metadata = load_checkpoint_payload("ignored.pth")

    assert metadata["ema_applied"] is True
    assert torch.count_nonzero(state["embedding.0.weight"]) == 0


def test_partial_ema_shadow_is_rejected():
    reference, _ = _build()
    full = reference.state_dict()
    shadow = {"embedding.0.weight": torch.zeros_like(full["embedding.0.weight"])}
    payload = {"state_dict": full, "ema": {"shadow": shadow}}

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(torch, "load", lambda *a, **k: payload)
        state, _config, metadata = load_checkpoint_payload("ignored.pth")

    assert metadata["ema_applied"] is False
    assert torch.equal(state["embedding.0.weight"], full["embedding.0.weight"])


def test_mismatched_architecture_raises_instead_of_loading_partially():
    reference, config = _build()
    state = dict(reference.state_dict())
    del state["embedding.0.bias"]

    with pytest.raises(RuntimeError, match="does not match the checkpoint tensors"):
        rebuild_generator(state, config, strict=True)


def test_non_generator_state_dict_is_rejected():
    with pytest.raises(ValueError, match="embedding.0.weight"):
        recover_structural_config({"fc.weight": torch.zeros(2, 2)})


@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_onnx_export_matches_eager_output(tmp_path, precision):
    reference, config = _build({"output_activation": "sigmoid"})
    checkpoint = tmp_path / "ckpt.pth"
    torch.save({"model_state_dict": reference.state_dict(), "config": config}, checkpoint)

    result = export_checkpoint_to_onnx(
        checkpoint,
        tmp_path / f"gen_{precision}.onnx",
        height=32,
        width=32,
        precision=precision,
    )

    assert result.onnx_path.is_file()
    assert result.load_report.exact
    assert result.parity["within_tolerance"], result.parity
    assert result.manifest["export"]["io_dtype"] == "float32"

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["architecture"]["output_activation"] == "sigmoid"
    assert manifest["export"]["precision"] == precision

    session = ort.InferenceSession(
        str(result.onnx_path), providers=["CPUExecutionProvider"]
    )
    torch.manual_seed(11)
    x = torch.rand(1, 3, 32, 32)
    actual = torch.from_numpy(
        session.run(None, {session.get_inputs()[0].name: x.numpy()})[0]
    )
    with torch.no_grad():
        expected = reference(x)
    tolerance = 1e-5 if precision == "fp32" else 5e-3
    assert torch.allclose(actual, expected, atol=tolerance)


def test_onnx_batch_axis_stays_dynamic(tmp_path):
    reference, config = _build()
    checkpoint = tmp_path / "ckpt.pth"
    torch.save({"model_state_dict": reference.state_dict(), "config": config}, checkpoint)

    result = export_checkpoint_to_onnx(
        checkpoint, tmp_path / "gen.onnx", height=32, width=32, verify=False
    )

    session = ort.InferenceSession(
        str(result.onnx_path), providers=["CPUExecutionProvider"]
    )
    torch.manual_seed(5)
    x = torch.rand(3, 3, 32, 32)
    actual = torch.from_numpy(
        session.run(None, {session.get_inputs()[0].name: x.numpy()})[0]
    )
    with torch.no_grad():
        expected = reference(x)
    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=1e-5)


def test_onnx_clamp_output_is_baked_into_the_graph(tmp_path):
    reference, config = _build({"output_activation": "none"})
    checkpoint = tmp_path / "ckpt.pth"
    torch.save({"model_state_dict": reference.state_dict(), "config": config}, checkpoint)

    result = export_checkpoint_to_onnx(
        checkpoint,
        tmp_path / "clamped.onnx",
        height=32,
        width=32,
        clamp_output=True,
    )

    session = ort.InferenceSession(
        str(result.onnx_path), providers=["CPUExecutionProvider"]
    )
    torch.manual_seed(7)
    x = torch.rand(1, 3, 32, 32)
    actual = session.run(None, {session.get_inputs()[0].name: x.numpy()})[0]
    assert actual.min() >= 0.0
    assert actual.max() <= 1.0
