"""Deployment backend and ONNX export (specification sections 9.2, 9.8, 9.9).

onnx / onnxruntime are optional. The tests that need them skip cleanly when they are
absent -- but the operator-replacement tests, which are the part that actually decides
whether an export is *possible*, run unconditionally.
"""

from __future__ import annotations

import pytest
import torch

from cas_hsi import build_cas_hsi
from cas_hsi.blocks import (
    ConvSpatialMixer,
    CrossChannelAttention,
    DilatedLocalAttention,
    HybridLocalStripeAttention,
    MultiDilationDepthwiseMixer,
)
from cas_hsi.deployment import (
    attention_mixer_paths,
    has_attention_mixers,
    replace_attention_mixers,
)
from cas_hsi.deployment.export_onnx import export_onnx


# ------------------------------------------------------- operator replacement --


def test_research_model_has_attention_mixers(small_model):
    assert has_attention_mixers(small_model)


def test_edge_model_has_no_attention_mixers(small_edge_model):
    assert not has_attention_mixers(small_edge_model)
    assert attention_mixer_paths(small_edge_model) == []


def test_replace_attention_mixers_swaps_every_one(small_config):
    model = build_cas_hsi(small_config)
    expected = len(attention_mixer_paths(model))
    assert expected > 0

    model, report = replace_attention_mixers(model)

    assert len(report) == expected
    assert not has_attention_mixers(model)
    assert model.config.backend == "edge", "get_model_info() would still claim 'research'"


def test_replacement_maps_each_mixer_to_the_right_conv(small_config):
    model = build_cas_hsi(small_config)
    kinds = {
        name: type(module).__name__
        for name, module in model.named_modules()
        if isinstance(module, (DilatedLocalAttention, HybridLocalStripeAttention))
    }
    model, _ = replace_attention_mixers(model)

    for path, old_kind in kinds.items():
        new = model.get_submodule(path)
        if old_kind == "DilatedLocalAttention":
            assert isinstance(new, MultiDilationDepthwiseMixer), path
        else:
            assert isinstance(new, ConvSpatialMixer), path


def test_replacement_preserves_the_block_shell(small_config):
    """Spec 9.2: only the spatial mixer changes. Norms, channel attention, FFN, LayerScale stay."""
    model = build_cas_hsi(small_config)
    before = {
        name: type(module).__name__
        for name, module in model.named_modules()
        if not name.endswith("spatial_mixer") and "spatial_mixer." not in name
    }
    model, _ = replace_attention_mixers(model)
    after = {
        name: type(module).__name__
        for name, module in model.named_modules()
        if not name.endswith("spatial_mixer") and "spatial_mixer." not in name
    }
    assert before == after

    # Cross-channel attention must survive: its cost is independent of H*W and it is
    # what carries the spectral prior.
    assert any(isinstance(m, CrossChannelAttention) for m in model.modules())


def test_swapped_model_matches_a_fresh_edge_model_exactly(small_config):
    """A swapped research model and a from-scratch edge model must be the same architecture."""
    swapped, _ = replace_attention_mixers(build_cas_hsi(small_config))
    fresh = build_cas_hsi(small_config.as_edge())

    assert sum(p.numel() for p in swapped.parameters()) == sum(
        p.numel() for p in fresh.parameters()
    )
    assert set(swapped.state_dict()) == set(fresh.state_dict())


def test_swapped_model_still_runs(small_config):
    model, _ = replace_attention_mixers(build_cas_hsi(small_config))
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 31, 47))
    assert out.shape == (1, 31, 31, 47)
    assert torch.isfinite(out).all()


# ------------------------------------------------------------------- export ----


def test_export_refuses_a_model_with_attention(small_model, tmp_path):
    """Spec 9.7 keeps neighborhood attention out of the deployment graph."""
    with pytest.raises(RuntimeError, match="attention spatial mixer"):
        export_onnx(small_model, tmp_path / "x.onnx", replace_attention=False, check=False)


def test_export_edge_model_with_dynamic_axes(small_edge_model, tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    path = export_onnx(
        small_edge_model,
        tmp_path / "cas_hsi.onnx",
        example_input=torch.randn(1, 3, 64, 64),
        check=False,
    )
    assert path.exists()

    import onnx

    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)

    # The height/width axes must be symbolic, not baked to 64.
    rgb = graph.graph.input[0]
    dims = rgb.type.tensor_type.shape.dim
    assert dims[2].dim_param, "height is a fixed dimension, not dynamic"
    assert dims[3].dim_param, "width is a fixed dimension, not dynamic"


def test_onnx_matches_torch_at_a_different_size(small_edge_model, tmp_path):
    """Spec 9.9. The check size deliberately differs from the trace size -- exporting at
    64x64 and checking at 64x64 would pass even on a statically-shaped graph."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    pytest.importorskip("onnxruntime")

    import numpy as np

    from cas_hsi.deployment import run_onnx

    path = export_onnx(
        small_edge_model,
        tmp_path / "cas_hsi.onnx",
        example_input=torch.randn(1, 3, 64, 64),
        check=False,
    )

    probe = torch.randn(1, 3, 124, 196)  # different from the trace size, still divisible by four
    with torch.no_grad():
        torch_output = small_edge_model(probe).cpu().numpy()

    onnx_output = run_onnx(path, {"rgb": probe.cpu().numpy()})[0]

    assert onnx_output.shape == torch_output.shape == (1, 31, 124, 196)
    np.testing.assert_allclose(torch_output, onnx_output, rtol=1e-3, atol=1e-4)


def test_export_with_replace_attention_succeeds(small_model, tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    model = build_cas_hsi(small_model.config)
    path = export_onnx(
        model,
        tmp_path / "converted.onnx",
        example_input=torch.randn(1, 3, 32, 32),
        replace_attention=True,
        check=False,
    )
    assert path.exists()


def test_export_rejects_an_unpadded_example_input(small_edge_model, tmp_path):
    with pytest.raises(ValueError, match="divisible by 4"):
        export_onnx(
            small_edge_model,
            tmp_path / "invalid.onnx",
            example_input=torch.randn(1, 3, 127, 193),
            check=False,
        )


# ------------------------------------------------------------- quantization ----


def test_qat_rejects_a_research_model(small_model):
    from cas_hsi.deployment import prepare_edge_model_for_qat

    with pytest.raises(ValueError, match="attention spatial mixers"):
        prepare_edge_model_for_qat(build_cas_hsi(small_model.config))


def test_qat_prepares_the_edge_model(small_config):
    from cas_hsi.deployment import prepare_edge_model_for_qat

    pytest.importorskip("torch.ao.quantization")
    model = build_cas_hsi(small_config.as_edge())
    try:
        # backend=None auto-selects an engine this torch build actually supports.
        prepared = prepare_edge_model_for_qat(model)
    except RuntimeError as exc:  # torch.ao missing or incompatible in this build
        pytest.skip(str(exc))

    # Convolutions carry a qconfig; normalization and attention must not.
    from cas_hsi.layers import BiasFreeLayerNorm2d

    norms = [m for m in prepared.modules() if isinstance(m, BiasFreeLayerNorm2d)]
    assert norms
    assert all(getattr(m, "qconfig", None) is None for m in norms), (
        "normalization was handed a qconfig; spec 9.6 keeps it in float"
    )
