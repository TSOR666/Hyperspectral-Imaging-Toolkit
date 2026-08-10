"""Regression tests for ONNX Runtime inference boundary handling."""

from __future__ import annotations

import importlib.util
import sys
import types
import uuid
from pathlib import Path

import numpy as np
import pytest
import torch


def _load_onnx_tester_module():
    path = Path(__file__).resolve().parents[1] / "onnx_test_ntire.py"
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    name = f"onnx_test_ntire_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _fake_onnxruntime(
    input_type: str,
    output_type: str,
    observed: dict[str, object],
) -> types.SimpleNamespace:
    input_dtype = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
    }[input_type]
    output_dtype = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
    }[output_type]

    class SessionOptions:
        intra_op_num_threads = 0

    class InferenceSession:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def get_providers(self):
            return ["CPUExecutionProvider"]

        def get_inputs(self):
            return [
                types.SimpleNamespace(
                    name="rgb",
                    type=input_type,
                    shape=[None, 3, 4, 5],
                )
            ]

        def get_outputs(self):
            return [
                types.SimpleNamespace(
                    name="hsi",
                    type=output_type,
                    shape=[None, 31, 4, 5],
                )
            ]

        def run(self, names, feeds):
            assert names == ["hsi"]
            array = feeds["rgb"]
            observed["dtype"] = array.dtype
            observed["contiguous"] = array.flags.c_contiguous
            assert array.dtype == np.dtype(input_dtype)
            return [
                np.ones(
                    (array.shape[0], 31, array.shape[2], array.shape[3]),
                    dtype=output_dtype,
                )
            ]

    return types.SimpleNamespace(
        SessionOptions=SessionOptions,
        InferenceSession=InferenceSession,
        get_available_providers=lambda: ["CPUExecutionProvider"],
    )


@pytest.mark.parametrize(
    ("input_type", "output_type", "expected_input_dtype"),
    [
        ("tensor(float)", "tensor(float)", np.float32),
        ("tensor(float16)", "tensor(float16)", np.float16),
        ("tensor(double)", "tensor(double)", np.float64),
    ],
)
def test_onnx_generator_feeds_the_graph_declared_dtype(
    monkeypatch,
    input_type,
    output_type,
    expected_input_dtype,
):
    tool = _load_onnx_tester_module()
    observed: dict[str, object] = {}
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        _fake_onnxruntime(input_type, output_type, observed),
    )

    model = tool.OnnxGenerator("fake.onnx")
    # A non-contiguous tensor also exercises the explicit C-array conversion.
    rgb = torch.rand(1, 3, 5, 4).transpose(-1, -2)
    prediction = model(rgb)

    assert observed["dtype"] == np.dtype(expected_input_dtype)
    assert observed["contiguous"] is True
    assert prediction.dtype == torch.float32
    assert tuple(prediction.shape) == (1, 31, 4, 5)


def test_onnx_tester_rejects_unknown_graph_dtype():
    tool = _load_onnx_tester_module()

    with pytest.raises(TypeError, match="Unsupported ONNX input type"):
        tool._numpy_dtype_for_onnx_tensor("tensor(bfloat16)", role="input")


def test_onnx_tester_allows_public_test_mosaics_by_default():
    tool = _load_onnx_tester_module()

    args = tool.parse_args(["--onnx", "model.onnx", "--data_root", "dataset"])
    assert args.require_gt is False

    strict = tool.parse_args(
        ["--onnx", "model.onnx", "--data_root", "dataset", "--require_gt"]
    )
    assert strict.require_gt is True
