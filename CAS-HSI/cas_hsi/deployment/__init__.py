"""Deployment backend: operator replacement, ONNX export, INT8 QAT (spec 9)."""

from .export_onnx import export_onnx, run_onnx
from .quantization import convert_qat_model, prepare_edge_model_for_qat
from .replace_attention import (
    attention_mixer_paths,
    has_attention_mixers,
    replace_attention_mixers,
)

__all__ = [
    "attention_mixer_paths",
    "convert_qat_model",
    "export_onnx",
    "has_attention_mixers",
    "prepare_edge_model_for_qat",
    "replace_attention_mixers",
    "run_onnx",
]
