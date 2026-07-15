"""INT8 quantization-aware training for the edge model (specification section 9.6).

The mixed profile the spec asks for:

    INT8   convolution weights and activations -- pointwise, depthwise, the RGB prior,
           the spectral output projection. These are the bulk of the compute.
    float  normalization, the attention softmax, the learned temperature, and residual
           accumulation. These carry small numbers whose *relative* precision matters:
           quantizing a softmax or a LayerScale gamma to 256 levels destroys exactly the
           low-magnitude structure a spectral reconstruction lives on.

Post-training quantization is not offered. A residual restoration network's output is a
*small correction* to a linear prior, and PTQ's calibration-free rounding lands squarely
on that correction. QAT lets the network adapt to the rounding instead.

*** DO NOT ASSUME INT8 IS FASTER. ***
Spec 9.6 is explicit and it is worth repeating: INT8 is a memory and (sometimes) a
throughput win on hardware with INT8 kernels for *depthwise* convolution. Plenty of
targets have no such kernel and fall back to float with an added quantize/dequantize
round-trip per layer, i.e. strictly slower. Measure on the actual device before
believing anything.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

__all__ = ["prepare_edge_model_for_qat", "convert_qat_model", "QAT_UNAVAILABLE"]

QAT_UNAVAILABLE = (
    "torch.ao.quantization is unavailable or incompatible in this build. QAT for "
    "CAS-HSI needs an eager-mode-quantizable torch (>=1.13 with torch.ao). Export the "
    "float edge model instead and quantize with your target's own toolchain "
    "(TensorRT INT8, ONNX Runtime QDQ, vendor SDK)."
)


def _require_ao() -> Any:
    try:
        import torch.ao.quantization as tq  # type: ignore
    except (ImportError, AttributeError) as exc:  # pragma: no cover - version dependent
        raise RuntimeError(QAT_UNAVAILABLE) from exc
    return tq


def _default_backend() -> str:
    """Pick a quantized engine this build actually supports.

    Hard-coding 'qnnpack' (the ARM engine) raises on an x86 CPU build, and hard-coding
    'fbgemm' raises on ARM. Ask torch which engines it was compiled with.
    """
    supported = list(getattr(torch.backends.quantized, "supported_engines", []))
    for candidate in ("fbgemm", "qnnpack", "x86", "onednn"):
        if candidate in supported:
            return candidate
    raise RuntimeError(
        f"No supported quantized engine in this torch build (supported: {supported}). "
        + QAT_UNAVAILABLE
    )


def prepare_edge_model_for_qat(
    model: nn.Module,
    backend: Optional[str] = None,
    example_input: Optional[torch.Tensor] = None,
) -> nn.Module:
    """Insert fake-quant observers on the convolutions only, then return the model.

    Train the returned model as usual (a short fine-tune from the float edge weights,
    at a low LR), then call :func:`convert_qat_model`.

    Args:
        model: an **edge** CAS-HSI model (no attention spatial mixers). Passing a
            research model is a mistake -- its neighborhood attention will not lower.
        backend: ``qnnpack`` (ARM), ``fbgemm``/``x86`` (Intel/AMD). Defaults to whichever
            engine this torch build actually supports.
        example_input: unused by the eager path; accepted so a caller can pass one
            without special-casing.
    """
    from .replace_attention import has_attention_mixers

    if has_attention_mixers(model):
        raise ValueError(
            "This model still contains attention spatial mixers. QAT targets the "
            "convolutional edge backend -- build_cas_hsi(variant, backend='edge') or "
            "replace_attention_mixers(model) first."
        )

    tq = _require_ao()
    del example_input

    if backend is None:
        backend = _default_backend()
    supported = list(getattr(torch.backends.quantized, "supported_engines", []))
    if backend not in supported:
        raise RuntimeError(
            f"Quantized engine {backend!r} is not supported by this torch build "
            f"(supported: {supported})."
        )
    torch.backends.quantized.engine = backend

    qconfig = tq.get_default_qat_qconfig(backend)

    # Everything float by default; opt convolutions IN. Doing it the other way round
    # (quantize all, opt sensitive modules out) is how norms and softmaxes end up
    # quantized by accident when a new module type is added later.
    model.qconfig = None  # type: ignore[assignment]
    quantized_modules = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            module.qconfig = qconfig  # type: ignore[assignment]
            quantized_modules += 1

    if quantized_modules == 0:  # pragma: no cover - defensive
        raise RuntimeError("No Conv2d modules found to quantize.")

    model.train()
    tq.prepare_qat(model, inplace=True)
    return model


def convert_qat_model(model: nn.Module) -> nn.Module:
    """Fold the observers and produce the INT8 inference model."""
    tq = _require_ao()
    model.eval()
    return tq.convert(model, inplace=False)
