"""ONNX export with dynamic *pre-padded* spatial dimensions (specification 9.8/9.9).

The export path is the **edge model**. The research backend's neighborhood attention
is a gather-shift-softmax over 9 offsets; even where it traces, spec 9.7 explicitly
lists custom neighborhood attention among the operators to keep out of the deployment
graph. :func:`export_onnx` therefore refuses an attention-bearing model unless you ask
it to convert one (and tells you what that costs).

The eager model accepts arbitrary spatial dimensions by padding to a multiple of four,
then cropping back. The current PyTorch symbolic-shape exporter cannot preserve that
arbitrary modulo-and-reflect-pad path: a graph can be labelled dynamic while hard-coding
the PixelUnshuffle reshape. The ONNX contract is therefore explicit and portable: feed
RGB already padded to H/W divisible by four, run the graph at any such spatial size,
then crop its result outside ONNX. This mirrors deployment preprocessing and avoids a
silently invalid export. The equivalence check uses a different valid resolution.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .replace_attention import attention_mixer_paths, replace_attention_mixers

__all__ = ["export_onnx", "run_onnx"]

_ONNX_HINT = (
    "ONNX export needs the optional dependencies: "
    "pip install onnx onnxscript onnxruntime"
)


class _PaddedExportWrapper(nn.Module):
    """Expose CAS-HSI's multiple-of-four core without its eager-only padding path."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.model.forward_padded(rgb)  # type: ignore[attr-defined]


def _check_padded_input(value: torch.Tensor, name: str) -> None:
    if value.dim() != 4:
        raise ValueError(f"{name} must be a 4-D NCHW RGB tensor, got {tuple(value.shape)}")
    height, width = value.shape[-2:]
    if height % 4 or width % 4:
        raise ValueError(
            f"{name} spatial dimensions must be divisible by 4 for ONNX export, got "
            f"{height}x{width}. Pad RGB before the ONNX call and crop the output back."
        )


def _require_onnx_export_dependencies() -> None:
    """Check the dependencies needed by the dynamic torch.export ONNX path."""
    try:
        import onnx as _onnx  # noqa: F401
        import onnxscript as _onnxscript  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(_ONNX_HINT) from exc


def _require_onnxruntime():
    try:
        import onnxruntime  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(_ONNX_HINT) from exc
    return onnxruntime


def run_onnx(path: str | Path, inputs: Dict[str, Any]) -> List[Any]:
    """Run an exported graph. Returns the output tensors as numpy arrays."""
    onnxruntime = _require_onnxruntime()
    session = onnxruntime.InferenceSession(
        str(path), providers=["CPUExecutionProvider"]
    )
    return session.run(None, inputs)


def export_onnx(
    model: nn.Module,
    path: str | Path,
    *,
    example_input: Optional[torch.Tensor] = None,
    check_input: Optional[torch.Tensor] = None,
    opset: int = 18,
    replace_attention: bool = False,
    check: bool = True,
    rtol: float = 1e-3,
    atol: float = 1e-4,
) -> Path:
    """Export ``model`` to ONNX with dynamic pre-padded height and width.

    Args:
        model: a CAS-HSI model. Must be free of attention spatial mixers, or
            ``replace_attention=True`` must be set.
        path: destination ``.onnx`` file.
        example_input: pre-padded tracing input. Defaults to ``[1, 3, 128, 128]``.
        check_input: pre-padded input for the equivalence check. Defaults to a
            *different* valid spatial size than ``example_input``.
        replace_attention: convert attention mixers to their conv equivalents first.
            Note this randomizes those weights (see :mod:`.replace_attention`); it is
            only meaningful on a model you are about to distil or retrain.
        check: compare the exported graph against PyTorch (needs onnxruntime).

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # --- spec 9.8 pre-export checklist ---
    # 1. eval mode (this is also what disables stochastic depth: DropPath is an exact
    #    identity outside training, so nothing random can be baked into the graph).
    model.eval()
    for module in model.modules():
        if type(module).__name__ == "DropPath" and module.training:  # pragma: no cover
            raise RuntimeError("DropPath is still in training mode; call model.eval().")

    # 2. no unsupported attention operators remain.
    remaining = attention_mixer_paths(model)
    if remaining:
        if not replace_attention:
            raise RuntimeError(
                f"{len(remaining)} attention spatial mixer(s) remain in the graph, e.g. "
                f"{remaining[:3]}. Spec 9.7 keeps neighborhood attention out of the "
                "deployment graph.\n\n"
                "Either export an edge model -- build_cas_hsi(variant, backend='edge'), "
                "trained from configs/cas_hsi_edge.yaml -- or pass replace_attention=True "
                "to swap the mixers here. Be aware the swapped-in convolutions have RANDOM "
                "weights: that path is for distillation, not for a free edge model."
            )
        model, _ = replace_attention_mixers(model)
        model.eval()

    if not hasattr(model, "forward_padded"):
        raise TypeError(
            "export_onnx expects a CAS-HSI model with forward_padded(); use the "
            "package's CASHSI builder rather than an arbitrary nn.Module."
        )

    if example_input is None:
        example_input = torch.randn(1, 3, 128, 128)
    _check_padded_input(example_input, "example_input")

    _require_onnx_export_dependencies()

    export_kwargs: Dict[str, Any] = {
        "input_names": ["rgb"],
        "output_names": ["hsi"],
        "opset_version": opset,
        "do_constant_folding": True,
    }
    # Dynamic spatial axes are a contract, not merely symbolic labels. The input is
    # pre-padded, so H/W can be expressed as multiples of four; that relationship is
    # what makes PixelUnshuffle's internal reshapes valid at every supported size.
    if "dynamo" not in inspect.signature(torch.onnx.export).parameters:
        raise RuntimeError(
            "Dynamic CAS-HSI ONNX export requires a PyTorch release with the "
            "torch.export-based ONNX exporter (PyTorch >= 2.6)."
        )
    dim = torch.export.Dim
    export_kwargs.update(
        dynamo=True,
        dynamic_shapes={
            "rgb": {
                2: 4 * dim("height", min=1),
                3: 4 * dim("width", min=1),
            }
        },
    )

    export_model = _PaddedExportWrapper(model).eval()
    with torch.no_grad():
        torch.onnx.export(export_model, example_input, str(path), **export_kwargs)

    if not check:
        return path

    # --- spec 9.9 equivalence, at a size the graph has never seen ---
    if check_input is None:
        # Different from the trace size, while honoring the pre-padded ONNX contract.
        check_input = torch.randn(1, 3, 124, 196)
    _check_padded_input(check_input, "check_input")

    with torch.no_grad():
        torch_output = export_model(check_input).cpu().numpy()

    onnx_output = run_onnx(path, {"rgb": check_input.cpu().numpy()})[0]

    import numpy as np

    if onnx_output.shape != torch_output.shape:
        raise AssertionError(
            f"ONNX output shape {onnx_output.shape} != torch {torch_output.shape} at "
            f"{tuple(check_input.shape)}. The dynamic axes are not working."
        )
    np.testing.assert_allclose(torch_output, onnx_output, rtol=rtol, atol=atol)

    return path
