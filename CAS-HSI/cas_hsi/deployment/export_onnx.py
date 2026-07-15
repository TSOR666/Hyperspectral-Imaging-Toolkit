"""ONNX export with dynamic spatial dimensions (specification sections 9.8, 9.9).

The export path is the **edge model**. The research backend's neighborhood attention
is a gather-shift-softmax over 9 offsets; even where it traces, spec 9.7 explicitly
lists custom neighborhood attention among the operators to keep out of the deployment
graph. :func:`export_onnx` therefore refuses an attention-bearing model unless you ask
it to convert one (and tells you what that costs).

Dynamic axes are not decoration: a hyperspectral scene is 482x512, a training patch is
128x128, and a tiled pass is 256x256. An export pinned to one of those is useless for
the other two. The equivalence check deliberately runs the graph at a size *different*
from the example input, because that is the only thing that actually proves the dynamic
axes work -- checking at the example size would pass even on a statically-shaped graph.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .replace_attention import attention_mixer_paths, replace_attention_mixers

__all__ = ["export_onnx", "run_onnx"]

_ONNX_HINT = (
    "ONNX export needs the optional dependencies:  pip install onnx onnxruntime"
)


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
    """Export ``model`` to ONNX with dynamic batch, height and width.

    Args:
        model: a CAS-HSI model. Must be free of attention spatial mixers, or
            ``replace_attention=True`` must be set.
        path: destination ``.onnx`` file.
        example_input: tracing input. Defaults to ``[1, 3, 128, 128]``.
        check_input: input for the equivalence check. Defaults to a *different*
            spatial size than ``example_input`` -- that is the point of the check.
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

    if example_input is None:
        example_input = torch.randn(1, 3, 128, 128)

    try:
        import onnx as _onnx  # noqa: F401  (import-for-presence: torch.onnx needs it installed)

        del _onnx
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(_ONNX_HINT) from exc

    with torch.no_grad():
        torch.onnx.export(
            model,
            example_input,
            str(path),
            input_names=["rgb"],
            output_names=["hsi"],
            dynamic_axes={
                "rgb": {0: "batch", 2: "height", 3: "width"},
                "hsi": {0: "batch", 2: "height", 3: "width"},
            },
            opset_version=opset,
            do_constant_folding=True,
        )

    if not check:
        return path

    # --- spec 9.9 equivalence, at a size the graph has never seen ---
    if check_input is None:
        # Deliberately odd and unequal to the example: it exercises the internal
        # reflect-pad-to-multiple-of-4 and the crop back, as well as the dynamic axes.
        check_input = torch.randn(1, 3, 127, 193)

    with torch.no_grad():
        torch_output = model(check_input).cpu().numpy()

    onnx_output = run_onnx(path, {"rgb": check_input.cpu().numpy()})[0]

    import numpy as np

    if onnx_output.shape != torch_output.shape:
        raise AssertionError(
            f"ONNX output shape {onnx_output.shape} != torch {torch_output.shape} at "
            f"{tuple(check_input.shape)}. The dynamic axes are not working."
        )
    np.testing.assert_allclose(torch_output, onnx_output, rtol=rtol, atol=atol)

    return path
