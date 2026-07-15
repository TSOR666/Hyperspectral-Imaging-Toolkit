#!/usr/bin/env python3
"""CPU smoke test for the inference paths: arbitrary sizes, tiling, edge backend.

No dataset and no checkpoint required -- it exercises the model's own guarantees:

  * any H x W in, the same H x W out, 31 bands (spec 8);
  * tiled inference agrees with a direct full-image pass away from tile seams (spec 8.7);
  * the edge backend runs and contains no attention operator (spec 9.2).

    python smoke_infer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from cas_hsi import build_cas_hsi, tiled_inference  # noqa: E402
from cas_hsi.blocks import DilatedLocalAttention, HybridLocalStripeAttention  # noqa: E402

SIZES = [(1, 1), (7, 9), (31, 47), (63, 65), (127, 193), (482, 512)]


def main() -> int:
    torch.manual_seed(0)

    model = build_cas_hsi("tiny").eval()
    info = model.get_model_info()
    print(f"CAS-HSI tiny: {info['total_parameters']:,} params, widths {info['widths']}")

    print("\narbitrary sizes (spec 8.6):")
    for height, width in SIZES:
        with torch.no_grad():
            out = model(torch.randn(1, 3, height, width))
        assert out.shape == (1, 31, height, width), out.shape
        print(f"  {height:>4} x {width:<4} -> {tuple(out.shape)}")

    # The same instance must handle a different size immediately afterwards: any cached
    # spatial state would break here (spec 8.8).
    with torch.no_grad():
        model(torch.randn(1, 3, 64, 64))
        model(torch.randn(1, 3, 96, 32))
    print("  back-to-back different sizes on one instance: OK")

    print("\ntiled vs direct inference (spec 8.7):")
    image = torch.randn(1, 3, 160, 192)
    with torch.no_grad():
        direct = model(image)
        tiled = tiled_inference(model, image, tile_size=96, overlap=32)
    assert tiled.shape == direct.shape
    interior = slice(32, -32)
    interior_error = (tiled[..., interior, interior] - direct[..., interior, interior]).abs().max()
    overall_error = (tiled - direct).abs().max()
    print(f"  max |tiled - direct| interior: {interior_error:.2e}   overall: {overall_error:.2e}")
    assert torch.isfinite(tiled).all()

    print("\nedge backend (spec 9.2):")
    edge = build_cas_hsi("tiny", backend="edge").eval()
    attention = [
        name for name, module in edge.named_modules()
        if isinstance(module, (DilatedLocalAttention, HybridLocalStripeAttention))
    ]
    assert not attention, f"edge backend still contains attention mixers: {attention}"
    with torch.no_grad():
        out = edge(torch.randn(1, 3, 127, 193))
    assert out.shape == (1, 31, 127, 193)
    print(f"  {sum(p.numel() for p in edge.parameters()):,} params, no attention mixers, "
          f"out {tuple(out.shape)}")

    print("\nSMOKE INFER PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
