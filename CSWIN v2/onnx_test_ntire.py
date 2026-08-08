#!/usr/bin/env python3
"""Score an exported ``.onnx`` generator on ARAD-1K with the NTIRE protocol.

The reconstruction runs entirely in ONNX Runtime - no generator class, no
checkpoint, no architecture config. Only the data loading and the metric
definitions come from this repository, so the numbers are directly comparable
with ``cswin_test_ntire.py``: same split resolution, same MST++ RGB
normalization, same 226x256 center-crop scoring window, same
MRAE/RMSE/PSNR/SAM/SSIM/MAE implementations.

Export first, then evaluate::

    python export_onnx.py --checkpoint old_best.pth --output old_best.onnx \
        --height 128 --width 128 --precision fp16

    python onnx_test_ntire.py --onnx old_best.onnx --data_root /data/ARAD_1K

Tiling is chosen from the graph itself: a graph frozen at 128x128 is tiled with
the same overlap-blend as ``PatchInference``; a graph frozen at the full image
size runs in one pass. ``--compare_checkpoint`` additionally runs the original
PyTorch weights so you can see exactly what the export cost you.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cswin_test_ntire import (  # noqa: E402
    NTIRESplitDataset,
    _crop_for_metrics,
    _summarize_metrics,
)
from hsi_model.utils.inference import geometric_self_ensemble  # noqa: E402
from hsi_model.utils.metrics import compute_metrics, compute_mrae  # noqa: E402
from hsi_model.utils.patch_inference import PatchInference  # noqa: E402

LOGGER = logging.getLogger("onnx_test_ntire")


@dataclass
class OnnxTestConfig:
    onnx_path: str
    data_root: str
    output_dir: str = "./onnx_test_results"
    split: str = "auto"
    providers: Optional[List[str]] = None
    tiling: str = "auto"
    patch_size: Optional[int] = None
    overlap: int = 16
    patch_batch_size: int = 4
    intra_op_threads: int = 0
    ensemble_mode: str = "none"
    bgr2rgb: bool = True
    rgb_normalization: str = "mst"
    crop_border: int = 128
    crop_mode: str = "arad1k"
    compute_all_metrics: bool = True
    save_predictions: bool = False
    save_format: str = "mat"
    max_samples: Optional[int] = None
    start_idx: int = 0
    require_gt: bool = True
    quiet_patches: bool = True
    compare_checkpoint: Optional[str] = None


class OnnxGenerator(nn.Module):
    """``nn.Module`` facade over an ONNX Runtime session.

    Being an ``nn.Module`` lets :class:`PatchInference` drive it unchanged, so
    the tiling and cosine-blend stitching are bit-identical to the PyTorch path.
    The attribute is deliberately named ``session`` and not ``generator``:
    ``PatchInference`` treats a ``.generator`` attribute as a GAN wrapper.
    """

    def __init__(
        self,
        onnx_path: str,
        providers: Optional[Sequence[str]] = None,
        intra_op_threads: int = 0,
    ) -> None:
        super().__init__()
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "onnx_test_ntire.py needs onnxruntime: pip install onnxruntime "
                "(or onnxruntime-gpu for CUDA)"
            ) from exc

        options = ort.SessionOptions()
        if intra_op_threads > 0:
            options.intra_op_num_threads = int(intra_op_threads)

        resolved = list(providers) if providers else None
        if resolved is None:
            available = set(ort.get_available_providers())
            resolved = [
                name
                for name in ("CUDAExecutionProvider", "CPUExecutionProvider")
                if name in available
            ] or ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(
            str(onnx_path), sess_options=options, providers=resolved
        )
        self.providers = list(self.session.get_providers())

        spec_in = self.session.get_inputs()[0]
        spec_out = self.session.get_outputs()[0]
        self.input_name = spec_in.name
        self.output_name = spec_out.name
        self.input_numpy_dtype = (
            np.float16 if "float16" in spec_in.type else np.float32
        )
        self.input_shape = list(spec_in.shape)
        self.static_hw = self._static_hw(self.input_shape)
        self.out_channels = (
            int(spec_out.shape[1]) if isinstance(spec_out.shape[1], int) else None
        )

    @staticmethod
    def _static_hw(shape: Sequence[Any]) -> Optional[Tuple[int, int]]:
        if len(shape) != 4:
            return None
        height, width = shape[2], shape[3]
        if isinstance(height, int) and isinstance(width, int):
            return int(height), int(width)
        return None

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        array = rgb.detach().cpu().numpy().astype(self.input_numpy_dtype, copy=False)
        output = self.session.run([self.output_name], {self.input_name: array})[0]
        return torch.from_numpy(np.asarray(output, dtype=np.float32))


class OnnxNTIRETester:
    def __init__(self, config: OnnxTestConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.dataset = NTIRESplitDataset(
            data_root=config.data_root,
            split=config.split,
            bgr2rgb=config.bgr2rgb,
            rgb_normalization=config.rgb_normalization,
            max_samples=config.max_samples,
            start_idx=config.start_idx,
            require_gt=config.require_gt,
        )
        self.model = OnnxGenerator(
            config.onnx_path,
            providers=config.providers,
            intra_op_threads=config.intra_op_threads,
        )
        LOGGER.info(
            "ONNX session providers=%s input=%s dtype=%s",
            self.model.providers,
            self.model.input_shape,
            self.model.input_numpy_dtype.__name__,
        )

        self.manifest = self._load_manifest(Path(config.onnx_path))
        self.reference_model = self._load_reference_model()

    @staticmethod
    def _load_manifest(onnx_path: Path) -> Dict[str, Any]:
        manifest_path = onnx_path.with_suffix(".json")
        if not manifest_path.is_file():
            return {}
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.warning("Could not read manifest %s: %s", manifest_path, exc)
            return {}

    def _load_reference_model(self) -> Optional[nn.Module]:
        if not self.config.compare_checkpoint:
            return None
        from hsi_model.utils.onnx_export import load_generator_from_any_checkpoint

        generator, recovery, report, _metadata = load_generator_from_any_checkpoint(
            self.config.compare_checkpoint
        )
        LOGGER.info(
            "Reference checkpoint rebuilt (%s); state match: %s",
            recovery.summary(),
            report.describe(),
        )
        return generator.eval()

    def _make_patcher(self, model: nn.Module, patch_size: int) -> PatchInference:
        return PatchInference(
            model=model,
            patch_size=patch_size,
            overlap=self.config.overlap,
            batch_size=self.config.patch_batch_size,
            device=torch.device("cpu"),
            apply_sigmoid=False,
        )

    def _resolve_tiling(self, image_hw: Tuple[int, int]) -> Optional[int]:
        """Return the tile size to use, or ``None`` for a single full-frame pass."""
        mode = self.config.tiling.strip().lower()
        if mode not in ("auto", "tile", "full"):
            raise ValueError("--tiling must be one of: auto, tile, full")

        static_hw = self.model.static_hw
        if mode == "full":
            if static_hw is not None and static_hw != image_hw:
                raise ValueError(
                    f"--tiling full needs a graph matching the image size, but the "
                    f"graph is frozen at {static_hw} and the image is {image_hw}. "
                    "Re-export with --height/--width, or use --tiling tile."
                )
            return None

        if mode == "tile":
            patch = self.config.patch_size or (
                static_hw[0] if static_hw else 128
            )
            if static_hw is not None and static_hw != (patch, patch):
                raise ValueError(
                    f"--patch_size {patch} does not match the graph's frozen input "
                    f"{static_hw}. Re-export at the tile size you want."
                )
            return int(patch)

        # auto
        if static_hw is None:
            return int(self.config.patch_size) if self.config.patch_size else None
        if static_hw == image_hw:
            return None
        if static_hw[0] != static_hw[1]:
            raise ValueError(
                f"Graph is frozen at a non-square {static_hw} that does not match "
                f"the image size {image_hw}; tiling needs a square graph. "
                "Re-export at the image size or at a square tile size."
            )
        return int(static_hw[0])

    def _predict(self, model: nn.Module, rgb: torch.Tensor) -> torch.Tensor:
        batched = rgb.unsqueeze(0)
        image_hw = (int(batched.shape[-2]), int(batched.shape[-1]))
        patch = self._resolve_tiling(image_hw)

        if patch is None:
            def run(tensor: torch.Tensor) -> torch.Tensor:
                with torch.inference_mode():
                    return model(tensor).float()
        else:
            patcher = self._make_patcher(model, patch)

            def run(tensor: torch.Tensor) -> torch.Tensor:
                return patcher.predict(
                    tensor, show_progress=not self.config.quiet_patches
                )

        if self.config.ensemble_mode == "d4":
            return geometric_self_ensemble(run, batched).float()
        return run(batched).float()

    def _metrics_for(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Dict[str, float]:
        pred_unclamped = pred.float()
        pred_clamped = pred_unclamped.clamp(0.0, 1.0)
        target = target.float()

        unclamped_pred, cropped_target = _crop_for_metrics(
            pred_unclamped, target, self.config.crop_border, self.config.crop_mode
        )
        clamped_pred, cropped_target = _crop_for_metrics(
            pred_clamped, target, self.config.crop_border, self.config.crop_mode
        )
        metrics = compute_metrics(
            clamped_pred, cropped_target, compute_all=self.config.compute_all_metrics
        )
        metrics["mrae_unclamped"] = compute_mrae(unclamped_pred, cropped_target).item()
        return metrics

    def _save_prediction(self, pred: torch.Tensor, name: str) -> Path:
        save_dir = self.output_dir / "predictions"
        save_dir.mkdir(parents=True, exist_ok=True)
        cube = (
            pred.squeeze(0)
            .clamp(0.0, 1.0)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        if self.config.save_format == "mat":
            import scipy.io as sio

            save_path = save_dir / f"{name}.mat"
            sio.savemat(str(save_path), {"cube": cube})
        elif self.config.save_format == "npy":
            save_path = save_dir / f"{name}.npy"
            np.save(save_path, cube)
        elif self.config.save_format == "h5":
            import h5py

            save_path = save_dir / f"{name}.h5"
            with h5py.File(save_path, "w") as fout:
                fout.create_dataset("cube", data=cube, compression="gzip")
        else:
            raise ValueError(f"Unsupported save_format={self.config.save_format!r}")
        return save_path

    def run(self) -> Dict[str, Any]:
        per_sample: List[Dict[str, Any]] = []
        reference_samples: List[Dict[str, Any]] = []
        scored = 0

        for idx in tqdm(range(len(self.dataset)), desc="ONNX ARAD-1K"):
            sample = self.dataset[idx]
            pred = self._predict(self.model, sample.rgb)

            record: Dict[str, Any] = {
                "index": idx,
                "name": sample.name,
                "prediction_shape": list(pred.shape),
            }

            if sample.target is not None:
                target = sample.target.unsqueeze(0)
                record["metrics"] = self._metrics_for(pred, target)
                scored += 1

                if self.reference_model is not None:
                    reference_pred = self._predict(self.reference_model, sample.rgb)
                    reference_samples.append(
                        {
                            "index": idx,
                            "name": sample.name,
                            "metrics": self._metrics_for(reference_pred, target),
                        }
                    )

            if self.config.save_predictions:
                record["prediction_path"] = str(
                    self._save_prediction(pred, sample.name)
                )

            per_sample.append(record)

        if scored == 0:
            LOGGER.warning(
                "No ground-truth cubes were found; predictions were produced but "
                "no ARAD-1K metrics could be computed."
            )

        results: Dict[str, Any] = {
            "config": asdict(self.config),
            "onnx": {
                "path": str(Path(self.config.onnx_path).resolve()),
                "providers": self.model.providers,
                "input_shape": self.model.input_shape,
                "input_dtype": self.model.input_numpy_dtype.__name__,
                "precision": self.manifest.get("export", {}).get("precision"),
                "clamped_in_graph": self.manifest.get("export", {}).get(
                    "clamp_output"
                ),
                "source_checkpoint": self.manifest.get("checkpoint", {}).get(
                    "source_checkpoint"
                ),
            },
            "split": {
                "resolved": self.dataset.split_name,
                "split_file": str(self.dataset.split_file)
                if self.dataset.split_file
                else None,
                "num_samples": len(self.dataset),
                "num_scored": scored,
            },
            "metrics": _summarize_metrics(per_sample),
            "samples": per_sample,
        }

        if reference_samples:
            results["reference_checkpoint"] = {
                "path": self.config.compare_checkpoint,
                "metrics": _summarize_metrics(reference_samples),
            }
            results["export_delta"] = {
                metric: results["metrics"][metric]["mean"]
                - results["reference_checkpoint"]["metrics"][metric]["mean"]
                for metric in results["metrics"]
                if metric != "count"
                and metric in results["reference_checkpoint"]["metrics"]
            }

        out_path = self.output_dir / "onnx_test_results.json"
        with out_path.open("w", encoding="utf-8") as fout:
            json.dump(results, fout, indent=2, default=str)
        LOGGER.info("Wrote %s", out_path)
        return results


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--onnx", required=True, help="Exported .onnx generator.")
    parser.add_argument(
        "--data_root", required=True, help="ARAD-1K root (contains split_txt/, *_RGB/)."
    )
    parser.add_argument("--output_dir", default="./onnx_test_results")
    parser.add_argument(
        "--split", default="auto", choices=["auto", "test", "valid", "train"]
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        default=None,
        help="ONNX Runtime providers, e.g. CUDAExecutionProvider CPUExecutionProvider. "
        "Defaults to CUDA when available, else CPU.",
    )
    parser.add_argument(
        "--tiling",
        default="auto",
        choices=["auto", "tile", "full"],
        help="auto: tile when the graph's frozen size is smaller than the image.",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=None,
        help="Tile size; defaults to the graph's frozen input size.",
    )
    parser.add_argument("--overlap", type=int, default=16)
    parser.add_argument("--patch_batch_size", type=int, default=4)
    parser.add_argument(
        "--intra_op_threads",
        type=int,
        default=0,
        help="ONNX Runtime intra-op thread count (0 = library default).",
    )
    parser.add_argument("--ensemble_mode", default="none", choices=["none", "d4"])
    parser.add_argument("--no_bgr2rgb", action="store_true")
    parser.add_argument("--rgb_normalization", default="mst", choices=["mst", "uint8"])
    parser.add_argument("--crop_border", type=int, default=128)
    parser.add_argument(
        "--crop_mode", default="arad1k", choices=["arad1k", "border", "none"]
    )
    parser.add_argument("--essential_metrics_only", action="store_true")
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--save_format", default="mat", choices=["mat", "npy", "h5"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument(
        "--allow_missing_gt",
        action="store_true",
        help="Do not fail when some samples have no ground-truth cube.",
    )
    parser.add_argument("--show_patch_progress", action="store_true")
    parser.add_argument(
        "--compare_checkpoint",
        default=None,
        help="Also evaluate the original PyTorch checkpoint and report the delta "
        "the ONNX export introduced.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    config = OnnxTestConfig(
        onnx_path=args.onnx,
        data_root=args.data_root,
        output_dir=args.output_dir,
        split=args.split,
        providers=args.providers,
        tiling=args.tiling,
        patch_size=args.patch_size,
        overlap=args.overlap,
        patch_batch_size=args.patch_batch_size,
        intra_op_threads=args.intra_op_threads,
        ensemble_mode=args.ensemble_mode,
        bgr2rgb=not args.no_bgr2rgb,
        rgb_normalization=args.rgb_normalization,
        crop_border=0 if args.crop_mode == "none" else args.crop_border,
        crop_mode="border" if args.crop_mode == "none" else args.crop_mode,
        compute_all_metrics=not args.essential_metrics_only,
        save_predictions=args.save_predictions,
        save_format=args.save_format,
        max_samples=args.max_samples,
        start_idx=args.start_idx,
        require_gt=not args.allow_missing_gt,
        quiet_patches=not args.show_patch_progress,
        compare_checkpoint=args.compare_checkpoint,
    )

    results = OnnxNTIRETester(config).run()

    summary = results["metrics"]
    print(
        f"\nARAD-1K ({results['split']['resolved']} split, "
        f"{results['split']['num_scored']}/{results['split']['num_samples']} scored, "
        f"crop={config.crop_mode})"
    )
    for metric in ("mrae", "mrae_unclamped", "rmse", "psnr", "sam", "ssim", "mae"):
        if metric in summary:
            print(
                f"  {metric:16s} {summary[metric]['mean']:.6f} "
                f"+/- {summary[metric]['std']:.6f}"
            )

    if "export_delta" in results:
        print("\nONNX minus PyTorch checkpoint (negative = ONNX better)")
        for metric, delta in sorted(results["export_delta"].items()):
            print(f"  {metric:16s} {delta:+.6f}")

    print(f"\nResults: {Path(config.output_dir) / 'onnx_test_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
