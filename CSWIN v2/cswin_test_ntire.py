#!/usr/bin/env python3
"""NTIRE/ARAD-1K testing and submission tool for CSWIN v2.

This mirrors the role of ``mswr_v2/mswr_test_ntire.py`` for the CSWIN v2
generator, but delegates full-image reconstruction to
``hsi_model.utils.patch_inference.PatchInference`` so large images can be
processed without exhausting GPU memory.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import h5py
import numpy as np
import scipy.io as sio
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hsi_model.constants import ARAD1K_NUM_BANDS  # noqa: E402
from hsi_model.utils.inference import (  # noqa: E402
    geometric_self_ensemble,
    load_generator,
)
from hsi_model.utils.metrics import compute_metrics, crop_center_arad1k  # noqa: E402

# ARAD-1K / MST++ center-crop window (NTIRE-2022 scoring region).
_ARAD_CROP_H = 226
_ARAD_CROP_W = 256
from hsi_model.utils.patch_inference import PatchInference  # noqa: E402
from hsi_model.utils.data.mst_dataset import (  # noqa: E402
    _align_hyper_to_rgb,
    _load_mst_cube,
)


LOGGER = logging.getLogger("cswin_test_ntire")

RGB_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
SPEC_SUFFIXES = (".mat",)

RGB_DIRS = {
    "test": ("Test_RGB", "Testing_RGB", "Valid_RGB", "Validation_RGB", "Val_RGB"),
    "valid": ("Valid_RGB", "Validation_RGB", "Val_RGB", "Test_RGB"),
    "train": ("Train_RGB",),
}
SPEC_DIRS = {
    "test": ("Test_Spec", "Testing_Spec", "Valid_Spec", "Validation_Spec", "Val_Spec"),
    "valid": ("Valid_Spec", "Validation_Spec", "Val_Spec", "Test_Spec"),
    "train": ("Train_Spec",),
}
SPLIT_FILES = {
    "test": ("test_list.txt", "testing_list.txt", "test.txt"),
    "valid": ("valid_list.txt", "val_list.txt", "validation_list.txt"),
    "train": ("train_list.txt", "train.txt"),
}


@dataclass
class TestConfig:
    model_path: str
    data_root: str
    output_dir: str = "./cswin_test_results"
    split: str = "auto"
    device: str = "cuda"
    patch_size: int = 128
    overlap: int = 16
    patch_batch_size: int = 4
    use_fp16: bool = True
    prefer_ema: bool = True
    strict_load: bool = True
    ensemble_mode: str = "none"
    bgr2rgb: bool = True
    rgb_normalization: str = "mst"
    crop_border: int = 128
    # 'arad1k' = fixed 226x256 center window (the NTIRE-2022 / MST++ scoring
    # region; size-robust and leaderboard-comparable). 'border' = strip
    # crop_border px from each side (only equals the 226x256 window at the
    # canonical 482x512 ARAD size). Default 'arad1k' is bit-identical to
    # border@128 on 482x512 images but correct for any other size.
    crop_mode: str = "arad1k"
    compute_all_metrics: bool = True
    save_predictions: bool = False
    save_format: str = "mat"
    save_hsi_viz_inputs: bool = False
    max_samples: Optional[int] = None
    start_idx: int = 0
    require_gt: bool = False
    quiet_patches: bool = False


@dataclass
class NTIRESample:
    name: str
    rgb: torch.Tensor
    target: Optional[torch.Tensor]


def _read_split_stems(split_path: Path) -> List[str]:
    with split_path.open("r", encoding="utf-8") as fin:
        return [Path(line.strip()).stem for line in fin if line.strip()]


def _find_existing_file(
    data_root: Path,
    stem: str,
    subdirs: Sequence[str],
    suffixes: Sequence[str],
) -> Optional[Path]:
    for subdir in subdirs:
        directory = data_root / subdir
        if not directory.exists():
            continue
        for suffix in suffixes:
            candidate = directory / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
    return None


def _discover_stems_from_rgb(data_root: Path, split_name: str) -> List[str]:
    stems: List[str] = []
    for subdir in RGB_DIRS[split_name]:
        directory = data_root / subdir
        if not directory.exists():
            continue
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in RGB_SUFFIXES:
                stems.append(path.stem)
        if stems:
            LOGGER.info("Discovered %d RGB files from %s", len(stems), directory)
            return stems
    return stems


def _resolve_split(data_root: Path, requested_split: str) -> Tuple[str, List[str], Optional[Path]]:
    split_dir = data_root / "split_txt"
    split_order = ("test", "valid") if requested_split == "auto" else (requested_split,)

    for split_name in split_order:
        if split_name not in SPLIT_FILES:
            raise ValueError(f"Unsupported split {split_name!r}")
        if split_dir.exists():
            for split_file in SPLIT_FILES[split_name]:
                candidate = split_dir / split_file
                if candidate.exists():
                    stems = _read_split_stems(candidate)
                    if stems:
                        return split_name, stems, candidate
        stems = _discover_stems_from_rgb(data_root, split_name)
        if stems:
            return split_name, stems, None

    raise FileNotFoundError(
        f"Could not resolve split={requested_split!r} under {data_root}. "
        "Expected split_txt/{test,valid}_list.txt or RGB files in a known split directory."
    )


def _load_rgb(path: Path, bgr2rgb: bool, normalization: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read RGB image: {path}")
    if bgr2rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    rgb = image.astype(np.float32)
    if normalization == "mst":
        rgb_min = float(np.min(rgb))
        rgb_max = float(np.max(rgb))
        rgb_range = rgb_max - rgb_min
        if np.isfinite(rgb_range) and rgb_range >= 1e-12:
            rgb = (rgb - rgb_min) / rgb_range
        else:
            LOGGER.warning("Degenerate RGB range for %s; using zeros.", path)
            rgb = np.zeros_like(rgb, dtype=np.float32)
    elif normalization == "uint8":
        rgb = rgb / 255.0
    else:
        raise ValueError(f"Unsupported rgb_normalization={normalization!r}")

    rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
    return np.transpose(rgb, (2, 0, 1)).astype(np.float32, copy=False)


def _load_target(path: Path, rgb_hw: Tuple[int, int]) -> np.ndarray:
    hsi = _load_mst_cube(path)
    hsi = _align_hyper_to_rgb(hsi, rgb_hw, path)
    if hsi.shape[0] != ARAD1K_NUM_BANDS:
        raise ValueError(f"Expected {ARAD1K_NUM_BANDS} HSI bands in {path}, got {hsi.shape}")
    return np.nan_to_num(hsi.astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=0.0)


class NTIRESplitDataset:
    """Split-list dataset for MST++/ARAD-style RGB and optional HSI files."""

    def __init__(
        self,
        data_root: str,
        split: str = "auto",
        bgr2rgb: bool = True,
        rgb_normalization: str = "mst",
        max_samples: Optional[int] = None,
        start_idx: int = 0,
        require_gt: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        self.bgr2rgb = bgr2rgb
        self.rgb_normalization = rgb_normalization
        self.require_gt = require_gt

        split_name, stems, split_file = _resolve_split(self.data_root, split)
        if start_idx < 0:
            raise ValueError("start_idx must be non-negative")
        if max_samples is not None and max_samples < 0:
            raise ValueError("max_samples must be non-negative")

        end_idx = None if max_samples is None else start_idx + max_samples
        self.stems = stems[start_idx:end_idx]
        self.split_name = split_name
        self.split_file = split_file

        self.pairs: List[Tuple[str, Path, Optional[Path]]] = []
        missing_rgb: List[str] = []
        missing_gt: List[str] = []
        for stem in self.stems:
            rgb_path = _find_existing_file(self.data_root, stem, RGB_DIRS[split_name], RGB_SUFFIXES)
            if rgb_path is None:
                missing_rgb.append(stem)
                continue
            target_path = _find_existing_file(self.data_root, stem, SPEC_DIRS[split_name], SPEC_SUFFIXES)
            if target_path is None:
                missing_gt.append(stem)
                if require_gt:
                    continue
            self.pairs.append((stem, rgb_path, target_path))

        if missing_rgb:
            LOGGER.warning("Missing RGB files for %d stems; first few: %s", len(missing_rgb), missing_rgb[:5])
        if missing_gt:
            LOGGER.info("Ground truth missing for %d stems; metrics will be skipped for those.", len(missing_gt))
        if require_gt and missing_gt:
            raise FileNotFoundError(f"Missing ground truth for {len(missing_gt)} samples; first few: {missing_gt[:5]}")
        if not self.pairs:
            raise FileNotFoundError(f"No usable samples found in {self.data_root} for split={split_name}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> NTIRESample:
        stem, rgb_path, target_path = self.pairs[idx]
        rgb_np = _load_rgb(rgb_path, self.bgr2rgb, self.rgb_normalization)
        rgb = torch.from_numpy(rgb_np)

        target: Optional[torch.Tensor] = None
        if target_path is not None:
            target_np = _load_target(target_path, rgb_hw=(rgb_np.shape[1], rgb_np.shape[2]))
            target = torch.from_numpy(target_np)

        return NTIRESample(name=stem, rgb=rgb, target=target)


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def _crop_for_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    crop_border: int,
    crop_mode: str = "border",
) -> Tuple[torch.Tensor, torch.Tensor]:
    if pred.shape != target.shape:
        raise ValueError(f"Metric shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}")

    mode = str(crop_mode).strip().lower()
    if mode in ("arad1k", "arad", "mst", "center", "centre"):
        _, _, height, width = pred.shape
        if height >= _ARAD_CROP_H and width >= _ARAD_CROP_W:
            # Fixed 226x256 center window — identical to border-crop 128 at the
            # canonical 482x512 ARAD size, protocol-correct at any other size.
            return (
                crop_center_arad1k(pred, _ARAD_CROP_H, _ARAD_CROP_W),
                crop_center_arad1k(target, _ARAD_CROP_H, _ARAD_CROP_W),
            )
        LOGGER.warning(
            "Image %dx%d is smaller than the ARAD-1K crop %dx%d; falling back "
            "to border crop.",
            height, width, _ARAD_CROP_H, _ARAD_CROP_W,
        )

    if crop_border <= 0:
        return pred, target

    _, _, height, width = pred.shape
    if height <= 2 * crop_border or width <= 2 * crop_border:
        LOGGER.warning(
            "Skipping crop_border=%d for small tensor shape %s.",
            crop_border,
            tuple(pred.shape),
        )
        return pred, target
    return (
        pred[:, :, crop_border:-crop_border, crop_border:-crop_border],
        target[:, :, crop_border:-crop_border, crop_border:-crop_border],
    )


def _mean_std(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}


def _summarize_metrics(per_sample: List[Dict[str, Any]]) -> Dict[str, Any]:
    metric_names = sorted(
        {metric for sample in per_sample for metric in sample.get("metrics", {}).keys()}
    )
    summary: Dict[str, Any] = {"count": len(per_sample)}
    for metric in metric_names:
        values = [
            float(sample["metrics"][metric])
            for sample in per_sample
            if metric in sample.get("metrics", {})
        ]
        summary[metric] = _mean_std(values)
    return summary


class CSWINNTIRETester:
    def __init__(self, config: TestConfig) -> None:
        self.config = config
        self.device = _resolve_device(config.device)
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
        self.generator, self.checkpoint_info = load_generator(
            config.model_path,
            device=self.device,
            prefer_ema=config.prefer_ema,
            strict=config.strict_load,
        )
        self.patch_infer = PatchInference(
            model=self.generator,
            patch_size=config.patch_size,
            overlap=config.overlap,
            batch_size=config.patch_batch_size,
            device=self.device,
            use_fp16=(config.use_fp16 and self.device.type == "cuda"),
            apply_sigmoid=False,
        )

        n_params = sum(p.numel() for p in self.generator.parameters())
        LOGGER.info("Loaded CSWIN generator with %d parameters.", n_params)
        LOGGER.info(
            "Dataset split=%s samples=%d split_file=%s",
            self.dataset.split_name,
            len(self.dataset),
            self.dataset.split_file,
        )

    def _predict(self, rgb: torch.Tensor) -> torch.Tensor:
        batched = rgb.unsqueeze(0)
        if self.config.ensemble_mode == "d4":
            pred = geometric_self_ensemble(
                lambda x: self.patch_infer.predict(x, show_progress=False),
                batched,
            )
        else:
            pred = self.patch_infer.predict(
                batched,
                show_progress=not self.config.quiet_patches,
            )
        return pred.float()

    def _save_prediction(self, pred: torch.Tensor, name: str) -> Path:
        save_dir = self.output_dir / "predictions"
        save_dir.mkdir(parents=True, exist_ok=True)

        cube = (
            pred.squeeze(0)
            .clamp(0.0, 1.0)
            .float()
            .permute(1, 2, 0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        if self.config.save_format == "mat":
            save_path = save_dir / f"{name}.mat"
            sio.savemat(str(save_path), {"cube": cube})
        elif self.config.save_format == "npy":
            save_path = save_dir / f"{name}.npy"
            np.save(save_path, cube)
        elif self.config.save_format == "h5":
            save_path = save_dir / f"{name}.h5"
            with h5py.File(save_path, "w") as fout:
                fout.create_dataset("cube", data=cube, compression="gzip")
        else:
            raise ValueError(f"Unsupported save_format={self.config.save_format!r}")
        return save_path

    def _save_hsi_viz_input(
        self,
        pred: torch.Tensor,
        target: Optional[torch.Tensor],
        name: str,
        metrics: Dict[str, float],
    ) -> None:
        hsi_dir = self.output_dir / "hsi"
        metrics_dir = self.output_dir / "metrics"
        hsi_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        pred_chw = pred.squeeze(0).clamp(0.0, 1.0).float().cpu().numpy().astype(np.float32)
        np.save(hsi_dir / f"{name}.npy", pred_chw)
        if target is not None:
            np.save(hsi_dir / f"{name}_target.npy", target.float().cpu().numpy().astype(np.float32))
        if metrics:
            with (metrics_dir / f"{name}_metrics.json").open("w", encoding="utf-8") as fout:
                json.dump(metrics, fout, indent=2)

    def run(self) -> Dict[str, Any]:
        per_sample: List[Dict[str, Any]] = []

        for idx in tqdm(range(len(self.dataset)), desc="Testing CSWIN"):
            sample = self.dataset[idx]
            pred = self._predict(sample.rgb)

            sample_result: Dict[str, Any] = {
                "index": idx,
                "name": sample.name,
                "prediction_shape": list(pred.shape),
            }

            if sample.target is not None:
                target = sample.target.unsqueeze(0).to(self.device)
                pred_for_metrics = pred.clamp(0.0, 1.0)
                target_for_metrics = target.float()
                pred_for_metrics, target_for_metrics = _crop_for_metrics(
                    pred_for_metrics,
                    target_for_metrics,
                    self.config.crop_border,
                    self.config.crop_mode,
                )
                metrics = compute_metrics(
                    pred_for_metrics,
                    target_for_metrics,
                    compute_all=self.config.compute_all_metrics,
                )
                sample_result["metrics"] = metrics

            if self.config.save_predictions:
                sample_result["prediction_path"] = str(self._save_prediction(pred, sample.name))

            if self.config.save_hsi_viz_inputs:
                self._save_hsi_viz_input(
                    pred,
                    sample.target,
                    sample.name,
                    sample_result.get("metrics", {}),
                )

            per_sample.append(sample_result)

            # Releasing the allocator cache every image forces a device sync and
            # a re-cudaMalloc on the next image (allocator thrash). Steady-state
            # tile inference does not grow memory, so reclaim only periodically.
            if torch.cuda.is_available() and (idx + 1) % 25 == 0:
                torch.cuda.empty_cache()

        results: Dict[str, Any] = {
            "config": asdict(self.config),
            "split": {
                "resolved": self.dataset.split_name,
                "split_file": str(self.dataset.split_file) if self.dataset.split_file else None,
                "num_samples": len(self.dataset),
            },
            "checkpoint": {
                "epoch": self.checkpoint_info.get("epoch"),
                "ema_applied": self.checkpoint_info.get("ema_applied"),
                "output_activation": self.checkpoint_info.get("output_activation"),
                "val_metrics": self.checkpoint_info.get("val_metrics"),
            },
            "metrics": _summarize_metrics(per_sample),
            "samples": per_sample,
        }

        out_path = self.output_dir / "test_results.json"
        with out_path.open("w", encoding="utf-8") as fout:
            json.dump(results, fout, indent=2, default=str)
        LOGGER.info("Saved results to %s", out_path)
        self._print_summary(results)
        return results

    @staticmethod
    def _print_summary(results: Dict[str, Any]) -> None:
        print("\n" + "=" * 60)
        print("CSWIN v2 NTIRE TEST SUMMARY")
        print("=" * 60)
        split_info = results["split"]
        print(f"Split: {split_info['resolved']} ({split_info['num_samples']} samples)")
        metrics = results.get("metrics", {})
        metric_count = int(metrics.get("count", 0))
        if metric_count > 0 and any(k in metrics for k in ("mrae", "rmse", "psnr")):
            print("\nMetrics:")
            for key in ("mrae", "rmse", "psnr", "ssim", "sam", "mae"):
                if key in metrics:
                    val = metrics[key]
                    print(f"{key.upper():8s}: {val['mean']:.6f} +/- {val['std']:.6f}")
        else:
            print("\nMetrics: no ground truth available.")
        print("=" * 60)


def parse_args() -> TestConfig:
    parser = argparse.ArgumentParser(
        description="NTIRE/ARAD-1K patch inference tester for CSWIN v2."
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to CSWIN generator checkpoint.")
    parser.add_argument("--data_root", type=str, required=True, help="ARAD/MST-style data root.")
    parser.add_argument("--output_dir", type=str, default="./cswin_test_results")
    parser.add_argument("--split", type=str, default="auto", choices=["auto", "test", "valid", "train"])
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--overlap", type=int, default=16)
    parser.add_argument("--patch_batch_size", type=int, default=4)
    parser.add_argument("--use_fp16", dest="use_fp16", action="store_true", default=True)
    parser.add_argument("--no_use_fp16", dest="use_fp16", action="store_false")
    parser.add_argument("--prefer_ema", dest="prefer_ema", action="store_true", default=True)
    parser.add_argument("--no_prefer_ema", dest="prefer_ema", action="store_false")
    parser.add_argument("--strict_load", dest="strict_load", action="store_true", default=True)
    parser.add_argument("--non_strict_load", dest="strict_load", action="store_false")
    parser.add_argument("--ensemble_mode", type=str, default="none", choices=["none", "d4"])

    parser.add_argument("--bgr2rgb", dest="bgr2rgb", action="store_true", default=True)
    parser.add_argument("--no_bgr2rgb", dest="bgr2rgb", action="store_false")
    parser.add_argument("--rgb_normalization", type=str, default="mst", choices=["mst", "uint8"])
    parser.add_argument("--crop_border", type=int, default=128)
    parser.add_argument(
        "--crop_mode", type=str, default="arad1k", choices=["arad1k", "border"],
        help="arad1k: fixed 226x256 center window (leaderboard protocol); "
             "border: strip crop_border px each side (legacy).",
    )
    parser.add_argument("--compute_all_metrics", dest="compute_all_metrics", action="store_true", default=True)
    parser.add_argument("--essential_metrics_only", dest="compute_all_metrics", action="store_false")

    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--save_format", type=str, default="mat", choices=["mat", "npy", "h5"])
    parser.add_argument("--save_hsi_viz_inputs", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--require_gt", action="store_true")
    parser.add_argument("--quiet_patches", action="store_true")
    args = parser.parse_args()
    return TestConfig(**vars(args))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    config = parse_args()
    tester = CSWINNTIRETester(config)
    tester.run()


if __name__ == "__main__":
    main()
