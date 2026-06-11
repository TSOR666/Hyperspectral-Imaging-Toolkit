#!/usr/bin/env python3
"""Evaluate repository and MST++ model-zoo checkpoints on HSI datasets."""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from hsi_benchmark.data import DatasetOptions, SampleRecord, discover_samples, load_sample
from hsi_benchmark.metrics import compute_hsi_metrics, summarize_metric_rows
from hsi_benchmark.models import (
    MST_METHODS,
    ModelRequest,
    load_model_adapter,
    parse_model_request,
    predict_tiled,
)
from hsi_benchmark.report import (
    save_comparison_figure,
    save_result_arrays,
    save_sample_figure,
    write_json,
    write_method_report,
    write_paper_tables,
)

LOGGER = logging.getLogger("benchmark_hsi")
ROOT = Path(__file__).resolve().parent


def _key_value(value: str, description: str) -> Tuple[str, str]:
    key, separator, item = value.partition("=")
    if not separator or not key.strip() or not item.strip():
        raise ValueError(f"Invalid {description} {value!r}; expected NAME=VALUE")
    return key.strip(), item.strip()


def _parse_datasets(values: Sequence[str]) -> Dict[str, Path]:
    datasets: Dict[str, Path] = {}
    for value in values:
        name, path = _key_value(value, "--dataset")
        datasets[name.lower()] = Path(path)
    return datasets


def _parse_named_paths(values: Optional[Sequence[str]], description: str) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for value in values or []:
        name, path = _key_value(value, description)
        result[name] = Path(path)
    return result


def _device(value: str) -> torch.device:
    requested = torch.device(value)
    if requested.type == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but unavailable; using CPU")
        return torch.device("cpu")
    return requested


def _progress(values: Sequence[Any], description: str) -> Any:
    try:
        from tqdm import tqdm

        return tqdm(values, desc=description)
    except ImportError:
        return values


def _dataset_options(
    name: str,
    root: Path,
    args: argparse.Namespace,
    manifests: Mapping[str, Path],
) -> DatasetOptions:
    source_range = tuple(args.source_range) if args.source_range else None
    target_range = tuple(args.target_range)
    return DatasetOptions(
        preset=name,
        root=root,
        manifest=manifests.get(name),
        rgb_root=args.rgb_root,
        hsi_key=args.hsi_key,
        rgb_key=args.rgb_key,
        rgb_source=args.rgb_source,
        response_file=args.response_file,
        wavelengths_file=args.wavelengths_file,
        source_range=source_range,
        target_range=target_range,
        target_bands=args.target_bands,
        hsi_scale=args.hsi_scale,
        allow_spatial_resize=args.allow_spatial_resize,
    )


def _protocol(
    options: DatasetOptions,
    *,
    crop_border: int,
    tile_size: int,
    overlap: int,
    ensemble: str,
    epsilon: float,
) -> Dict[str, Any]:
    value = asdict(options)
    value.update(
        {
            "root": str(options.root),
            "manifest": str(options.manifest) if options.manifest else None,
            "rgb_root": str(options.rgb_root) if options.rgb_root else None,
            "response_file": str(options.response_file) if options.response_file else None,
            "wavelengths_file": (
                str(options.wavelengths_file) if options.wavelengths_file else None
            ),
            "crop_border": crop_border,
            "tile_size": tile_size,
            "tile_overlap": overlap,
            "ensemble": ensemble,
            "mrae_epsilon": epsilon,
        }
    )
    return value


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate_method_dataset(
    request: ModelRequest,
    adapter: Any,
    *,
    dataset_name: str,
    options: DatasetOptions,
    records: Sequence[SampleRecord],
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    method_dir = output_dir / dataset_name / request.name
    sample_rows: List[Dict[str, Any]] = []
    per_band_rows: List[Dict[str, Any]] = []
    metadata_rows: List[Dict[str, Any]] = []

    for index, record in enumerate(
        _progress(records, f"{dataset_name}/{request.name}")
    ):
        try:
            sample = load_sample(record, options)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            _sync(device)
            started = time.perf_counter()
            prediction_tensor = predict_tiled(
                adapter,
                sample.rgb,
                tile_size=args.tile_size,
                overlap=args.tile_overlap,
                tile_batch_size=args.tile_batch_size,
                ensemble=args.ensemble,
            )
            _sync(device)
            runtime = time.perf_counter() - started
            prediction = prediction_tensor.numpy().astype(np.float32)
            if prediction.shape != sample.target.shape:
                raise ValueError(
                    f"Output shape {prediction.shape} does not match resampled target "
                    f"{sample.target.shape}"
                )
            metrics, details = compute_hsi_metrics(
                prediction,
                sample.target,
                epsilon=args.mrae_epsilon,
                crop_border=args.crop_border,
            )
            height, width = prediction.shape[-2:]
            peak_memory = (
                torch.cuda.max_memory_allocated(device) / 1024**2
                if device.type == "cuda"
                else 0.0
            )
            row: Dict[str, Any] = {
                "dataset": dataset_name,
                "method": request.name,
                "sample": sample.name,
                **metrics,
                "runtime_s": runtime,
                "megapixels_per_s": (height * width / 1e6) / max(runtime, 1e-12),
                "peak_gpu_memory_mb": peak_memory,
                "height": height,
                "width": width,
            }
            sample_rows.append(row)
            metadata_rows.append({"sample": sample.name, **sample.metadata})
            for band, wavelength in enumerate(sample.wavelengths):
                per_band_rows.append(
                    {
                        "dataset": dataset_name,
                        "method": request.name,
                        "sample": sample.name,
                        "band": band,
                        "wavelength_nm": float(wavelength),
                        **{
                            metric: float(details[metric][band])
                            for metric in ("mrae", "rmse", "psnr", "mae")
                        },
                    }
                )

            if not args.no_save_arrays:
                save_result_arrays(method_dir, sample, prediction, metrics)
            if index < args.figures:
                save_sample_figure(
                    method_dir / "figures" / sample.name,
                    sample,
                    prediction,
                    details,
                    metrics,
                    dpi=args.dpi,
                )
        except Exception:
            LOGGER.exception(
                "Failed sample %s for %s/%s", record.name, dataset_name, request.name
            )
            if not args.continue_on_error:
                raise

    metric_rows = [
        {
            key: float(value)
            for key, value in row.items()
            if key
            in {
                "mrae",
                "rmse",
                "psnr",
                "sam",
                "ssim",
                "mae",
                "runtime_s",
                "megapixels_per_s",
                "peak_gpu_memory_mb",
            }
        }
        for row in sample_rows
    ]
    summary = summarize_metric_rows(
        metric_rows,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    write_json(method_dir / "sample_metadata.json", metadata_rows)
    return write_method_report(
        method_dir,
        dataset_name=dataset_name,
        method_name=request.name,
        model_info=adapter.describe(),
        protocol=_protocol(
            options,
            crop_border=args.crop_border,
            tile_size=args.tile_size,
            overlap=args.tile_overlap,
            ensemble=args.ensemble,
            epsilon=args.mrae_epsilon,
        ),
        sample_rows=sample_rows,
        per_band_rows=per_band_rows,
        summary=summary,
    )


def make_comparison_figures(
    *,
    dataset_name: str,
    options: DatasetOptions,
    records: Sequence[SampleRecord],
    requests: Sequence[ModelRequest],
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    if args.figures <= 0 or args.no_save_arrays or len(requests) < 2:
        return
    for record in records[: args.figures]:
        sample = load_sample(record, options)
        predictions: Dict[str, np.ndarray] = {}
        maps: Dict[str, Mapping[str, np.ndarray]] = {}
        for request in requests:
            path = output_dir / dataset_name / request.name / "hsi" / f"{sample.name}.npy"
            if not path.exists():
                continue
            prediction = np.load(path)
            _, details = compute_hsi_metrics(
                prediction,
                sample.target,
                epsilon=args.mrae_epsilon,
                crop_border=args.crop_border,
            )
            predictions[request.name] = prediction
            maps[request.name] = details
        if len(predictions) >= 2:
            save_comparison_figure(
                output_dir / dataset_name / "comparison_figures" / sample.name,
                sample,
                predictions,
                maps,
                dpi=args.dpi,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unified RGB-to-HSI checkpoint evaluation with BGU/ICVL/CAVE "
            "presets and paper-ready reports."
        )
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="NAME=TYPE@CHECKPOINT",
        help=(
            "Repeat for model comparison. TYPE: auto, cswin, mswr[:size], "
            "hsifusion[:size], sharp[:size], wavediff[:type], or mst:method."
        ),
    )
    parser.add_argument(
        "--model-config",
        action="append",
        metavar="NAME=JSON",
        help="Architecture config for checkpoints that do not embed one.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        metavar="PRESET=ROOT",
        help="Repeat to evaluate cave, icvl, bgu, or custom dataset roots.",
    )
    parser.add_argument(
        "--manifest",
        action="append",
        metavar="PRESET=CSV",
        help="Optional manifest with name,hsi,rgb columns for a dataset.",
    )
    parser.add_argument("--output", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--mst-root", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trust-checkpoint", action="store_true")
    parser.add_argument("--allow-partial-load", action="store_true")
    parser.add_argument("--no-prefer-ema", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--rgb-normalization",
        choices=["auto", "mst", "unit", "wavediff"],
        default="auto",
    )

    parser.add_argument("--rgb-root", type=Path)
    parser.add_argument("--hsi-key")
    parser.add_argument("--rgb-key")
    parser.add_argument(
        "--rgb-source",
        choices=["auto", "paired", "cie", "response"],
        default="auto",
    )
    parser.add_argument("--response-file", type=Path)
    parser.add_argument("--wavelengths-file", type=Path)
    parser.add_argument("--source-range", type=float, nargs=2, metavar=("MIN", "MAX"))
    parser.add_argument(
        "--target-range",
        type=float,
        nargs=2,
        default=(400.0, 700.0),
        metavar=("MIN", "MAX"),
    )
    parser.add_argument("--target-bands", type=int, default=31)
    parser.add_argument(
        "--hsi-scale",
        default="auto",
        help="auto, none, or a numeric divisor applied before metrics.",
    )
    parser.add_argument("--allow-spatial-resize", action="store_true")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--start-index", type=int, default=0)

    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--tile-overlap", type=int, default=32)
    parser.add_argument("--tile-batch-size", type=int, default=1)
    parser.add_argument("--ensemble", choices=["none", "d4"], default="none")
    parser.add_argument("--crop-border", type=int, default=0)
    parser.add_argument("--mrae-epsilon", type=float, default=1e-6)
    parser.add_argument("--sampling-steps", type=int, default=20)
    parser.add_argument("--latent-mode", choices=["direct", "diffusion"], default="direct")

    parser.add_argument("--figures", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no-save-arrays", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _device(args.device)
    requests = [parse_model_request(value) for value in args.model]
    configs = _parse_named_paths(args.model_config, "--model-config")
    requests = [
        replace(request, config_path=configs.get(request.name)) for request in requests
    ]
    if len({request.name for request in requests}) != len(requests):
        raise ValueError("Every --model NAME must be unique")
    datasets = _parse_datasets(args.dataset)
    manifests = _parse_named_paths(args.manifest, "--manifest")
    unknown_manifests = set(manifests) - set(datasets)
    if unknown_manifests:
        raise ValueError(f"Manifest names do not match datasets: {sorted(unknown_manifests)}")

    contexts: Dict[str, Tuple[DatasetOptions, List[SampleRecord]]] = {}
    for name, root in datasets.items():
        options = _dataset_options(name, root, args, manifests)
        records = discover_samples(options)
        if args.start_index < 0:
            raise ValueError("--start-index must be non-negative")
        end = None if args.max_samples is None else args.start_index + args.max_samples
        records = records[args.start_index:end]
        if not records:
            raise ValueError(f"No selected samples remain for dataset {name}")
        contexts[name] = (options, records)
        LOGGER.info("Dataset %s: %d selected samples", name, len(records))

    args.output.mkdir(parents=True, exist_ok=True)
    all_results: List[Dict[str, Any]] = []
    for request in requests:
        LOGGER.info("Loading model %s (%s)", request.name, request.kind)
        adapter = load_model_adapter(
            request,
            repository_root=ROOT,
            device=device,
            mst_root=args.mst_root,
            trust_checkpoint=args.trust_checkpoint,
            allow_partial=args.allow_partial_load,
            prefer_ema=not args.no_prefer_ema,
            use_amp=not args.no_amp,
            normalization_override=args.rgb_normalization,
            sampling_steps=args.sampling_steps,
            latent_mode=args.latent_mode,
        )
        LOGGER.info(
            "Loaded %s with %.2f M parameters from %s",
            request.name,
            adapter.parameter_count / 1e6,
            adapter.state_source,
        )
        for dataset_name, (options, records) in contexts.items():
            result = evaluate_method_dataset(
                request,
                adapter,
                dataset_name=dataset_name,
                options=options,
                records=records,
                output_dir=args.output,
                args=args,
                device=device,
            )
            all_results.append(result)
        del adapter
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for dataset_name, (options, records) in contexts.items():
        make_comparison_figures(
            dataset_name=dataset_name,
            options=options,
            records=records,
            requests=requests,
            output_dir=args.output,
            args=args,
        )
    write_paper_tables(args.output, all_results)
    LOGGER.info("Done. Paper tables and complete results: %s", args.output.resolve())


if __name__ == "__main__":
    main()
