from __future__ import annotations

import importlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)

MST_METHODS = {
    "mst_plus_plus",
    "mst",
    "mirnet",
    "hinet",
    "mprnet",
    "restormer",
    "edsr",
    "hdnet",
    "hrnet",
    "hscnn_plus",
    "awan",
}


@dataclass(frozen=True)
class ModelRequest:
    name: str
    kind: str
    checkpoint: Path
    config_path: Optional[Path] = None


def parse_model_request(value: str) -> ModelRequest:
    name, separator, remainder = value.partition("=")
    if not separator:
        raise ValueError(
            f"Invalid --model {value!r}; expected NAME=TYPE@CHECKPOINT"
        )
    kind, separator, checkpoint = remainder.partition("@")
    if not separator:
        raise ValueError(
            f"Invalid --model {value!r}; expected NAME=TYPE@CHECKPOINT"
        )
    return ModelRequest(name.strip(), kind.strip().lower(), Path(checkpoint.strip()))


def _load_json(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Model config must be a JSON object: {path}")
    return value


def _torch_load(path: Path, *, trust_checkpoint: bool) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception as exc:
        if not trust_checkpoint:
            raise RuntimeError(
                f"Safe checkpoint loading failed for {path}: {exc}. "
                "Pass --trust-checkpoint only for a checkpoint you trust if it "
                "contains pickled configuration objects."
            ) from exc
        LOGGER.warning(
            "Safe loading failed for %s; retrying trusted pickle load: %s", path, exc
        )
        return torch.load(path, map_location="cpu", weights_only=False)


def _mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        return {
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return {}


def _state_candidates(checkpoint: Any, prefer_ema: bool) -> List[Tuple[str, Dict[str, torch.Tensor]]]:
    if not isinstance(checkpoint, Mapping):
        return []
    candidates: List[Tuple[str, Dict[str, torch.Tensor]]] = []

    def add(label: str, value: Any) -> None:
        if not isinstance(value, Mapping):
            return
        tensor_items = {
            str(key): item for key, item in value.items() if isinstance(item, torch.Tensor)
        }
        if tensor_items:
            candidates.append((label, tensor_items))
        for child_key in ("ema_state", "shadow", "model", "state_dict", "model_state_dict"):
            child = value.get(child_key)
            if isinstance(child, Mapping):
                add(f"{label}.{child_key}", child)

    ema_keys = ("ema_model_state_dict", "ema_state_dict", "ema")
    base_keys = ("model_state_dict", "state_dict", "generator_state_dict", "model")
    ordered = (*ema_keys, *base_keys) if prefer_ema else (*base_keys, *ema_keys)
    for key in ordered:
        add(key, checkpoint.get(key))
    add("checkpoint", checkpoint)
    return candidates


def _strip_prefixes(
    state: Mapping[str, torch.Tensor], prefixes: Sequence[str]
) -> Dict[str, torch.Tensor]:
    normalized: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        updated = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if updated.startswith(prefix):
                    updated = updated[len(prefix) :]
                    changed = True
        normalized[updated] = value
    return normalized


def _state_variants(state: Mapping[str, torch.Tensor]) -> Iterable[Dict[str, torch.Tensor]]:
    common = ("module.", "_orig_mod.")
    base = _strip_prefixes(state, common)
    yield base
    for prefix in ("generator.", "model.", "net.", "network."):
        selected = {
            key[len(prefix) :]: value
            for key, value in base.items()
            if key.startswith(prefix)
        }
        if selected:
            yield selected


def _load_state_checked(
    model: nn.Module,
    checkpoint: Any,
    *,
    prefer_ema: bool,
    allow_partial: bool,
) -> str:
    model_state = model.state_dict()
    model_numel = sum(value.numel() for value in model_state.values())
    variants: List[Tuple[str, Dict[str, torch.Tensor]]] = []
    for label, candidate in _state_candidates(checkpoint, prefer_ema):
        variants.extend((label, variant) for variant in _state_variants(candidate))

    if prefer_ema:
        base_variants = [
            (label, state)
            for label, state in variants
            if "ema" not in label.lower()
        ]
        ema_variants = [
            (label, state)
            for label, state in variants
            if "ema" in label.lower()
        ]
        for base_label, base_state in base_variants:
            base_compatible = {
                key: value
                for key, value in base_state.items()
                if key in model_state
                and tuple(value.shape) == tuple(model_state[key].shape)
            }
            if set(base_compatible) != set(model_state):
                continue
            for ema_label, ema_state in ema_variants:
                overlay = {
                    key: value
                    for key, value in ema_state.items()
                    if key in model_state
                    and tuple(value.shape) == tuple(model_state[key].shape)
                }
                if overlay:
                    variants.append(
                        (
                            f"{base_label}+{ema_label}",
                            {**base_compatible, **overlay},
                        )
                    )

    best: Optional[Tuple[int, str, Dict[str, torch.Tensor]]] = None
    for label, variant in variants:
        matched = {
            key: value
            for key, value in variant.items()
            if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
        }
        matched_numel = sum(value.numel() for value in matched.values())
        prefer_overlay = "+" in label and (best is None or "+" not in best[1])
        if best is None or matched_numel > best[0] or (
            matched_numel == best[0] and prefer_overlay
        ):
            best = (matched_numel, label, variant)
    if best is None or best[0] == 0:
        raise RuntimeError("Checkpoint contains no tensor state matching this architecture")
    matched_numel, label, state = best
    coverage = matched_numel / max(1, model_numel)
    exact = (
        set(state) == set(model_state)
        and all(tuple(state[key].shape) == tuple(value.shape) for key, value in model_state.items())
    )
    if exact:
        model.load_state_dict(state, strict=True)
    elif not allow_partial:
        matched_keys = sum(
            key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
            for key, value in state.items()
        )
        raise RuntimeError(
            f"Checkpoint state {label!r} is not an exact architecture match "
            f"({matched_keys}/{len(model_state)} keys, {coverage:.1%} parameters). "
            "Select the correct model type/config or pass --allow-partial-load."
        )
    else:
        if coverage < 0.90:
            raise RuntimeError(
                f"Partial checkpoint coverage is only {coverage:.1%}; refusing inference"
            )
        compatible = {
            key: value
            for key, value in state.items()
            if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
        }
        result = model.load_state_dict(compatible, strict=False)
        LOGGER.warning(
            "Partially loaded %s at %.1f%% parameter coverage; missing=%d unexpected=%d",
            label,
            coverage * 100.0,
            len(result.missing_keys),
            len(result.unexpected_keys),
        )
    return f"{label} ({coverage:.1%} coverage)"


def _insert_path(path: Path) -> None:
    value = str(path.resolve())
    if value not in sys.path:
        sys.path.insert(0, value)


def _import_from_root(root: Path, module: str) -> ModuleType:
    _insert_path(root)
    return importlib.import_module(module)


def _first_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, Mapping):
        for key in ("hsi_output", "prediction", "pred", "output", "final"):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value
        for value in output.values():
            if isinstance(value, torch.Tensor):
                return value
    if isinstance(output, (tuple, list)):
        for value in output:
            if isinstance(value, torch.Tensor):
                return value
    raise TypeError(f"Model returned no tensor prediction: {type(output)}")


class ModelAdapter:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        *,
        name: str,
        kind: str,
        normalization: str = "mst",
        input_multiple: int = 1,
        min_size: int = 1,
        use_amp: bool = True,
        state_source: str = "",
        checkpoint_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.name = name
        self.kind = kind
        self.normalization = normalization
        self.input_multiple = max(1, int(input_multiple))
        self.min_size = max(1, int(min_size))
        self.use_amp = bool(use_amp and device.type == "cuda")
        self.state_source = state_source
        self.checkpoint_info = checkpoint_info or {}

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())

    def preprocess(self, rgb: torch.Tensor) -> torch.Tensor:
        if self.normalization == "mst":
            minimum = rgb.amin(dim=(1, 2, 3), keepdim=True)
            maximum = rgb.amax(dim=(1, 2, 3), keepdim=True)
            rgb = torch.where(
                maximum - minimum > 1e-8,
                (rgb - minimum) / (maximum - minimum).clamp_min(1e-8),
                rgb,
            )
        elif self.normalization == "wavediff":
            rgb = rgb * 2.0 - 1.0
        elif self.normalization != "unit":
            raise ValueError(f"Unknown RGB normalization {self.normalization!r}")
        return rgb

    def _forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return _first_tensor(self.model(rgb))

    def predict_batch(self, rgb: torch.Tensor) -> torch.Tensor:
        rgb = self.preprocess(rgb.to(self.device, non_blocking=True))
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.use_amp,
        ):
            prediction = self._forward(rgb)
        return prediction.float().clamp(0.0, 1.0)

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "parameters": self.parameter_count,
            "normalization": self.normalization,
            "input_multiple": self.input_multiple,
            "min_size": self.min_size,
            "state_source": self.state_source,
            **self.checkpoint_info,
        }


class WaveDiffAdapter(ModelAdapter):
    def __init__(
        self,
        *args: Any,
        config: Mapping[str, Any],
        sampling_steps: int,
        latent_mode: str,
        normalization: str = "wavediff",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, normalization=normalization, **kwargs)
        self.config = dict(config)
        self.sampling_steps = sampling_steps
        self.latent_mode = latent_mode

    def _forward(self, rgb: torch.Tensor) -> torch.Tensor:
        result = self.model.rgb_to_hsi(
            rgb,
            sampling_steps=self.sampling_steps,
            return_stages=False,
            latent_mode=self.latent_mode,
        )
        prediction = _first_tensor(result)
        if self.config.get("hsi_normalize_to_neg_one_to_one", False):
            prediction = (prediction + 1.0) * 0.5
        prediction = prediction * float(self.config.get("hsi_max_value", 1.0) or 1.0)
        return prediction


def _checkpoint_kind(checkpoint: Any) -> str:
    if not isinstance(checkpoint, Mapping):
        return ""
    config = _mapping(checkpoint.get("config"))
    if "model_type" in config and any(
        token in str(config["model_type"]).lower() for token in ("wavelet", "base")
    ):
        return "wavediff"
    keys = {
        key
        for _, state in _state_candidates(checkpoint, prefer_ema=True)
        for key in state
    }
    joined = "\n".join(keys)
    if "output_proj" in joined and "input_skip" in joined:
        return "mswr"
    if "to_spectral" in joined or "spectral_head" in joined and "generator." in joined:
        return "cswin"
    if "uncertainty_head" in joined or "lightning" in str(type(checkpoint.get("config"))).lower():
        return "hsifusion"
    if "rbf" in joined or "sparse_attention" in joined:
        return "sharp"
    return ""


def _resolve_kind(requested: str, checkpoint: Any) -> str:
    if requested != "auto":
        return requested
    detected = _checkpoint_kind(checkpoint)
    if not detected:
        raise ValueError(
            "Could not identify checkpoint architecture. Use an explicit type such "
            "as cswin, mswr:base, hsifusion:base, sharp:base, wavediff, or "
            "mst:mst_plus_plus."
        )
    return detected


def _split_kind(kind: str) -> Tuple[str, Optional[str]]:
    family, separator, variant = kind.partition(":")
    return family, variant if separator else None


def _build_cswin(
    root: Path, config: Mapping[str, Any]
) -> Tuple[nn.Module, int, int]:
    module = _import_from_root(root / "CSWIN v2" / "src", "hsi_model.models.generator_v3")
    model = module.NoiseRobustCSWinGenerator(dict(config))
    return model, 16, 16


def _build_mswr(root: Path, checkpoint: Mapping[str, Any], variant: Optional[str]) -> Tuple[nn.Module, int, int]:
    module = _import_from_root(root / "mswr_v2", "model.mswr_net_v212")
    model_config = checkpoint.get("model_config")
    if isinstance(model_config, Mapping):
        config = module.MSWRDualConfig(**dict(model_config))
        model = module.IntegratedMSWRNet(config)
    else:
        size = variant or "base"
        factories = {
            "tiny": module.create_mswr_tiny,
            "small": module.create_mswr_small,
            "base": module.create_mswr_base,
            "large": module.create_mswr_large,
        }
        if size not in factories:
            raise ValueError(f"Unknown MSWR size {size!r}")
        model = factories[size]()
        config = model.config
    multiple = 2 ** max(1, int(config.num_stages))
    minimum = 2 ** (int(config.num_stages) + 2)
    return model, multiple, minimum


def _build_hsifusion(
    root: Path, checkpoint: Mapping[str, Any], variant: Optional[str]
) -> Tuple[nn.Module, int, int]:
    module = _import_from_root(
        root / "HSIFUSION&SHARP", "hsifusion_v252_complete"
    )
    config = _mapping(checkpoint.get("config"))
    size = variant or str(config.get("model_size", "base"))
    kwargs = {
        "in_channels": int(config.get("in_channels", 3)),
        "out_channels": int(config.get("out_channels", 31)),
        "cross_attention_max_tokens": config.get("cross_attention_max_tokens", 1024),
        "estimate_uncertainty": bool(config.get("estimate_uncertainty", False)),
    }
    model = module.create_hsifusion_lightning_pro(
        model_size=size,
        compile_mode=None,
        lazy_compile=False,
        force_compile=False,
        **kwargs,
    )
    minimum = int(getattr(getattr(model, "config", None), "min_input_size", 64))
    return model, 16, minimum


def _build_sharp(
    root: Path, checkpoint: Mapping[str, Any], variant: Optional[str]
) -> Tuple[nn.Module, int, int]:
    module = _import_from_root(root / "HSIFUSION&SHARP", "sharp_v322_hardened")
    config = _mapping(checkpoint.get("config"))
    size = variant or str(config.get("model_size", "base"))
    names = (
        "sparse_sparsity_ratio",
        "rbf_centers_per_head",
        "sparse_k_cap",
        "sparse_block_size",
        "sparse_q_block_size",
        "sparse_window_size",
        "sparse_max_tokens",
        "max_global_tokens",
        "key_rbf_mode",
        "sparsemax_pad_value",
        "ema_update_every",
    )
    kwargs = {name: config[name] for name in names if name in config}
    model = module.create_sharp_v32(
        model_size=size,
        in_channels=int(config.get("in_channels", 3)),
        out_channels=int(config.get("out_channels", 31)),
        compile_model=False,
        verbose=False,
        **kwargs,
    )
    return model, 16, 16


def _build_mst_zoo(mst_root: Path, method: str) -> Tuple[nn.Module, int, int]:
    if method not in MST_METHODS:
        raise ValueError(f"Unknown MST++ zoo method {method!r}; choices: {sorted(MST_METHODS)}")
    code_root = mst_root / "test_develop_code"
    if not (code_root / "architecture" / "__init__.py").exists():
        raise FileNotFoundError(
            f"Expected official MST++ test_develop_code under {mst_root}"
        )
    existing = sys.modules.get("architecture")
    if existing is not None:
        existing_path = Path(getattr(existing, "__file__", "")).resolve()
        if code_root.resolve() not in existing_path.parents:
            for key in list(sys.modules):
                if key == "architecture" or key.startswith("architecture."):
                    del sys.modules[key]
    architecture = _import_from_root(code_root, "architecture")
    factories = {
        "mirnet": lambda: architecture.MIRNet(n_RRG=3, n_MSRB=1, height=3, width=1),
        "mst_plus_plus": architecture.MST_Plus_Plus,
        "mst": lambda: architecture.MST(dim=31, stage=2, num_blocks=[4, 7, 5]),
        "hinet": lambda: architecture.HINet(depth=4),
        "mprnet": lambda: architecture.MPRNet(num_cab=4),
        "restormer": architecture.Restormer,
        "edsr": architecture.EDSR,
        "hdnet": architecture.HDNet,
        "hrnet": architecture.SGN,
        "hscnn_plus": architecture.HSCNN_Plus,
        "awan": architecture.AWAN,
    }
    return factories[method](), 16, 16


def load_model_adapter(
    request: ModelRequest,
    *,
    repository_root: Path,
    device: torch.device,
    mst_root: Optional[Path],
    trust_checkpoint: bool,
    allow_partial: bool,
    prefer_ema: bool,
    use_amp: bool,
    normalization_override: str,
    sampling_steps: int,
    latent_mode: str,
) -> ModelAdapter:
    if not request.checkpoint.exists():
        raise FileNotFoundError(request.checkpoint)
    checkpoint = _torch_load(request.checkpoint, trust_checkpoint=trust_checkpoint)
    kind = _resolve_kind(request.kind, checkpoint)
    family, variant = _split_kind(kind)
    config_override = _load_json(request.config_path)
    checkpoint_mapping = checkpoint if isinstance(checkpoint, Mapping) else {}

    if family == "cswin":
        config = config_override or _mapping(checkpoint_mapping.get("config"))
        if not config:
            raise ValueError(
                "CSWIN checkpoint has no architecture config; attach one with "
                "--model-config NAME=path.json."
            )
        model, multiple, minimum = _build_cswin(repository_root, config)
    elif family == "mswr":
        model, multiple, minimum = _build_mswr(
            repository_root, checkpoint_mapping, variant
        )
    elif family == "hsifusion":
        model, multiple, minimum = _build_hsifusion(
            repository_root, checkpoint_mapping, variant
        )
    elif family == "sharp":
        model, multiple, minimum = _build_sharp(
            repository_root, checkpoint_mapping, variant
        )
    elif family == "mst":
        if mst_root is None:
            raise ValueError(f"Model {request.name} requires --mst-root")
        model, multiple, minimum = _build_mst_zoo(
            mst_root, variant or "mst_plus_plus"
        )
    elif family == "wavediff":
        module = _import_from_root(repository_root / "WaveDiff", "inference")
        model, config = module.load_model(
            str(request.checkpoint),
            device=torch.device("cpu"),
            model_type=variant,
            use_ema_weights=prefer_ema,
        )
        normalization = (
            normalization_override
            if normalization_override != "auto"
            else "wavediff"
        )
        return WaveDiffAdapter(
            model,
            device,
            name=request.name,
            kind=kind,
            input_multiple=16,
            min_size=16,
            use_amp=use_amp,
            normalization=normalization,
            state_source="WaveDiff loader",
            checkpoint_info={"checkpoint": str(request.checkpoint)},
            config=config,
            sampling_steps=sampling_steps,
            latent_mode=latent_mode,
        )
    else:
        raise ValueError(f"Unsupported model family {family!r}")

    state_source = _load_state_checked(
        model,
        checkpoint,
        prefer_ema=prefer_ema,
        allow_partial=allow_partial,
    )
    default_normalization = "mst"
    normalization = (
        normalization_override
        if normalization_override != "auto"
        else default_normalization
    )
    return ModelAdapter(
        model,
        device,
        name=request.name,
        kind=kind,
        normalization=normalization,
        input_multiple=multiple,
        min_size=minimum,
        use_amp=use_amp,
        state_source=state_source,
        checkpoint_info={
            "checkpoint": str(request.checkpoint),
            "epoch": checkpoint_mapping.get("epoch"),
            "best_mrae": checkpoint_mapping.get("best_mrae"),
        },
    )


def _pad_for_model(
    batch: torch.Tensor, multiple: int, min_size: int
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    height, width = batch.shape[-2:]
    target_h = max(min_size, ((height + multiple - 1) // multiple) * multiple)
    target_w = max(min_size, ((width + multiple - 1) // multiple) * multiple)
    pad_h = target_h - height
    pad_w = target_w - width
    if pad_h or pad_w:
        mode = (
            "reflect"
            if height > 1
            and width > 1
            and pad_h < height
            and pad_w < width
            else "replicate"
        )
        batch = F.pad(batch, (0, pad_w, 0, pad_h), mode=mode)
    return batch, (height, width)


def _tile_starts(length: int, tile_size: int, overlap: int) -> List[int]:
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    starts = list(range(0, max(1, length - tile_size + 1), stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _blend_window(height: int, width: int) -> torch.Tensor:
    y = torch.hann_window(height, periodic=False) if height > 1 else torch.ones(1)
    x = torch.hann_window(width, periodic=False) if width > 1 else torch.ones(1)
    return torch.outer(y, x).clamp_min(1e-3).unsqueeze(0)


def predict_tiled(
    adapter: ModelAdapter,
    rgb_chw: np.ndarray | torch.Tensor,
    *,
    tile_size: int,
    overlap: int,
    tile_batch_size: int,
    ensemble: str = "none",
) -> torch.Tensor:
    rgb = torch.as_tensor(rgb_chw, dtype=torch.float32)
    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise ValueError(f"Expected RGB CHW input, got {tuple(rgb.shape)}")
    if tile_size <= 0:
        tile_size = max(rgb.shape[-2:])
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must satisfy 0 <= overlap < tile_size")

    def predict_once(image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        effective_tile = max(tile_size, adapter.min_size)
        y_starts = _tile_starts(height, effective_tile, overlap)
        x_starts = _tile_starts(width, effective_tile, overlap)
        entries: List[Tuple[int, int, int, int, torch.Tensor]] = []
        for y in y_starts:
            for x in x_starts:
                tile = image[:, y : y + effective_tile, x : x + effective_tile]
                tile_h, tile_w = tile.shape[-2:]
                tile, _ = _pad_for_model(
                    tile.unsqueeze(0), adapter.input_multiple, adapter.min_size
                )
                entries.append((y, x, tile_h, tile_w, tile.squeeze(0)))

        output: Optional[torch.Tensor] = None
        weights = torch.zeros(1, height, width, dtype=torch.float32)
        for start in range(0, len(entries), max(1, tile_batch_size)):
            chunk = entries[start : start + max(1, tile_batch_size)]
            shapes = {tuple(entry[4].shape) for entry in chunk}
            if len(shapes) != 1:
                batches = [[entry] for entry in chunk]
            else:
                batches = [chunk]
            for group in batches:
                batch = torch.stack([entry[4] for entry in group])
                predictions = adapter.predict_batch(batch).cpu()
                for prediction, (y, x, tile_h, tile_w, _) in zip(predictions, group):
                    prediction = prediction[:, :tile_h, :tile_w]
                    if prediction.shape[-2:] != (tile_h, tile_w):
                        raise RuntimeError(
                            f"{adapter.name} changed spatial size from {(tile_h, tile_w)} "
                            f"to {tuple(prediction.shape[-2:])}"
                        )
                    if output is None:
                        output = torch.zeros(
                            prediction.shape[0], height, width, dtype=torch.float32
                        )
                    blend = _blend_window(tile_h, tile_w)
                    output[:, y : y + tile_h, x : x + tile_w] += prediction * blend
                    weights[:, y : y + tile_h, x : x + tile_w] += blend
        if output is None:
            raise RuntimeError("No inference tiles were produced")
        return output / weights.clamp_min(1e-8)

    if ensemble == "none":
        return predict_once(rgb).clamp(0.0, 1.0)
    if ensemble != "d4":
        raise ValueError("ensemble must be none or d4")
    predictions = []
    for flip in (False, True):
        for rotation in range(4):
            transformed = torch.flip(rgb, dims=[-1]) if flip else rgb
            transformed = torch.rot90(transformed, rotation, dims=(-2, -1))
            prediction = predict_once(transformed)
            prediction = torch.rot90(prediction, -rotation, dims=(-2, -1))
            if flip:
                prediction = torch.flip(prediction, dims=[-1])
            predictions.append(prediction)
    return torch.stack(predictions).mean(dim=0).clamp(0.0, 1.0)
