"""Tests for the canonical MSWR training config layout."""

import sys
from pathlib import Path

import yaml

import train_mswr_v212_logging as trainer


ROOT = Path(__file__).resolve().parents[1]


def test_default_launch_uses_canonical_train_config(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_mswr_v212_logging.py"])

    args = trainer.load_config(trainer.parse_arguments())

    assert Path(args.config).resolve() == (ROOT / "configs" / "train.yaml").resolve()
    assert args.use_spectral_attn is True
    assert args.wavelet_levels == [1, 1, 1]
    assert args.use_ema is True


def test_only_one_top_level_training_yaml_exists():
    top_level_yamls = sorted(path.name for path in (ROOT / "configs").glob("*.yaml"))

    assert top_level_yamls == ["train.yaml"]


def test_all_shipped_config_keys_are_recognized(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_mswr_v212_logging.py"])
    parser_args = trainer.parse_arguments()
    known_keys = set(vars(parser_args))

    config_paths = [ROOT / "configs" / "train.yaml"]
    config_paths.extend(sorted((ROOT / "configs" / "experiments").glob("*.yaml")))

    for config_path in config_paths:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        unknown = set(config) - known_keys
        assert not unknown, f"{config_path.name} has unknown keys: {sorted(unknown)}"


def test_spectralffn_experiment_plumbs_architecture_knobs(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_mswr_v212_logging.py",
            "--config",
            str(ROOT / "configs" / "experiments" / "robust_mrae_spectralffn.yaml"),
        ],
    )

    args = trainer.load_config(trainer.parse_arguments())

    assert args.use_spectral_attn is True
    assert args.spectral_attn_heads == 1
    assert args.spectral_ffn is True
    assert args.spectral_ffn_mult == 2
    assert args.wavelet_detail_processing is True
    assert args.cache_dtype == "float32"
    assert args.use_enhanced_loss is False
