import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from hsi_model.utils.data.hf_arad_dataset import (
    HuggingFaceARADHSDBDataset,
    _sample_key,
)
from hsi_model.utils.data.loaders import create_training_datasets


class _LabelFeature:
    def __init__(self, names):
        self.names = names


class _FakeHFDataset(list):
    def __init__(self, records, label_names):
        super().__init__(records)
        self.features = {"label": _LabelFeature(label_names)}


def _fake_arad_source(target_channels=31):
    label_names = [
        "NTIRE2020_Train_Clean",
        "NTIRE2020_Train_RealWorld",
        "NTIRE2020_Validation_Clean",
        "NTIRE2020_Validation_RealWorld",
    ]

    train_hsi = np.random.rand(8, 8, target_channels).astype(np.float32)
    train_rgb = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
    val_hsi = np.random.rand(8, 8, target_channels).astype(np.float32)
    val_rgb = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)

    return _FakeHFDataset(
        [
            {"image": train_hsi, "label": 0, "filename": "Train_Clean/scene001.tif"},
            {"image": train_rgb, "label": 1, "filename": "Train_RealWorld/scene001.png"},
            {"image": val_hsi, "label": 2, "filename": "Validation_Clean/scene002.tif"},
            {"image": val_rgb, "label": 3, "filename": "Validation_RealWorld/scene002.png"},
        ],
        label_names,
    )


def test_hf_arad_adapter_keeps_mst_sample_contract(monkeypatch):
    fake_source = _fake_arad_source()
    fake_datasets = types.SimpleNamespace(
        load_dataset=lambda dataset_name, split: fake_source
    )
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    train_dataset, val_dataset = create_training_datasets(
        {
            "dataset_source": "huggingface",
            "patch_size": 4,
            "hf_patches_per_image": 2,
            "hf_dataset_name": "mhmdjouni/arad_hsdb",
            "hf_split": "train",
        },
        seed=123,
    )

    assert len(train_dataset) == 2
    assert len(val_dataset) == 1

    rgb_patch, hsi_patch = train_dataset[0]
    assert rgb_patch.shape == (3, 4, 4)
    assert hsi_patch.shape == (31, 4, 4)
    assert np.isfinite(rgb_patch).all()
    assert np.isfinite(hsi_patch).all()

    rgb_full, hsi_full = val_dataset[0]
    assert rgb_full.shape == (3, 8, 8)
    assert hsi_full.shape == (31, 8, 8)


def test_hf_arad_adapter_rejects_rgb_only_targets():
    dataset = HuggingFaceARADHSDBDataset(
        source=_fake_arad_source(target_channels=3),
        training=False,
        include_label_keywords="validation",
    )

    with pytest.raises(ValueError, match="not a 31-band hyperspectral target"):
        dataset[0]


def test_hf_arad_sample_keys_preserve_scene_ids_and_drop_band_suffixes():
    assert _sample_key("Train_Clean/scene001.tif") == "scene001"
    assert _sample_key("Train_Clean/scene001_band_05.png") == "scene001"
    assert _sample_key("ARAD_001_clean.png") == "arad_001"


@pytest.mark.parametrize(
    "script_name",
    ["training_script_fixed.py", "train_optimized.py"],
)
def test_training_entrypoints_show_help_when_run_by_file_path(script_name):
    root = Path(__file__).resolve().parents[1]
    script = root / "src" / "hsi_model" / script_name

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert "config" in result.stdout.lower()
