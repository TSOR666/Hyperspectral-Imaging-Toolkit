import importlib.util
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
h5py = pytest.importorskip("h5py")


def _load_tool_module():
    tool_path = Path(__file__).resolve().parents[1] / "cswin_test_ntire.py"
    spec = importlib.util.spec_from_file_location("cswin_test_ntire", tool_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_ntire_dataset_auto_prefers_test_rgb_without_gt(tmp_path):
    tool = _load_tool_module()
    case_dir = tmp_path / f"case_{uuid.uuid4().hex}"
    rgb_dir = case_dir / "Test_RGB"
    split_dir = case_dir / "split_txt"
    rgb_dir.mkdir(parents=True)
    split_dir.mkdir()

    scene = "ARAD_1K_0951"
    (split_dir / "test_list.txt").write_text(f"{scene}\n", encoding="utf-8")
    rgb = np.random.randint(0, 255, (4, 5, 3), dtype=np.uint8)
    assert cv2.imwrite(str(rgb_dir / f"{scene}.jpg"), rgb)

    dataset = tool.NTIRESplitDataset(data_root=str(case_dir), split="auto")
    sample = dataset[0]

    assert dataset.split_name == "test"
    assert sample.name == scene
    assert sample.target is None
    assert tuple(sample.rgb.shape) == (3, 4, 5)
    assert np.isfinite(sample.rgb.numpy()).all()


def test_ntire_dataset_valid_loads_mst_hsi_orientation(tmp_path):
    tool = _load_tool_module()
    case_dir = tmp_path / f"case_{uuid.uuid4().hex}"
    rgb_dir = case_dir / "Valid_RGB"
    spec_dir = case_dir / "Valid_Spec"
    split_dir = case_dir / "split_txt"
    rgb_dir.mkdir(parents=True)
    spec_dir.mkdir()
    split_dir.mkdir()

    scene = "ARAD_1K_0901"
    (split_dir / "valid_list.txt").write_text(f"{scene}\n", encoding="utf-8")

    rgb = np.random.randint(0, 255, (4, 5, 3), dtype=np.uint8)
    assert cv2.imwrite(str(rgb_dir / f"{scene}.jpg"), rgb)
    hyper = np.random.rand(31, 5, 4).astype(np.float32)
    with h5py.File(spec_dir / f"{scene}.mat", "w") as mat:
        mat.create_dataset("cube", data=hyper)

    dataset = tool.NTIRESplitDataset(data_root=str(case_dir), split="valid")
    sample = dataset[0]

    assert dataset.split_name == "valid"
    assert tuple(sample.rgb.shape) == (3, 4, 5)
    assert sample.target is not None
    assert tuple(sample.target.shape) == (31, 4, 5)
    assert np.isfinite(sample.target.numpy()).all()
