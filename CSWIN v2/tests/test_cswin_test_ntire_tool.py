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


def test_ntire_mst_rgb_normalization_matches_training_loader(tmp_path):
    tool = _load_tool_module()
    from hsi_model.utils.data.mst_dataset import _load_normalized_rgb

    rgb_path = tmp_path / "scene.png"
    rgb = np.array(
        [
            [[0, 20, 40], [60, 80, 100], [120, 140, 160]],
            [[180, 200, 220], [240, 250, 255], [10, 30, 50]],
        ],
        dtype=np.uint8,
    )
    assert cv2.imwrite(str(rgb_path), rgb)

    ntire_rgb = tool._load_rgb(rgb_path, bgr2rgb=True, normalization="mst")
    train_rgb = _load_normalized_rgb(rgb_path, bgr2rgb=True)

    assert ntire_rgb.dtype == np.float32
    assert np.array_equal(ntire_rgb, train_rgb)


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


def test_ntire_dataset_auto_falls_back_to_validation_when_test_gt_is_missing(tmp_path):
    tool = _load_tool_module()
    case_dir = tmp_path / f"case_{uuid.uuid4().hex}"
    test_rgb_dir = case_dir / "Test_RGB"
    valid_rgb_dir = case_dir / "Valid_RGB"
    valid_spec_dir = case_dir / "Valid_Spec"
    split_dir = case_dir / "split_txt"
    for directory in (test_rgb_dir, valid_rgb_dir, valid_spec_dir, split_dir):
        directory.mkdir(parents=True)

    test_scene = "ARAD_1K_0952"
    valid_scene = "ARAD_1K_0902"
    (split_dir / "test_list.txt").write_text(f"{test_scene}\n", encoding="utf-8")
    (split_dir / "valid_list.txt").write_text(f"{valid_scene}\n", encoding="utf-8")

    rgb = np.random.randint(0, 255, (4, 5, 3), dtype=np.uint8)
    assert cv2.imwrite(str(test_rgb_dir / f"{test_scene}.jpg"), rgb)
    assert cv2.imwrite(str(valid_rgb_dir / f"{valid_scene}.jpg"), rgb)
    hyper = np.random.rand(31, 5, 4).astype(np.float32)
    with h5py.File(valid_spec_dir / f"{valid_scene}.mat", "w") as mat:
        mat.create_dataset("cube", data=hyper)

    dataset = tool.NTIRESplitDataset(data_root=str(case_dir), split="auto")
    sample = dataset[0]

    assert dataset.split_name == "valid"
    assert dataset.stems == [valid_scene]
    assert sample.target is not None
    assert tuple(sample.target.shape) == (31, 4, 5)


def test_ntire_dataset_uses_train_directories_as_compatibility_fallback(tmp_path):
    tool = _load_tool_module()
    case_dir = tmp_path / f"case_{uuid.uuid4().hex}"
    rgb_dir = case_dir / "Train_RGB"
    spec_dir = case_dir / "Train_Spec"
    split_dir = case_dir / "split_txt"
    for directory in (rgb_dir, spec_dir, split_dir):
        directory.mkdir(parents=True)

    scene = "ARAD_1K_0903"
    (split_dir / "valid_list.txt").write_text(f"{scene}\n", encoding="utf-8")
    rgb = np.random.randint(0, 255, (4, 5, 3), dtype=np.uint8)
    assert cv2.imwrite(str(rgb_dir / f"{scene}.jpg"), rgb)
    hyper = np.random.rand(31, 5, 4).astype(np.float32)
    with h5py.File(spec_dir / f"{scene}.mat", "w") as mat:
        mat.create_dataset("cube", data=hyper)

    dataset = tool.NTIRESplitDataset(data_root=str(case_dir), split="valid")
    assert dataset[0].target is not None
