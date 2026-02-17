import numpy as np
import torch
import pytest
import shutil
import uuid
from pathlib import Path

h5py = pytest.importorskip("h5py")
PIL_Image = pytest.importorskip("PIL.Image")
cv2 = pytest.importorskip("cv2")

from hsi_model.utils.data.arad_dataset import ARAD1KDataset
from hsi_model.utils.data.mst_dataset import MST_TrainDataset, MST_ValidDataset
from hsi_model.utils.data.transforms import normalize_batch, denormalize_batch


def test_arad1k_dataset_loads_minimal():
    root = Path(__file__).resolve().parents[1] / ".tmp_manual_dataset"
    case_dir = root / f"case_{uuid.uuid4().hex}"
    rgb_dir = case_dir / "ValidationRGB"
    hsi_dir = case_dir / "ValidationHSI"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    hsi_dir.mkdir(parents=True, exist_ok=True)

    try:
        rgb = np.zeros((4, 5, 3), dtype=np.uint8)
        PIL_Image.fromarray(rgb).save(rgb_dir / "sample.png")

        hsi = np.random.rand(4, 5, 31).astype(np.float32)
        with h5py.File(hsi_dir / "sample.mat", "w") as mat:
            mat.create_dataset("cube", data=hsi)

        dataset = ARAD1KDataset(data_dir=str(case_dir), validate_data=True)
        rgb_tensor, hsi_tensor = dataset[0]

        assert rgb_tensor.shape[0] == 3
        assert hsi_tensor.shape[0] == 31
        assert rgb_tensor.shape[1:] == hsi_tensor.shape[1:]
        assert torch.isfinite(rgb_tensor).all()
        assert torch.isfinite(hsi_tensor).all()
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_normalize_denormalize_handles_zero_std():
    batch = torch.ones(2, 3, 4, 4)
    mean = torch.zeros(3)
    std = torch.zeros(3)

    normalized = normalize_batch(batch, mean, std)
    restored = denormalize_batch(normalized, mean, std)

    assert torch.isfinite(normalized).all()
    assert torch.isfinite(restored).all()


def test_mst_dataset_constant_rgb_no_division_by_zero():
    root = Path(__file__).resolve().parents[1] / ".tmp_manual_dataset"
    case_dir = root / f"mst_case_{uuid.uuid4().hex}"
    spec_dir = case_dir / "Train_Spec"
    rgb_dir = case_dir / "Train_RGB"
    split_dir = case_dir / "split_txt"
    spec_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    scene = "scene001"
    (split_dir / "train_list.txt").write_text(f"{scene}\n", encoding="utf-8")
    (split_dir / "valid_list.txt").write_text(f"{scene}\n", encoding="utf-8")

    try:
        hyper = np.random.rand(31, 4, 4).astype(np.float32)
        with h5py.File(spec_dir / f"{scene}.mat", "w") as mat:
            mat.create_dataset("cube", data=hyper)

        rgb_const = np.full((4, 4, 3), 127, dtype=np.uint8)
        assert cv2.imwrite(str(rgb_dir / f"{scene}.jpg"), rgb_const)

        train_dataset = MST_TrainDataset(
            data_root=str(case_dir),
            crop_size=2,
            stride=1,
            arg=False,
            bgr2rgb=True,
        )
        rgb_patch, hsi_patch = train_dataset[0]
        assert np.isfinite(rgb_patch).all()
        assert np.isfinite(hsi_patch).all()
        assert float(np.max(rgb_patch)) == 0.0

        valid_dataset = MST_ValidDataset(data_root=str(case_dir), bgr2rgb=True)
        rgb_full, hsi_full = valid_dataset[0]
        assert np.isfinite(rgb_full).all()
        assert np.isfinite(hsi_full).all()
        assert float(np.max(rgb_full)) == 0.0
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)
