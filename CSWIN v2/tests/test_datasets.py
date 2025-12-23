import numpy as np
import torch
import pytest

h5py = pytest.importorskip("h5py")
Image = pytest.importorskip("PIL.Image").Image

from hsi_model.utils.data.arad_dataset import ARAD1KDataset


def test_arad1k_dataset_loads_minimal(tmp_path):
    rgb_dir = tmp_path / "ValidationRGB"
    hsi_dir = tmp_path / "ValidationHSI"
    rgb_dir.mkdir()
    hsi_dir.mkdir()

    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    Image.fromarray(rgb).save(rgb_dir / "sample.png")

    hsi = np.random.rand(4, 5, 31).astype(np.float32)
    with h5py.File(hsi_dir / "sample.mat", "w") as mat:
        mat.create_dataset("cube", data=hsi)

    dataset = ARAD1KDataset(data_dir=str(tmp_path), validate_data=True)
    rgb_tensor, hsi_tensor = dataset[0]

    assert rgb_tensor.shape[0] == 3
    assert hsi_tensor.shape[0] == 31
    assert rgb_tensor.shape[1:] == hsi_tensor.shape[1:]
    assert torch.isfinite(rgb_tensor).all()
    assert torch.isfinite(hsi_tensor).all()
