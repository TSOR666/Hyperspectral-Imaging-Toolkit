import numpy as np
import torch
import pytest
import pickle
import shutil
import uuid
from pathlib import Path

h5py = pytest.importorskip("h5py")
PIL_Image = pytest.importorskip("PIL.Image")
cv2 = pytest.importorskip("cv2")

from hsi_model.utils.data.arad_dataset import ARAD1KDataset
from hsi_model.utils.data.loaders import DistributedEvalSampler
from hsi_model.utils.data.mst_dataset import MST_TrainDataset, MST_ValidDataset
from hsi_model.utils.data import mst_dataset as mst_dataset_module
from hsi_model.utils.data.transforms import normalize_batch, denormalize_batch


def test_distributed_eval_sampler_partitions_without_duplicates():
    dataset = list(range(11))
    shards = [
        list(DistributedEvalSampler(dataset, num_replicas=4, rank=rank))
        for rank in range(4)
    ]

    flattened = [index for shard in shards for index in shard]
    assert sorted(flattened) == list(range(len(dataset)))
    assert len(flattened) == len(set(flattened))


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
        # Non-square W,H storage exercises the MST C,W,H -> C,H,W transpose.
        hyper = np.random.rand(31, 5, 4).astype(np.float32)
        with h5py.File(spec_dir / f"{scene}.mat", "w") as mat:
            mat.create_dataset("cube", data=hyper)

        rgb_const = np.full((4, 5, 3), 127, dtype=np.uint8)
        assert cv2.imwrite(str(rgb_dir / f"{scene}.jpg"), rgb_const)

        train_dataset = MST_TrainDataset(
            data_root=str(case_dir),
            crop_size=2,
            stride=1,
            arg=False,
            bgr2rgb=True,
        )
        assert len(train_dataset) == 12
        rgb_patch, hsi_patch = train_dataset[0]
        assert rgb_patch.shape == (3, 2, 2)
        assert hsi_patch.shape == (31, 2, 2)
        assert np.isfinite(rgb_patch).all()
        assert np.isfinite(hsi_patch).all()
        assert float(np.max(rgb_patch)) == 0.0

        valid_dataset = MST_ValidDataset(data_root=str(case_dir), bgr2rgb=True)
        rgb_full, hsi_full = valid_dataset[0]
        assert rgb_full.shape == (3, 4, 5)
        assert hsi_full.shape == (31, 4, 5)
        assert np.isfinite(rgb_full).all()
        assert np.isfinite(hsi_full).all()
        assert float(np.max(rgb_full)) == 0.0
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_mst_valid_dataset_prefers_validation_directories():
    root = Path(__file__).resolve().parents[1] / ".tmp_manual_dataset"
    case_dir = root / f"mst_val_case_{uuid.uuid4().hex}"
    spec_dir = case_dir / "Test_Spec"
    rgb_dir = case_dir / "Test_RGB"
    split_dir = case_dir / "split_txt"
    spec_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    scene = "ARAD_1K_0901"
    (split_dir / "valid_list.txt").write_text(f"{scene}\n", encoding="utf-8")

    try:
        # C,H,W storage is also accepted and corrected after the MST transpose.
        hyper = np.random.rand(31, 4, 5).astype(np.float32)
        with h5py.File(spec_dir / f"{scene}.mat", "w") as mat:
            mat.create_dataset("cube", data=hyper)

        rgb = np.random.randint(0, 255, (4, 5, 3), dtype=np.uint8)
        assert cv2.imwrite(str(rgb_dir / f"{scene}.jpg"), rgb)

        valid_dataset = MST_ValidDataset(data_root=str(case_dir), bgr2rgb=True)
        assert len(valid_dataset) == 1
        rgb_full, hsi_full = valid_dataset[0]
        assert rgb_full.shape == (3, 4, 5)
        assert hsi_full.shape == (31, 4, 5)
        assert np.isfinite(rgb_full).all()
        assert np.isfinite(hsi_full).all()
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_mst_lazy_mode_reads_matching_patches_without_full_cube_init(monkeypatch):
    root = Path(__file__).resolve().parents[1] / ".tmp_manual_dataset"
    case_dir = root / f"mst_lazy_case_{uuid.uuid4().hex}"
    spec_dir = case_dir / "Train_Spec"
    rgb_dir = case_dir / "Train_RGB"
    split_dir = case_dir / "split_txt"
    spec_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    scenes = ("scene001", "scene002")
    (split_dir / "train_list.txt").write_text(
        "\n".join(scenes) + "\n", encoding="utf-8"
    )
    (split_dir / "valid_list.txt").write_text("scene001\n", encoding="utf-8")

    try:
        for scene_idx, scene in enumerate(scenes):
            hyper = (
                np.arange(31 * 6 * 5, dtype=np.float32).reshape(31, 6, 5)
                + scene_idx * 10000
            )
            with h5py.File(spec_dir / f"{scene}.mat", "w") as mat:
                mat.create_dataset("cube", data=hyper)
            rgb = (
                np.arange(5 * 6 * 3, dtype=np.uint8).reshape(5, 6, 3)
                + scene_idx
            )
            assert cv2.imwrite(str(rgb_dir / f"{scene}.jpg"), rgb)

        standard = MST_TrainDataset(
            data_root=str(case_dir),
            crop_size=3,
            stride=1,
            arg=False,
            memory_mode="standard",
        )

        original_loader = mst_dataset_module._load_mst_cube

        def reject_full_cube_read(path):
            raise AssertionError(f"lazy initialization read full cube {path}")

        monkeypatch.setattr(
            mst_dataset_module,
            "_load_mst_cube",
            reject_full_cube_read,
        )
        lazy = MST_TrainDataset(
            data_root=str(case_dir),
            crop_size=3,
            stride=1,
            arg=False,
            memory_mode="lazy",
            lazy_cache_size=1,
        )
        monkeypatch.setattr(mst_dataset_module, "_load_mst_cube", original_loader)

        assert lazy.hypers == []
        assert lazy.bgrs == []
        standard_rgb, standard_hsi = standard[0]
        lazy_rgb, lazy_hsi = lazy[0]
        assert np.allclose(lazy_rgb, standard_rgb)
        assert np.array_equal(lazy_hsi, standard_hsi)

        _ = lazy[lazy.patch_per_img]
        assert len(lazy._rgb_cache) == 1
        assert len(lazy._h5_files) == 1
        restored = pickle.loads(pickle.dumps(lazy))
        assert restored._h5_files == {}
        assert restored._rgb_cache == {}

        valid = MST_ValidDataset(
            data_root=str(case_dir),
            memory_mode="lazy",
            lazy_cache_size=1,
        )
        rgb_full, hsi_full = valid[0]
        assert rgb_full.shape == (3, 5, 6)
        assert hsi_full.shape == (31, 5, 6)
        assert valid.hypers == []
        assert valid.bgrs == []
    finally:
        for name in ("lazy", "valid"):
            dataset = locals().get(name)
            if dataset is not None:
                dataset.close()
        shutil.rmtree(case_dir, ignore_errors=True)


def test_mst_float16_mode_reduces_resident_dtype():
    root = Path(__file__).resolve().parents[1] / ".tmp_manual_dataset"
    case_dir = root / f"mst_fp16_case_{uuid.uuid4().hex}"
    spec_dir = case_dir / "Train_Spec"
    rgb_dir = case_dir / "Train_RGB"
    split_dir = case_dir / "split_txt"
    spec_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    scene = "scene001"
    (split_dir / "train_list.txt").write_text(f"{scene}\n", encoding="utf-8")

    try:
        with h5py.File(spec_dir / f"{scene}.mat", "w") as mat:
            mat.create_dataset(
                "cube", data=np.random.rand(31, 6, 5).astype(np.float32)
            )
        assert cv2.imwrite(
            str(rgb_dir / f"{scene}.jpg"),
            np.random.randint(0, 255, (5, 6, 3), dtype=np.uint8),
        )
        dataset = MST_TrainDataset(
            data_root=str(case_dir),
            crop_size=3,
            stride=1,
            arg=False,
            memory_mode="float16",
        )
        assert dataset.bgrs[0].dtype == np.float16
        assert dataset.hypers[0].dtype == np.float16
        rgb_patch, hsi_patch = dataset[0]
        assert rgb_patch.dtype == np.float32
        assert hsi_patch.dtype == np.float32
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_mst_train_strict_files_rejects_missing_split_scene(tmp_path):
    spec_dir = tmp_path / "Train_Spec"
    rgb_dir = tmp_path / "Train_RGB"
    split_dir = tmp_path / "split_txt"
    spec_dir.mkdir()
    rgb_dir.mkdir()
    split_dir.mkdir()
    (split_dir / "train_list.txt").write_text(
        "ARAD_1K_0314\n", encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="ARAD_1K_0314"):
        MST_TrainDataset(
            data_root=str(tmp_path),
            crop_size=2,
            stride=1,
            arg=False,
            strict_files=True,
        )


def test_mst_train_allows_explicit_known_corrupt_scene_exclusion(tmp_path):
    spec_dir = tmp_path / "Train_Spec"
    rgb_dir = tmp_path / "Train_RGB"
    split_dir = tmp_path / "split_txt"
    spec_dir.mkdir()
    rgb_dir.mkdir()
    split_dir.mkdir()
    (split_dir / "train_list.txt").write_text(
        "ARAD_1K_0314\nscene_valid\n", encoding="utf-8"
    )

    with h5py.File(spec_dir / "scene_valid.mat", "w") as mat:
        mat.create_dataset(
            "cube", data=np.random.rand(31, 4, 4).astype(np.float32)
        )
    assert cv2.imwrite(
        str(rgb_dir / "scene_valid.jpg"),
        np.random.randint(0, 255, (4, 4, 3), dtype=np.uint8),
    )

    dataset = MST_TrainDataset(
        data_root=str(tmp_path),
        crop_size=2,
        stride=1,
        arg=False,
        strict_files=True,
        excluded_scene_stems=["ARAD_1K_0314"],
    )

    assert dataset.img_num == 1
    assert len(dataset) == 9
