import pickle
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import Dataset, TensorDataset

from hsi_model.models.generator_v3 import NoiseRobustCSWinGenerator
from hsi_model.train_generator import _build_train_loader, validate_generator
from hsi_model.utils.inference import load_generator
from hsi_model.utils.patch_inference import PatchInference


def _small_generator_config():
    return {
        "base_channels": 8,
        "num_heads": 2,
        "split_sizes": [2, 2, 2],
        "stage_depths": [1, 1, 1, 1, 1],
        "sampling": "bilinear",
        "spectral_attention_type": "s_msa",
        "cswin_max_long_axis": 64,
    }


def test_train_worker_initializer_is_spawn_picklable():
    dataset = TensorDataset(torch.zeros(2, 3), torch.zeros(2, 31))
    loader = _build_train_loader(
        dataset,
        batch_size=1,
        config={"num_workers": 1},
        distributed=False,
        seed=42,
        rank=0,
    )
    pickle.dumps(loader.worker_init_fn)


class _ValidationDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return torch.zeros(3, 8, 8), torch.zeros(31, 8, 8)


class _FailingModel(torch.nn.Module):
    def forward(self, x):
        raise RuntimeError("systematic validation failure")


class _UnevenValidationDataset(Dataset):
    def __len__(self):
        return 3

    def __getitem__(self, index):
        target_value = (1.0, 1.0, 3.0)[index]
        return (
            torch.zeros(3, 482, 512),
            torch.full((1, 482, 512), target_value),
        )


class _ZeroModel(torch.nn.Module):
    def forward(self, x):
        return torch.zeros(
            x.shape[0],
            1,
            x.shape[2],
            x.shape[3],
            device=x.device,
        )


def test_validation_does_not_turn_all_failed_batches_into_zero_loss():
    with pytest.raises(RuntimeError, match="Refusing to select checkpoints"):
        validate_generator(
            _FailingModel(),
            _ValidationDataset(),
            torch.nn.L1Loss(),
            torch.device("cpu"),
            iteration=1,
            config={
                "val_batch_size": 1,
                "num_workers": 0,
                "validation_max_batches": None,
            },
            distributed=False,
            seed=42,
            rank=0,
        )


def test_validation_loss_is_weighted_by_sample_count():
    metrics = validate_generator(
        _ZeroModel(),
        _UnevenValidationDataset(),
        torch.nn.L1Loss(),
        torch.device("cpu"),
        iteration=1,
        config={
            "val_batch_size": 2,
            "num_workers": 0,
            "validation_max_batches": None,
        },
        distributed=False,
        seed=42,
        rank=0,
    )

    assert metrics["gen_loss"] == pytest.approx(5.0 / 3.0)


def test_strict_checkpoint_load_rejects_missing_state(tmp_path):
    generator = NoiseRobustCSWinGenerator(_small_generator_config())
    state = generator.state_dict()
    state.pop(next(iter(state)))
    checkpoint = tmp_path / "incomplete.pth"
    torch.save(
        {"config": _small_generator_config(), "state_dict": state},
        checkpoint,
    )

    with pytest.raises(RuntimeError, match="Missing key"):
        load_generator(
            str(checkpoint),
            device=torch.device("cpu"),
            strict=True,
        )


class _PointwiseModel(torch.nn.Module):
    def forward(self, x):
        return x[:, :1]


def test_patch_inference_does_not_flush_cuda_cache_per_batch():
    inference = PatchInference(
        _PointwiseModel(),
        patch_size=16,
        overlap=4,
        batch_size=2,
        device=torch.device("cpu"),
    )
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.empty_cache") as empty_cache,
    ):
        output = inference.predict(torch.randn(1, 3, 40, 40), show_progress=False)

    assert output.shape == (1, 1, 40, 40)
    empty_cache.assert_not_called()
