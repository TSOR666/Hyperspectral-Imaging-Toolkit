from __future__ import annotations

import math
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset

from optimized_dataloader import DistributedEvalSampler
from sharp_v322_hardened import (
    SHARPv32,
    SHARPv32Config,
    SHARPv32Trainer,
    sparse_attention_local_landmark,
    sparse_attention_topk_streaming,
)


def _ddp_worker(
    rank: int,
    world_size: int,
    init_method: str,
    result_dir: str,
) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        config = SHARPv32Config(
            out_channels=3,
            base_dim=8,
            depths=(1, 1, 1, 1),
            heads=(1, 2, 4, 8),
            mlp_ratios=(1.0, 1.0, 1.0, 1.0),
            sparse_exact_topk_max_tokens=16,
            sparse_landmark_tokens=4,
            sparse_q_block_size=16,
            sparse_window_size=9,
        )
        model = DDP(SHARPv32(config), find_unused_parameters=False)
        trainer = SHARPv32Trainer(
            model,
            total_steps=2,
            use_amp=False,
            ema_decay=0.0,
            criterion=nn.MSELoss(),
        )
        inputs = torch.rand(1, 3, 8, 8)
        targets = torch.rand(1, 3, 8, 8)
        train_metrics = trainer.train_step(inputs, targets)
        train_metrics = trainer.train_step(inputs, targets)
        eval_metrics = trainer.evaluate([(inputs, targets)])
        checksum = sum(
            parameter.detach().double().sum().item()
            for parameter in model.module.parameters()
        )
        torch.save(
            {
                "loss": train_metrics["loss"],
                "mrae": eval_metrics["mrae"],
                "checksum": checksum,
            },
            Path(result_dir) / f"rank_{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


def test_large_sequence_uses_local_landmark_candidates(monkeypatch) -> None:
    called = {}

    def bounded_attention(q, k, v, **kwargs):
        called.update(kwargs)
        return torch.zeros(q.shape[0], q.shape[1], v.shape[-1])

    monkeypatch.setattr(
        "sharp_v322_hardened.sparse_attention_local_landmark",
        bounded_attention,
    )
    q = torch.randn(1, 65, 4)
    output = sparse_attention_topk_streaming(
        q,
        q,
        q,
        max_tokens=8192,
        exact_topk_max_tokens=64,
        landmark_tokens=8,
        window_size=9,
        spatial_shape=(5, 13),
    )

    assert output.shape == q.shape
    assert called["landmark_tokens"] == 8


def test_local_landmark_attention_backward_is_finite() -> None:
    q = torch.randn(2, 64, 8, requires_grad=True)
    k = torch.randn(2, 64, 8, requires_grad=True)
    v = torch.randn(2, 64, 6, requires_grad=True)
    output = sparse_attention_local_landmark(
        q,
        k,
        v,
        sparsity_ratio=0.75,
        scale=8 ** -0.5,
        window_size=9,
        landmark_tokens=8,
        k_cap=8,
        spatial_shape=(8, 8),
        q_block_size=16,
    )
    output.square().mean().backward()

    assert output.shape == (2, 64, 6)
    assert all(
        tensor.grad is not None and torch.isfinite(tensor.grad).all()
        for tensor in (q, k, v)
    )


def test_distributed_eval_sampler_does_not_pad() -> None:
    dataset = TensorDataset(torch.arange(5))
    shards = [
        list(DistributedEvalSampler(dataset, num_replicas=2, rank=rank))
        for rank in range(2)
    ]
    assert set(shards[0]).isdisjoint(shards[1])
    assert sorted(shards[0] + shards[1]) == list(range(5))


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_two_process_sharp_ddp_cpu_smoke() -> None:
    tmp_path = Path(".pytest_cache") / f"sharp_ddp_scaling_{os.getpid()}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    for name in ("init", "rank_0.pt", "rank_1.pt"):
        path = tmp_path / name
        if path.exists():
            path.unlink()
    init_file = tmp_path / "init"
    mp.spawn(
        _ddp_worker,
        args=(2, init_file.resolve().as_uri(), str(tmp_path.resolve())),
        nprocs=2,
        join=True,
    )
    results = [
        torch.load(tmp_path / f"rank_{rank}.pt", weights_only=True)
        for rank in range(2)
    ]
    assert all(math.isfinite(result["loss"]) for result in results)
    assert results[0]["mrae"] == pytest.approx(results[1]["mrae"])
    assert results[0]["checksum"] == pytest.approx(
        results[1]["checksum"], abs=1e-8
    )
