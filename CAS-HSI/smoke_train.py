#!/usr/bin/env python3
"""End-to-end CPU smoke test: synthetic ARAD-1K -> train -> validate -> test.

Fabricates a miniature ARAD-1K tree (900/50/50-shaped, but tiny scenes and only a
handful of them), runs the real trainer and the real NTIRE test tool against it, and
asserts the pipeline produces finite, improving metrics.

This exercises the actual code paths -- dataloader, MRAE loss, AMP wiring, cosine
schedule, checkpointing, full-scene validation, both metric protocols -- rather than
mocking them. It is the fastest way to know the training stack is wired correctly
before committing a GPU to 300 epochs.

    python smoke_train.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import h5py
import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

BANDS = 31


def make_scene(root: Path, stem: str, height: int, width: int, rng: np.random.Generator) -> None:
    """A scene whose HSI cube is a smooth, RGB-correlated function -- learnable, not noise."""
    rgb = rng.random((height, width, 3), dtype=np.float32)
    rgb = cv2.GaussianBlur(rgb, (5, 5), 0)

    # Give the cube real structure to learn: each band is a smooth mix of the RGB
    # channels. A cube of pure noise would make a falling loss impossible and the
    # smoke test vacuous.
    wavelengths = np.linspace(0.0, 1.0, BANDS, dtype=np.float32)
    basis = np.stack(
        [
            np.exp(-((wavelengths - 0.15) ** 2) / 0.05),
            np.exp(-((wavelengths - 0.50) ** 2) / 0.05),
            np.exp(-((wavelengths - 0.85) ** 2) / 0.05),
        ],
        axis=0,
    )  # (3, BANDS)
    cube = np.tensordot(rgb, basis, axes=([2], [0]))          # (H, W, BANDS)
    cube = np.clip(cube, 0.0, 1.0).astype(np.float32)
    cube = np.transpose(cube, (2, 0, 1))                      # (BANDS, H, W)

    (root / "Train_RGB").mkdir(parents=True, exist_ok=True)
    (root / "Train_Spec").mkdir(parents=True, exist_ok=True)

    bgr = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(root / "Train_RGB" / f"{stem}.jpg"), bgr)

    # ARAD stores (bands, width, height); the loader transposes it back.
    with h5py.File(root / "Train_Spec" / f"{stem}.mat", "w") as handle:
        handle.create_dataset("cube", data=np.transpose(cube, (0, 2, 1)))


def build_dataset(root: Path, n_train: int = 6, n_valid: int = 2, n_test: int = 2) -> None:
    rng = np.random.default_rng(0)
    stems = [f"ARAD_1K_{i:04d}" for i in range(1, n_train + n_valid + n_test + 1)]
    for stem in stems:
        make_scene(root, stem, height=96, width=112, rng=rng)

    split_dir = root / "split_txt"
    split_dir.mkdir(parents=True, exist_ok=True)
    train = stems[:n_train]
    valid = stems[n_train : n_train + n_valid]
    test = stems[n_train + n_valid :]
    (split_dir / "train_list.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (split_dir / "valid_list.txt").write_text("\n".join(valid) + "\n", encoding="utf-8")
    (split_dir / "test_list.txt").write_text("\n".join(test) + "\n", encoding="utf-8")
    print(f"synthetic ARAD-1K: {len(train)} train / {len(valid)} valid / {len(test)} test scenes")


def main() -> int:
    workspace = Path(tempfile.mkdtemp(prefix="cas_hsi_smoke_"))
    data_root = workspace / "ARAD_1K"
    output_root = workspace / "experiments"

    try:
        build_dataset(data_root)

        # --- splits verify cleanly (and are disjoint) --------------------------
        import prepare_splits

        assert prepare_splits.main(["--data_root", str(data_root)]) == 0, "split verification failed"

        # --- train -------------------------------------------------------------
        import train_cas_hsi

        config = train_cas_hsi.TrainConfig(
            variant="tiny",
            model_overrides={"base_width": 32, "depths": {
                "encoder_full": 1, "encoder_half": 1, "bottleneck": 3,
                "decoder_half": 1, "decoder_full": 1, "refinement": 1,
            }},
            data_root=str(data_root),
            output_root=str(output_root),
            patch_size=64,
            stride=32,
            batch_size=2,
            num_workers=0,
            epochs=3,
            steps_per_epoch=8,
            lr=1e-3,
            amp=False,
            channels_last=False,
            val_crop_border=16,          # the synthetic scenes are 96x112
            log_interval=4,
            experiment_name="smoke",
        )
        trainer = train_cas_hsi.Trainer(config)
        result = trainer.fit()

        assert np.isfinite(result["best_selection"]), "selection metric is not finite"
        last = result["last_val"]
        assert last is not None, "no validation ran"
        for protocol, values in last["protocols"].items():
            for key, value in values.items():
                assert np.isfinite(value), f"val {protocol}/{key} is not finite"
        print(f"\ntrain OK: best selection MRAE = {result['best_selection']:.6f}")

        history = trainer.exp_dir / "history.csv"
        metrics = trainer.exp_dir / "metrics.jsonl"
        assert history.exists() and metrics.exists(), "history/metrics not written"
        header = history.read_text(encoding="utf-8").splitlines()[0]
        for required in ("train/loss", "train/mrae", "train/psnr", "train/rmse", "train/ssim",
                         "val/loss", "val/full/mrae", "val/full/psnr", "val/full/rmse", "val/full/ssim"):
            assert required in header, f"history.csv is missing the {required!r} column"
        print(f"logged columns OK: {header}")

        best = trainer.ckpt_dir / "best.pth"
        assert best.exists(), "no best checkpoint written"

        # --- test on the held-out split ---------------------------------------
        import test_cas_ntire

        code = test_cas_ntire.main([
            "--checkpoint", str(best),
            "--data_root", str(data_root),
            "--split", "test",
            "--device", "cpu",
            "--crop_border", "16",
            "--json", str(workspace / "report.json"),
        ])
        assert code == 0, "test_cas_ntire failed"
        assert (workspace / "report.json").exists()

        print("\nSMOKE TEST PASSED: train -> validate -> checkpoint -> NTIRE test all work.")
        return 0

    finally:
        shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
