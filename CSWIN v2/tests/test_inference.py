import torch

from hsi_model.utils.patch_inference import PatchInference
from hsi_model.utils.inference import (
    load_generator,
    build_patch_inference,
    geometric_self_ensemble,
)
from hsi_model.models.generator_v3 import NoiseRobustCSWinGenerator


_GEN_CFG = {
    "in_channels": 3, "out_channels": 31, "base_channels": 16,
    "num_heads": 2, "split_sizes": [2, 2, 2], "norm_groups": 4,
    "spectral_attention_type": "s_msa", "sampling": "pixelshuffle",
    "use_noise_block": False, "blocks_per_stage": 1, "thick_output_head": True,
    "output_activation": "sigmoid",
}


def _build_gen(seed=0):
    torch.manual_seed(seed)
    return NoiseRobustCSWinGenerator(_GEN_CFG).eval()


def test_load_generator_roundtrip_bare_checkpoint(tmp_path):
    gen = _build_gen()
    ckpt = tmp_path / "best_model.pth"
    torch.save({"state_dict": gen.state_dict(), "config": _GEN_CFG, "ema_applied": True}, ckpt)

    loaded, info = load_generator(str(ckpt), device=torch.device("cpu"))
    x = torch.rand(1, 3, 16, 16)
    with torch.no_grad():
        assert torch.allclose(gen(x), loaded(x), atol=1e-6)
    assert info["applies_own_activation"] is True


def test_load_generator_legacy_full_model_checkpoint(tmp_path):
    gen = _build_gen(seed=1)
    full_state = {f"generator.{k}": v for k, v in gen.state_dict().items()}
    full_state["discriminator.input_proj.weight"] = torch.randn(4, 4)  # junk, must be ignored
    ckpt = tmp_path / "latest_checkpoint.pth"
    torch.save({"state_dict": full_state, "config": _GEN_CFG}, ckpt)

    loaded, _ = load_generator(str(ckpt), device=torch.device("cpu"))
    x = torch.rand(1, 3, 16, 16)
    with torch.no_grad():
        assert torch.allclose(gen(x), loaded(x), atol=1e-6)


def test_load_generator_applies_ema_shadow(tmp_path):
    gen = _build_gen(seed=2)
    # EMA shadow = raw + 0.05 on every tracked param.
    shadow = {n: (p.detach() + 0.05).clone() for n, p in gen.named_parameters()}
    ckpt = tmp_path / "latest_checkpoint.pth"
    torch.save(
        {"state_dict": gen.state_dict(), "config": _GEN_CFG, "ema": {"decay": 0.999, "shadow": shadow}},
        ckpt,
    )
    loaded, info = load_generator(str(ckpt), device=torch.device("cpu"), prefer_ema=True)
    assert info["ema_applied"] is True
    ref = next(iter(shadow))
    got = dict(loaded.named_parameters())[ref]
    assert torch.allclose(got, shadow[ref], atol=1e-6)
    # And NOT the raw weights.
    assert not torch.allclose(got, dict(gen.named_parameters())[ref], atol=1e-3)


def test_load_generator_rejects_partial_ema_shadow(tmp_path):
    gen = _build_gen(seed=4)
    first_name, first_param = next(iter(gen.named_parameters()))
    ckpt = tmp_path / "partial_ema.pth"
    torch.save(
        {
            "state_dict": gen.state_dict(),
            "config": _GEN_CFG,
            "ema": {
                "decay": 0.999,
                "shadow": {first_name: first_param.detach().clone() + 1.0},
            },
        },
        ckpt,
    )

    loaded, info = load_generator(
        str(ckpt),
        device=torch.device("cpu"),
        prefer_ema=True,
    )

    assert info["ema_applied"] is False
    assert torch.allclose(
        dict(loaded.named_parameters())[first_name],
        dict(gen.named_parameters())[first_name],
    )


def test_geometric_self_ensemble_is_exact_for_pointwise_op():
    # A 1x1 conv is equivariant to flips/rotations, so the x8 ensemble must
    # equal a single forward pass (up to float error), and preserve shape on a
    # non-square input.
    conv = torch.nn.Conv2d(3, 31, kernel_size=1)
    conv.eval()
    x = torch.rand(1, 3, 6, 8)
    with torch.no_grad():
        single = conv(x)
        ens = geometric_self_ensemble(lambda t: conv(t), x)
    assert ens.shape == single.shape == (1, 31, 6, 8)
    assert torch.allclose(ens, single, atol=1e-6)


def test_build_patch_inference_runs(tmp_path):
    gen = _build_gen(seed=3)
    ckpt = tmp_path / "best_model.pth"
    torch.save({"state_dict": gen.state_dict(), "config": _GEN_CFG, "ema_applied": True}, ckpt)
    infer = build_patch_inference(
        str(ckpt), device=torch.device("cpu"), patch_size=16, overlap=4, batch_size=2
    )
    out = infer.predict(torch.rand(1, 3, 24, 28), show_progress=False)
    assert out.shape == (1, 31, 24, 28)
    assert torch.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0  # sigmoid head


class _TinyGenerator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.generator = torch.nn.Conv2d(3, 31, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator(x)


def test_patch_inference_pads_tiny_edge_with_replicate_fallback():
    infer = PatchInference(
        _TinyGenerator(),
        patch_size=4,
        overlap=1,
        batch_size=2,
        device=torch.device("cpu"),
    )
    img = torch.rand(1, 3, 1, 5)
    out = infer.predict(img, show_progress=False)
    assert out.shape == (1, 31, 1, 5)
    assert torch.isfinite(out).all()


def test_patch_stitching_preserves_fp32_overlap_average():
    infer = PatchInference(
        torch.nn.Identity(),
        patch_size=4,
        overlap=2,
        batch_size=1,
        device=torch.device("cpu"),
    )
    patches = torch.stack(
        [
            torch.full((1, 4, 4), 0.1, dtype=torch.float16),
            torch.full((1, 4, 4), 0.2, dtype=torch.float16),
        ]
    )
    info = {
        "original_shape": (4, 6),
        "positions": [(0, 0, 4, 4), (0, 2, 4, 6)],
    }

    output = infer._stitch_patches(patches, info, out_channels=1)

    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()
