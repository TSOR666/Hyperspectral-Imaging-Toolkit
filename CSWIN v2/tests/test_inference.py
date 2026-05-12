import torch

from hsi_model.utils.patch_inference import PatchInference


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
