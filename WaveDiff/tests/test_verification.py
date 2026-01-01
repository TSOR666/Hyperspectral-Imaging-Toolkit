"""
WaveDiff Verification Test Script

This script verifies the correctness of the WaveDiff codebase after the 5-step audit.
It tests:
1. Shape correctness through forward passes
2. Numerical stability (no NaN/Inf)
3. Gradient flow (backward pass)
4. Component-level validation

Run with: python -m pytest WaveDiff/tests/test_verification.py -v
Or directly: python WaveDiff/tests/test_verification.py
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock matplotlib before any imports that might use it
class MockMatplotlib:
    def __getattr__(self, name):
        return MockMatplotlib()
    def __call__(self, *args, **kwargs):
        return MockMatplotlib()
    def __iter__(self):
        return iter([])
    def __len__(self):
        return 0

sys.modules['matplotlib'] = MockMatplotlib()
sys.modules['matplotlib.pyplot'] = MockMatplotlib()
sys.modules['matplotlib.colors'] = MockMatplotlib()
sys.modules['matplotlib.cm'] = MockMatplotlib()

import torch


def test_haar_wavelet_transform():
    """Test Haar wavelet transform forward and inverse."""
    from transforms.haar_wavelet import HaarWaveletTransform, InverseHaarWaveletTransform

    B, C, H, W = 2, 3, 64, 64
    x = torch.randn(B, C, H, W)

    # Forward transform: [B, C, H, W] -> [B, C, 4, H//2, W//2]
    wavelet = HaarWaveletTransform(C)
    coeffs = wavelet(x)

    assert coeffs.shape == (B, C, 4, H // 2, W // 2), f"Expected {(B, C, 4, H // 2, W // 2)}, got {coeffs.shape}"
    assert torch.isfinite(coeffs).all(), "Wavelet coefficients contain NaN/Inf"

    # Inverse transform: [B, C, 4, H//2, W//2] -> [B, C, H, W]
    inverse_wavelet = InverseHaarWaveletTransform()
    recon = inverse_wavelet(coeffs)

    assert recon.shape == x.shape, f"Expected {x.shape}, got {recon.shape}"
    assert torch.isfinite(recon).all(), "Reconstructed tensor contains NaN/Inf"

    # Check reconstruction accuracy (should be near-perfect for Haar)
    error = (x - recon).abs().max().item()
    assert error < 1e-5, f"Reconstruction error too large: {error}"

    print("  [PASS] Haar wavelet transform")


def test_adaptive_wavelet_thresholding():
    """Test adaptive wavelet thresholding methods."""
    from transforms.adaptive_wavelet import AdaptiveWaveletThresholding, WaveletNoiseEstimator

    B, C, H, W = 2, 64, 32, 32
    coeffs = torch.randn(B, C, 4, H, W)

    # Test soft thresholding
    soft_thresh = AdaptiveWaveletThresholding(C, method='soft')
    result_soft = soft_thresh(coeffs)
    assert result_soft.shape == coeffs.shape, f"Shape mismatch: {result_soft.shape}"
    assert torch.isfinite(result_soft).all(), "Soft threshold contains NaN/Inf"

    # Test hard thresholding
    hard_thresh = AdaptiveWaveletThresholding(C, method='hard')
    result_hard = hard_thresh(coeffs)
    assert torch.isfinite(result_hard).all(), "Hard threshold contains NaN/Inf"

    # Test garrote thresholding with noise estimation
    garrote_thresh = AdaptiveWaveletThresholding(C, method='garrote')
    noise_estimator = WaveletNoiseEstimator()
    noise_level = noise_estimator(coeffs)
    result_garrote = garrote_thresh(coeffs, noise_level)
    assert torch.isfinite(result_garrote).all(), "Garrote threshold contains NaN/Inf"

    # Test with extreme values
    extreme_coeffs = torch.randn(B, C, 4, H, W) * 100
    result_extreme = garrote_thresh(extreme_coeffs, noise_level)
    assert torch.isfinite(result_extreme).all(), "Garrote fails with extreme values"

    print("  [PASS] Adaptive wavelet thresholding")


def test_attention_modules():
    """Test attention modules for numerical stability."""
    from modules.attention import (
        SpectralAttention,
        CrossSpectralAttention,
        MultiHeadSpectralAttention,
    )

    B, C, H, W = 2, 64, 32, 32
    x = torch.randn(B, C, H, W)

    # Test SpectralAttention
    spec_attn = SpectralAttention(C)
    out = spec_attn(x)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape}"
    assert torch.isfinite(out).all(), "SpectralAttention contains NaN/Inf"

    # Test CrossSpectralAttention (with stabilized softmax)
    cross_attn = CrossSpectralAttention(C, num_heads=4)
    out = cross_attn(x)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape}"
    assert torch.isfinite(out).all(), "CrossSpectralAttention contains NaN/Inf"

    # Test with float16 (stress test for numerical stability)
    if torch.cuda.is_available():
        x_fp16 = x.cuda().half()
        cross_attn_cuda = cross_attn.cuda().half()
        out_fp16 = cross_attn_cuda(x_fp16)
        assert torch.isfinite(out_fp16).all(), "CrossSpectralAttention fails in float16"

    # Test MultiHeadSpectralAttention (with temperature clamping)
    mh_attn = MultiHeadSpectralAttention(C, num_heads=8)
    out = mh_attn(x)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape}"
    assert torch.isfinite(out).all(), "MultiHeadSpectralAttention contains NaN/Inf"

    print("  [PASS] Attention modules")


def test_encoders():
    """Test encoder modules with shape tracing."""
    from modules.encoders import RGBEncoder, WaveletRGBEncoder

    B, H, W = 2, 64, 64
    latent_dim = 64
    x = torch.randn(B, 3, H, W)

    # Test RGBEncoder: [B, 3, H, W] -> [B, latent_dim, H//4, W//4]
    encoder = RGBEncoder(in_channels=3, latent_dim=latent_dim)
    latent = encoder(x)

    expected_shape = (B, latent_dim, H // 4, W // 4)
    assert latent.shape == expected_shape, f"Expected {expected_shape}, got {latent.shape}"
    assert torch.isfinite(latent).all(), "RGBEncoder output contains NaN/Inf"

    # Test WaveletRGBEncoder (with fixed shape bug)
    wavelet_encoder = WaveletRGBEncoder(in_channels=3, latent_dim=latent_dim)
    latent_wav = wavelet_encoder(x)

    assert latent_wav.shape == expected_shape, f"Expected {expected_shape}, got {latent_wav.shape}"
    assert torch.isfinite(latent_wav).all(), "WaveletRGBEncoder output contains NaN/Inf"

    print("  [PASS] Encoders")


def test_decoders():
    """Test decoder modules."""
    from modules.decoders import HSIDecoder, WaveletHSIDecoder

    B, latent_dim, H, W = 2, 64, 16, 16
    out_channels = 31
    x = torch.randn(B, latent_dim, H, W)

    # Test HSIDecoder
    decoder = HSIDecoder(out_channels=out_channels, latent_dim=latent_dim)
    hsi = decoder(x)

    expected_shape = (B, out_channels, H * 4, W * 4)
    assert hsi.shape == expected_shape, f"Expected {expected_shape}, got {hsi.shape}"
    assert torch.isfinite(hsi).all(), "HSIDecoder output contains NaN/Inf"

    # Test WaveletHSIDecoder
    wavelet_decoder = WaveletHSIDecoder(out_channels=out_channels, latent_dim=latent_dim)
    hsi_wav = wavelet_decoder(x)

    assert hsi_wav.shape == expected_shape, f"Expected {expected_shape}, got {hsi_wav.shape}"
    assert torch.isfinite(hsi_wav).all(), "WaveletHSIDecoder output contains NaN/Inf"

    print("  [PASS] Decoders")


def test_denoisers():
    """Test denoiser modules with timestep embedding."""
    from modules.denoisers import UNetDenoiser, WaveletUNetDenoiser

    B, C, H, W = 2, 64, 32, 32
    x = torch.randn(B, C, H, W)
    t = torch.randint(0, 1000, (B,))

    # Test UNetDenoiser
    denoiser = UNetDenoiser(channels=C)
    out = denoiser(x, t)

    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    assert torch.isfinite(out).all(), "UNetDenoiser output contains NaN/Inf"

    # Test WaveletUNetDenoiser
    wavelet_denoiser = WaveletUNetDenoiser(channels=C)
    out_wav = wavelet_denoiser(x, t)

    assert out_wav.shape == x.shape, f"Expected {x.shape}, got {out_wav.shape}"
    assert torch.isfinite(out_wav).all(), "WaveletUNetDenoiser output contains NaN/Inf"

    print("  [PASS] Denoisers")


def test_dpm_ot():
    """Test DPM-OT diffusion process."""
    from diffusion.dpm_ot import DPMOT
    from diffusion.noise_schedule import BaseNoiseSchedule
    from modules.denoisers import UNetDenoiser

    B, C, H, W = 2, 64, 16, 16
    timesteps = 100  # Use fewer steps for faster testing

    denoiser = UNetDenoiser(channels=C)
    schedule = BaseNoiseSchedule(timesteps=timesteps)
    dpm_ot = DPMOT(denoiser=denoiser, spectral_schedule=schedule, timesteps=timesteps)

    # Test q_sample (forward diffusion)
    x_0 = torch.randn(B, C, H, W)
    t = torch.randint(0, timesteps, (B,))
    x_t, noise = dpm_ot.q_sample(x_0, t)

    assert x_t.shape == x_0.shape, f"Shape mismatch: {x_t.shape}"
    assert torch.isfinite(x_t).all(), "q_sample output contains NaN/Inf"

    # Test p_losses (training loss)
    loss, pred_noise, true_noise = dpm_ot.p_losses(x_0, t)

    assert torch.isfinite(loss), f"Loss is not finite: {loss}"
    assert torch.isfinite(pred_noise).all(), "Predicted noise contains NaN/Inf"

    # Test sampling with DPM-Solver (use few steps for speed)
    with torch.no_grad():
        sample = dpm_ot.sample(
            shape=(1, C, H, W),
            device=x_0.device,
            use_dpm_solver=True,
            steps=5,
        )

    assert sample.shape == (1, C, H, W), f"Sample shape mismatch: {sample.shape}"
    assert torch.isfinite(sample).all(), "DPM-Solver sample contains NaN/Inf"

    print("  [PASS] DPM-OT")


def test_noise_schedules():
    """Test noise schedule computations."""
    from diffusion.noise_schedule import BaseNoiseSchedule, SpectralNoiseSchedule

    timesteps = 1000

    # Test BaseNoiseSchedule
    base_schedule = BaseNoiseSchedule(timesteps=timesteps)

    # Verify buffer shapes
    assert base_schedule.betas.shape == (timesteps,)
    assert base_schedule.alphas_cumprod.shape == (timesteps,)
    assert base_schedule.posterior_variance.shape == (timesteps,)

    # Verify numerical properties
    assert (base_schedule.alphas_cumprod >= 0).all(), "Negative alphas_cumprod"
    assert (base_schedule.alphas_cumprod <= 1).all(), "alphas_cumprod > 1"
    assert torch.isfinite(base_schedule.posterior_variance).all(), "posterior_variance contains NaN/Inf"
    assert torch.isfinite(base_schedule.posterior_log_variance).all(), "posterior_log_variance contains NaN/Inf"

    # Test SpectralNoiseSchedule
    spectral_schedule = SpectralNoiseSchedule(timesteps=timesteps, num_freq_bands=8)
    x = torch.randn(2, 64, 32, 32)
    t = torch.randint(0, timesteps, (2,))
    beta_adapted = spectral_schedule(x, t)

    assert beta_adapted.shape == (2, 1, 1, 1), f"Shape mismatch: {beta_adapted.shape}"
    assert torch.isfinite(beta_adapted).all(), "Spectral schedule output contains NaN/Inf"

    print("  [PASS] Noise schedules")


def test_base_model_forward():
    """Test full model forward pass."""
    from models.base_model import HSILatentDiffusionModel

    B, H, W = 2, 64, 64
    out_channels = 31

    model = HSILatentDiffusionModel(
        latent_dim=64,
        out_channels=out_channels,
        timesteps=100,  # Fewer for faster testing
    )

    rgb = torch.randn(B, 3, H, W)

    # Forward pass
    outputs = model(rgb, use_masking=False)

    # Check output keys
    expected_keys = {'latent', 'diffusion_loss', 'pred_noise', 'noise', 'hsi_initial', 'hsi_output', 'rgb_from_hsi'}
    assert expected_keys.issubset(outputs.keys()), f"Missing keys: {expected_keys - outputs.keys()}"

    # Check shapes
    assert outputs['hsi_output'].shape == (B, out_channels, H, W), f"HSI shape: {outputs['hsi_output'].shape}"
    assert outputs['rgb_from_hsi'].shape == (B, 3, H, W), f"RGB shape: {outputs['rgb_from_hsi'].shape}"

    # Check numerical stability
    assert torch.isfinite(outputs['hsi_output']).all(), "HSI output contains NaN/Inf"
    assert torch.isfinite(outputs['rgb_from_hsi']).all(), "RGB output contains NaN/Inf"
    assert torch.isfinite(outputs['diffusion_loss']), "Diffusion loss is not finite"

    print("  [PASS] Base model forward")


def test_gradient_flow():
    """Test that gradients flow correctly through the model."""
    from models.base_model import HSILatentDiffusionModel

    model = HSILatentDiffusionModel(
        latent_dim=32,
        out_channels=31,
        timesteps=100,
    )

    rgb = torch.randn(1, 3, 32, 32, requires_grad=True)
    hsi_target = torch.randn(1, 31, 32, 32)

    outputs = model(rgb, use_masking=False)
    losses = model.calculate_losses(outputs, rgb, hsi_target)

    total_loss = losses['diffusion_loss'] + losses['cycle_loss'] + losses['l1_loss']
    total_loss.backward()

    # Check gradients
    assert rgb.grad is not None, "Input gradient is None"
    assert torch.isfinite(rgb.grad).all(), "Input gradient contains NaN/Inf"

    # Check model parameter gradients
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"Gradient for {name} contains NaN/Inf"

    print("  [PASS] Gradient flow")


def test_wavelet_loss():
    """Test wavelet loss functions."""
    from losses.wavelet_loss import WaveletLoss, MultiscaleWaveletLoss

    B, C, H, W = 2, 31, 64, 64
    pred = torch.randn(B, C, H, W)
    target = torch.randn(B, C, H, W)

    # Test WaveletLoss
    wavelet_loss = WaveletLoss(in_channels=C)
    loss = wavelet_loss(pred, target)
    assert loss.shape == (), f"WaveletLoss output shape mismatch: {loss.shape}"
    assert loss.item() >= 0, "WaveletLoss should be non-negative"
    assert torch.isfinite(loss), "WaveletLoss contains NaN/Inf"

    # Test MultiscaleWaveletLoss
    ms_loss = MultiscaleWaveletLoss(in_channels=C, num_levels=2)
    loss = ms_loss(pred, target)
    assert loss.shape == () or loss.numel() == 1, f"MultiscaleWaveletLoss shape mismatch"
    assert torch.isfinite(loss), "MultiscaleWaveletLoss contains NaN/Inf"

    print("  [PASS] Wavelet loss functions")


def test_masking_manager():
    """Test masking manager strategies."""
    from utils.masking import MaskingManager

    B, C, H, W = 2, 31, 64, 64
    x = torch.randn(B, C, H, W)
    device = x.device

    # Test random masking
    config_random = {'mask_strategy': 'random', 'mask_ratio': 0.5}
    manager = MaskingManager(config_random)
    mask = manager.generate_mask(x, B, C, H, W, device)
    assert mask.shape == (B, 1, H, W), f"Mask shape mismatch: {mask.shape}"
    assert ((mask == 0) | (mask == 1)).all(), "Mask should be binary"

    # Test block masking
    config_block = {'mask_strategy': 'block', 'mask_ratio': 0.5}
    manager_block = MaskingManager(config_block)
    mask_block = manager_block.generate_mask(x, B, C, H, W, device)
    assert mask_block.shape == (B, 1, H, W), f"Block mask shape mismatch"

    # Test spectral masking - note: spectral masks may have different shape depending on implementation
    config_spectral = {'mask_strategy': 'spectral', 'band_mask_ratio': 0.5}
    manager_spectral = MaskingManager(config_spectral)
    mask_spectral = manager_spectral.generate_mask(x, B, C, H, W, device)
    # Spectral masks can be (B, C, 1, 1) or (B, C, H, W) depending on whether they're band-wise or spatial
    assert mask_spectral.shape[0] == B and mask_spectral.shape[1] == C, f"Spectral mask batch/channel mismatch"

    print("  [PASS] Masking manager")


def test_wavelet_model():
    """Test wavelet HSI latent diffusion model."""
    from models.wavelet_model import WaveletHSILatentDiffusionModel

    B, H, W = 2, 64, 64
    rgb = torch.randn(B, 3, H, W)

    model = WaveletHSILatentDiffusionModel(
        latent_dim=32,
        out_channels=31,
        timesteps=100,
        use_batchnorm=True
    )

    # Test forward pass
    outputs = model(rgb)
    assert 'hsi_output' in outputs, "Model should output 'hsi_output'"
    assert outputs['hsi_output'].shape == (B, 31, H, W), f"HSI output shape mismatch: {outputs['hsi_output'].shape}"

    # Test encode/decode
    latent = model.encode(rgb)
    assert latent.shape[0] == B, "Latent batch size mismatch"
    assert latent.shape[1] == 32, "Latent dim mismatch"

    hsi = model.decode(latent)
    assert hsi.shape == (B, 31, H, W), "Decoded HSI shape mismatch"

    print("  [PASS] Wavelet model")


def test_adaptive_model():
    """Test adaptive wavelet HSI latent diffusion model initialization and basic methods."""
    from models.adaptive_model import AdaptiveWaveletHSILatentDiffusionModel

    B, H, W = 2, 64, 64
    rgb = torch.randn(B, 3, H, W)

    # Use latent_dim=64 (divisible by num_heads=8)
    model = AdaptiveWaveletHSILatentDiffusionModel(
        latent_dim=64,
        out_channels=31,
        timesteps=100,
        use_batchnorm=True,
        threshold_method='soft',
        init_threshold=0.1,
        trainable_threshold=True
    )

    # Test encode/decode (simpler path that doesn't involve diffusion)
    latent = model.encode(rgb)
    assert latent.shape[0] == B, "Latent batch size mismatch"
    assert latent.shape[1] == 64, "Latent dim mismatch"

    hsi = model.decode(latent)
    assert hsi.shape == (B, 31, H, W), "Decoded HSI shape mismatch"

    # Test threshold stats
    stats = model.get_adaptive_threshold_stats()
    assert isinstance(stats, dict), "Threshold stats should be a dict"

    # Test adaptive thresholding module directly
    from transforms.haar_wavelet import HaarWaveletTransform, InverseHaarWaveletTransform
    wavelet = HaarWaveletTransform(31)
    inv_wavelet = InverseHaarWaveletTransform()

    test_hsi = torch.randn(B, 31, H, W)
    coeffs = wavelet(test_hsi)
    thresholded = model.adaptive_thresholding(coeffs)
    assert thresholded.shape == coeffs.shape, "Thresholded coeffs shape mismatch"
    reconstructed = inv_wavelet(thresholded)
    assert reconstructed.shape == test_hsi.shape, "Reconstructed shape mismatch"

    print("  [PASS] Adaptive model")


def run_all_tests():
    """Run all verification tests."""
    print("\n" + "=" * 60)
    print("WaveDiff Verification Test Suite")
    print("=" * 60 + "\n")

    tests = [
        ("Haar Wavelet Transform", test_haar_wavelet_transform),
        ("Adaptive Wavelet Thresholding", test_adaptive_wavelet_thresholding),
        ("Attention Modules", test_attention_modules),
        ("Encoders", test_encoders),
        ("Decoders", test_decoders),
        ("Denoisers", test_denoisers),
        ("DPM-OT", test_dpm_ot),
        ("Noise Schedules", test_noise_schedules),
        ("Base Model Forward", test_base_model_forward),
        ("Gradient Flow", test_gradient_flow),
        ("Wavelet Loss", test_wavelet_loss),
        ("Masking Manager", test_masking_manager),
        ("Wavelet Model", test_wavelet_model),
        ("Adaptive Model", test_adaptive_model),
    ]

    passed = 0
    failed = 0
    failures = []

    for name, test_fn in tests:
        try:
            print(f"Testing {name}...")
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            failures.append((name, str(e)))
            print(f"  [FAIL] {name}: {e}")

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failures:
        print("\nFailures:")
        for name, error in failures:
            print(f"  - {name}: {error}")
        raise AssertionError(f"{failed} test(s) failed")

    print("\nVERIFICATION PASSED")
    return True


if __name__ == "__main__":
    run_all_tests()
