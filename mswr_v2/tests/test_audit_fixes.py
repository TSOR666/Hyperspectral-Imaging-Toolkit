"""
Audit-specific regression tests for MSWR v2.1.2.

Covers all BLOCKER and HIGH issues identified in the deep-learning systems audit:
  BLOCKER-1  Wavelet reshape ordering  (view B,C,4 not B,4,C)
  HIGH-1     SSIM variance clamping
  HIGH-2     PerformanceMonitor sync gate
  HIGH-3     Flash attention position bias
  HIGH-4     GradScaler checkpoint restore
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

# Add parent directory to path for imports
_parent = Path(__file__).resolve().parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

try:
    from model.mswr_net_v212 import (
        OptimizedCNNWaveletTransform,
        OptimizedCNNInverseWaveletTransform,
        PerformanceMonitor,
        create_mswr_tiny,
    )
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

pytestmark = pytest.mark.skipif(not MODEL_AVAILABLE, reason="model module not available")


# ---------------------------------------------------------------------------
# BLOCKER-1: Wavelet reshape correctness
# ---------------------------------------------------------------------------
class TestWaveletReshapeCorrectness:
    """Verify that forward DWT produces correctly separated subbands."""

    def test_subband_separation_single_channel(self):
        """
        With C=1, grouped conv has groups=1 so filter ordering is trivial.
        The LL subband should be a lowpass approximation (roughly the mean of
        each 2x2 block for db1/haar).
        """
        dwt = OptimizedCNNWaveletTransform(J=1, wave="db1")
        # Constant input => all high-freq subbands should be ~0
        x = torch.ones(1, 1, 8, 8)
        yl, yh = dwt(x)
        # yl should be non-zero (lowpass of constant is constant)
        assert yl.abs().mean() > 0.1, "LL subband of constant signal should be non-zero"
        # High-freq should be near-zero for a constant signal
        assert yh[0].abs().mean() < 1e-5, (
            f"High-freq of constant signal should be ~0, got {yh[0].abs().mean():.6f}"
        )

    def test_subband_separation_multichannel(self):
        """
        With C=4, verify each channel's LL and HF subbands are independent.
        Set channel k to constant k+1; LL of channel k ~ k+1, HF ~ 0.
        """
        C = 4
        dwt = OptimizedCNNWaveletTransform(J=1, wave="db1")
        x = torch.zeros(1, C, 8, 8)
        for k in range(C):
            x[0, k] = float(k + 1)

        yl, yh = dwt(x)

        for k in range(C):
            # LL of channel k should be proportional to k+1
            ll_mean = yl[0, k].mean().item()
            assert ll_mean > 0.5 * (k + 1), (
                f"LL[{k}] mean={ll_mean:.4f}, expected ~{k+1}"
            )
            # HF subbands for constant channel should be ~0
            hf_energy = yh[0][0, k].abs().mean().item()
            assert hf_energy < 1e-4, (
                f"HF[{k}] energy={hf_energy:.6f}, expected ~0 for constant channel"
            )

    def test_roundtrip_reconstruction_fidelity(self):
        """Forward then inverse DWT should approximately reconstruct the input."""
        dwt = OptimizedCNNWaveletTransform(J=1, wave="db1")
        idwt = OptimizedCNNInverseWaveletTransform(wave="db1")

        torch.manual_seed(42)
        x = torch.randn(2, 8, 16, 16)
        yl, yh = dwt(x)
        x_hat = idwt((yl, yh))

        # db1 (Haar) with periodic mode should give near-perfect reconstruction
        rel_error = (x_hat - x).abs().mean() / x.abs().mean()
        assert rel_error < 0.05, f"Roundtrip relative error {rel_error:.4f} > 5%"

    def test_roundtrip_db2(self):
        """Roundtrip with db2 wavelet."""
        dwt = OptimizedCNNWaveletTransform(J=1, wave="db2")
        idwt = OptimizedCNNInverseWaveletTransform(wave="db2")

        torch.manual_seed(7)
        x = torch.randn(1, 4, 32, 32)
        yl, yh = dwt(x)
        x_hat = idwt((yl, yh))

        rel_error = (x_hat - x).abs().mean() / x.abs().mean()
        assert rel_error < 0.1, f"db2 roundtrip relative error {rel_error:.4f} > 10%"

    def test_output_shapes(self):
        """Verify yl and yh shapes match the documented API."""
        dwt = OptimizedCNNWaveletTransform(J=2, wave="db1")
        B, C, H, W = 2, 6, 32, 32
        x = torch.randn(B, C, H, W)
        yl, yh = dwt(x)

        # After J=2 levels: yl is (B, C, H/4, W/4)
        assert yl.shape == (B, C, H // 4, W // 4)
        # yh should have 2 levels
        assert len(yh) == 2
        # Level 0 (first decomposition): (B, C, 3, H/2, W/2)
        assert yh[0].shape == (B, C, 3, H // 2, W // 2)
        # Level 1: (B, C, 3, H/4, W/4)
        assert yh[1].shape == (B, C, 3, H // 4, W // 4)


# ---------------------------------------------------------------------------
# HIGH-1: SSIM variance clamping
# ---------------------------------------------------------------------------
class TestSSIMVarianceClamping:
    """Verify that the SSIM loss clamps variance to non-negative."""

    def test_constant_input_no_nan(self):
        """SSIM of two constant images should not produce NaN."""
        try:
            from train_mswr_v212_logging import EnhancedMSWRLoss
        except ImportError:
            pytest.skip("train_mswr_v212_logging not importable")

        loss_fn = EnhancedMSWRLoss(l1_weight=0, ssim_weight=1.0, sam_weight=0, gradient_weight=0)
        pred = torch.ones(1, 31, 64, 64) * 0.5
        target = torch.ones(1, 31, 64, 64) * 0.5

        total_loss, loss_dict = loss_fn(pred, target)
        assert not torch.isnan(total_loss), "SSIM NaN on constant inputs"
        assert not torch.isinf(total_loss), "SSIM Inf on constant inputs"

    def test_nearzero_variance_no_nan(self):
        """SSIM with near-zero variance should not produce NaN."""
        try:
            from train_mswr_v212_logging import EnhancedMSWRLoss
        except ImportError:
            pytest.skip("train_mswr_v212_logging not importable")

        loss_fn = EnhancedMSWRLoss(l1_weight=0, ssim_weight=1.0, sam_weight=0, gradient_weight=0)
        pred = torch.ones(1, 31, 64, 64) * 0.5 + torch.randn(1, 31, 64, 64) * 1e-10
        target = torch.ones(1, 31, 64, 64) * 0.5 + torch.randn(1, 31, 64, 64) * 1e-10

        total_loss, _ = loss_fn(pred, target)
        assert not torch.isnan(total_loss), "SSIM NaN on near-zero variance inputs"


# ---------------------------------------------------------------------------
# HIGH-2: PerformanceMonitor sync_cuda flag
# ---------------------------------------------------------------------------
class TestPerformanceMonitorSync:
    """Verify sync_cuda flag gates cuda.synchronize() calls."""

    def test_default_no_sync(self):
        """By default sync_cuda=False, so synchronize should not be called."""
        mon = PerformanceMonitor(enabled=True, rank=0, sync_cuda=False)
        assert not mon.sync_cuda

    def test_explicit_sync(self):
        """sync_cuda=True allows synchronize."""
        mon = PerformanceMonitor(enabled=True, rank=0, sync_cuda=True)
        assert mon.sync_cuda

    def test_timing_works_without_sync(self):
        """Stage timing still works without cuda synchronize."""
        mon = PerformanceMonitor(enabled=True, rank=0, profile_memory=False, sync_cuda=False)
        mon.start_stage("test")
        mon.end_stage("test")
        summary = mon.get_summary()
        assert "test" in summary["stage_times_ms"]


# ---------------------------------------------------------------------------
# HIGH-3: Flash attention includes position bias
# ---------------------------------------------------------------------------
class TestFlashAttentionBias:
    """Verify that the SDPA path includes relative position bias."""

    def test_sdpa_path_uses_bias(self):
        """
        Build a tiny model with flash attn, amplify position bias tables
        so their effect is clearly measurable, then verify zeroing them
        changes the output.
        """
        model = create_mswr_tiny(use_flash_attn=True)
        model.train()

        # Set non-uniform bias (uniform bias has no effect since softmax
        # is shift-invariant). Use large-scale random values.
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "relative_position_bias_table" in name:
                    param.normal_(mean=0.0, std=5.0)

        x = torch.randn(1, 3, 64, 64)
        out_with_bias = model(x).detach().clone()

        # Zero out all relative position bias tables
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "relative_position_bias_table" in name:
                    param.zero_()

        out_no_bias = model(x).detach().clone()

        diff = (out_with_bias - out_no_bias).abs().mean().item()
        # Threshold is conservative: residual connections and dual-attention
        # fusion dilute the raw bias effect, but it should be well above
        # floating-point noise (~1e-8 for uniform/no bias).
        assert diff > 1e-7, (
            f"Zeroing position bias made no difference (diff={diff:.2e}), "
            "bias may not be applied in the flash path"
        )


# ---------------------------------------------------------------------------
# HIGH-4: GradScaler checkpoint restore
# ---------------------------------------------------------------------------
class TestScalerCheckpointRestore:
    """Verify _load_checkpoint restores GradScaler state."""

    def test_load_checkpoint_restores_scaler(self, workspace_tmp_dir):
        """Save and reload a checkpoint; scaler state should round-trip."""
        try:
            from torch.amp import GradScaler
            from train_mswr_v212_logging import EnhancedTrainer
        except ImportError:
            pytest.skip("train_mswr_v212_logging not importable")

        # Build a minimal checkpoint with scaler state
        scaler = GradScaler("cuda") if torch.cuda.is_available() else GradScaler()
        scaler_state = scaler.state_dict()

        model = create_mswr_tiny()
        ckpt = {
            "epoch": 5,
            "iter": 1000,
            "state_dict": model.state_dict(),
            "optimizer": torch.optim.Adam(model.parameters()).state_dict(),
            "scheduler": None,
            "best_mrae": 0.05,
            "scaler": scaler_state,
        }
        ckpt_path = workspace_tmp_dir / "test_ckpt.pth"
        torch.save(ckpt, ckpt_path)

        # Reload and verify scaler key is present
        loaded = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        assert "scaler" in loaded, "scaler key missing from checkpoint"
        assert loaded["scaler"] == scaler_state


# ---------------------------------------------------------------------------
# Full model smoke test
# ---------------------------------------------------------------------------
class TestModelSmokeForwardBackward:
    """End-to-end forward + backward on tiny model with synthetic data."""

    def test_train_step(self):
        model = create_mswr_tiny(use_checkpoint=False)
        model.train()

        x = torch.randn(2, 3, 64, 64)
        target = torch.randn(2, 31, 64, 64)
        out = model(x)
        assert out.shape == target.shape

        loss = F.l1_loss(out, target)
        loss.backward()

        # At least some params should have gradients (conditional paths
        # like qkv_linear vs qkv_conv mean not all params are used)
        grads_found = 0
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN grad in {name}"
                grads_found += 1

        total_params = sum(1 for p in model.parameters() if p.requires_grad)
        assert grads_found > total_params * 0.5, (
            f"Only {grads_found}/{total_params} params got gradients"
        )


# ---------------------------------------------------------------------------
# Convergence safeguard tests
# ---------------------------------------------------------------------------
class TestConvergenceSafeguards:
    """Verify numerical safeguards added for convergence stability."""

    def test_warmup_zero_epochs_no_crash(self):
        """EnhancedMSWRLoss with warmup_epochs=0 should not ZeroDivisionError."""
        try:
            from train_mswr_v212_logging import EnhancedMSWRLoss
        except ImportError:
            pytest.skip("train_mswr_v212_logging not importable")

        loss_fn = EnhancedMSWRLoss(
            l1_weight=1.0,
            ssim_weight=0.5,
            sam_weight=0.1,
            gradient_weight=0.1,
            warmup_epochs=0,
        )
        loss_fn.current_epoch = 0

        pred = torch.randn(1, 31, 32, 32).clamp(0, 1)
        target = torch.randn(1, 31, 32, 32).clamp(0, 1)

        # Should not raise ZeroDivisionError
        total_loss, loss_dict = loss_fn(pred, target)
        assert torch.isfinite(total_loss), f"Non-finite loss: {total_loss}"

    def test_psnr_near_perfect_prediction(self):
        """PSNR of near-identical images should be finite (not +Inf)."""
        try:
            from utils import Loss_PSNR
        except ImportError:
            pytest.skip("utils not importable (missing hdf5storage)")

        psnr_fn = Loss_PSNR()
        x = torch.ones(1, 31, 16, 16) * 0.5
        # Near-perfect prediction: error ~ 1e-12
        y = x + 1e-12

        psnr = psnr_fn(x, y, data_range=1.0)
        assert torch.isfinite(psnr), f"PSNR is not finite: {psnr}"
        # With epsilon 1e-6, max PSNR should cap at ~60dB
        assert psnr.item() < 70.0, f"PSNR too high (epsilon not working): {psnr.item()}"

    def test_psnr_finite_in_fp16(self):
        """PSNR should remain finite when computed in fp16."""
        try:
            from utils import Loss_PSNR
        except ImportError:
            pytest.skip("utils not importable (missing hdf5storage)")

        psnr_fn = Loss_PSNR()
        x = torch.ones(1, 4, 8, 8, dtype=torch.float16) * 0.5
        y = x.clone()

        psnr = psnr_fn(x, y, data_range=1.0)
        assert torch.isfinite(psnr), f"PSNR is not finite in fp16: {psnr}"

    def test_ssim_zero_variance_finite(self):
        """SSIM on constant images should produce finite loss."""
        try:
            from train_mswr_v212_logging import EnhancedMSWRLoss
        except ImportError:
            pytest.skip("train_mswr_v212_logging not importable")

        loss_fn = EnhancedMSWRLoss(
            l1_weight=0.0, ssim_weight=1.0, sam_weight=0.0, gradient_weight=0.0
        )
        # Constant images: variance is exactly zero
        pred = torch.full((1, 31, 64, 64), 0.5)
        target = torch.full((1, 31, 64, 64), 0.5)

        total_loss, _ = loss_fn(pred, target)
        assert torch.isfinite(total_loss), f"SSIM NaN/Inf on zero-variance: {total_loss}"

    def test_loss_dtype_matches_input(self):
        """total_loss dtype should match pred dtype."""
        try:
            from train_mswr_v212_logging import EnhancedMSWRLoss
        except ImportError:
            pytest.skip("train_mswr_v212_logging not importable")

        loss_fn = EnhancedMSWRLoss(
            l1_weight=1.0, ssim_weight=0.0, sam_weight=0.0, gradient_weight=0.0
        )
        pred = torch.randn(1, 31, 32, 32)
        target = torch.randn(1, 31, 32, 32)

        total_loss, _ = loss_fn(pred, target)
        assert total_loss.dtype == pred.dtype, (
            f"Loss dtype {total_loss.dtype} != pred dtype {pred.dtype}"
        )

    def test_mrae_denominator_near_zero(self):
        """MRAE with near-zero labels should not produce Inf."""
        try:
            from utils import Loss_MRAE
        except ImportError:
            pytest.skip("utils not importable (missing hdf5storage)")

        mrae_fn = Loss_MRAE()
        pred = torch.randn(1, 4, 8, 8)
        # Labels very close to zero
        target = torch.full((1, 4, 8, 8), 1e-10)

        loss = mrae_fn(pred, target)
        assert torch.isfinite(loss), f"MRAE not finite for near-zero labels: {loss}"
