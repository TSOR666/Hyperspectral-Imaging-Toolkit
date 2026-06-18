"""
MSWR-Net v2.1.2 - Production-Ready Dual Transformer with CNN Wavelets (FULLY FIXED)
Author : Thierry Silvio Claude Soreze
======================================================================================

CRITICAL FIXES IN THIS VERSION:
1. ✅ Fixed LayerNorm dimension mismatch in Sequential blocks
2. ✅ Added LayerNorm2d wrapper for CNN contexts  
3. ✅ Fixed all normalization handling throughout the model
4. ✅ Enhanced memory efficiency and compute optimizations
5. ✅ Production-quality error handling and validation
6. ✅ Optimized CNN wavelet implementation with caching
7. ✅ Comprehensive performance monitoring
8. ✅ Enhanced gradient checkpointing strategy
9. ✅ Improved multi-GPU and distributed training support
10. ✅ Removed divergent QKV linear path for small feature maps
11. ✅ FIXED BatchNorm/GroupNorm in attention/FFN blocks

LATEST FIXES (Critical):
- Removed the separate qkv_linear small-feature-map path in OptimizedWindowAttention2D.
  A 1x1 Conv2d is mathematically the same per-token projection, and using only
  qkv_conv prevents train/validation parameter divergence across crop sizes.
  
- Fixed BatchNorm/GroupNorm in attention and FFN blocks by using AdaptiveNorm2d
  which properly handles NCHW format instead of expecting NHWC after permutation.

- Added PerformanceMonitor reset at start of each forward pass for clean metrics.

- Improved model compilation to keep reference to original forward for debugging.

- Enhanced test suite to be GPU-aware and handle memory constraints.
  
Performance Optimizations:
- Fused operations where possible
- Optimized memory layout and tensor operations  
- Enhanced caching for wavelet filters and attention patterns
- Improved gradient accumulation and mixed precision
- Better CUDA memory management
- Clean performance metrics per forward pass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import logging
from typing import Optional, Tuple, List, Dict, Literal, Any
from dataclasses import dataclass, asdict
import torch.utils.checkpoint as checkpoint
from einops import rearrange
import time
from collections import defaultdict
import warnings

# Suppress specific warnings for production
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# Setup logging
logger = logging.getLogger(__name__)

_WAVELET_GATE_INIT_BIAS = 4.0


def _wrap_block_with_checkpoint(block: nn.Module) -> nn.Module:
    """Apply activation checkpointing with the supported torch.utils API."""
    original_forward = block.forward

    def checkpointed_forward(x: torch.Tensor) -> torch.Tensor:
        return checkpoint.checkpoint(original_forward, x, use_reentrant=False)

    block.forward = checkpointed_forward
    return block

# ===================== OPTIMIZED CNN WAVELET TRANSFORM =====================

class OptimizedCNNWaveletTransform(nn.Module):
    """
    Highly optimized CNN-based Discrete Wavelet Transform with memory efficiency
    
    Key optimizations:
    - Filter caching with automatic memory management
    - Fused operations for better performance
    - Support for different precision modes
    - Minimal memory allocations during forward pass
    """
    def __init__(self, J: int = 1, wave: str = 'db1', mode: str = 'periodic') -> None:
        super().__init__()
        self.J = J
        self.wave = wave
        self.mode = mode
        
        # Get filter coefficients
        h0, h1 = self._get_filter_coeffs(wave)
        
        # Store as buffers for automatic device management
        self.register_buffer('h0', h0, persistent=False)
        self.register_buffer('h1', h1, persistent=False)
        
        # Enhanced caching with memory limits
        self._filter_cache = {}
        self._cache_size_limit = 16  # Maximum cached filter sets
        self._cache_access_count = defaultdict(int)
    
    def _get_filter_coeffs(self, wave: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get optimized wavelet filter coefficients"""
        filters = {
            'haar': ([0.7071067811865476, 0.7071067811865476],
                    [0.7071067811865476, -0.7071067811865476]),
            'db1': ([0.7071067811865476, 0.7071067811865476],
                   [0.7071067811865476, -0.7071067811865476]),
            'db2': ([0.4829629131445341, 0.8365163037378079, 0.2241438680420134, -0.1294095225512604],
                   [0.1294095225512604, 0.2241438680420134, -0.8365163037378079, 0.4829629131445341]),
            'db3': ([0.3326705529500826, 0.8068915093110925, 0.4598775021184915, -0.1350110200102545, -0.0854412738820267, 0.0352262918857095],
                   [-0.0352262918857095, -0.0854412738820267, 0.1350110200102545, 0.4598775021184915, -0.8068915093110925, 0.3326705529500826]),
            'db4': ([0.2303778133074431, 0.7148465705484058, 0.6308807679358788, -0.0279837694166834, -0.1870348117179132, 0.0308413818353661, 0.0328830116666778, -0.0105974017850021],
                   [0.0105974017850021, 0.0328830116666778, -0.0308413818353661, -0.1870348117179132, 0.0279837694166834, 0.6308807679358788, -0.7148465705484058, 0.2303778133074431])
        }
        
        h0_list, h1_list = filters.get(wave, filters['haar'])
        return torch.tensor(h0_list, dtype=torch.float32), torch.tensor(h1_list, dtype=torch.float32)
    
    def _manage_cache_memory(self) -> None:
        """Intelligent cache management to prevent memory bloat"""
        if len(self._filter_cache) > self._cache_size_limit:
            # Remove least recently used filters
            # Only consider keys that exist in both dicts for consistency
            valid_keys = set(self._filter_cache.keys()) & set(self._cache_access_count.keys())

            if not valid_keys:
                # Edge case: no valid keys, clear everything
                self._filter_cache.clear()
                self._cache_access_count.clear()
                return

            sorted_keys = sorted(valid_keys, key=lambda k: self._cache_access_count[k])
            num_to_remove = len(self._filter_cache) - self._cache_size_limit

            for key in sorted_keys[:num_to_remove]:
                # Safe deletion - check both dicts
                self._filter_cache.pop(key, None)
                self._cache_access_count.pop(key, None)
    
    def _get_conv_filters(self, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Get cached convolution filters with memory management"""
        cache_key = f"{channels}_{self.wave}_{device}_{dtype}"
        
        if cache_key in self._filter_cache:
            self._cache_access_count[cache_key] += 1
            return self._filter_cache[cache_key]
        
        # Create 2D separable filters (more memory efficient)
        h0_2d = self.h0.to(device=device, dtype=dtype).view(-1, 1) * self.h0.to(device=device, dtype=dtype).view(1, -1)  # (K, 1) * (1, K) -> (K, K), broadcast
        h0h1_2d = self.h0.to(device=device, dtype=dtype).view(-1, 1) * self.h1.to(device=device, dtype=dtype).view(1, -1)  # (K, 1) * (1, K) -> (K, K), broadcast
        h1h0_2d = self.h1.to(device=device, dtype=dtype).view(-1, 1) * self.h0.to(device=device, dtype=dtype).view(1, -1)  # (K, 1) * (1, K) -> (K, K), broadcast
        h1_2d = self.h1.to(device=device, dtype=dtype).view(-1, 1) * self.h1.to(device=device, dtype=dtype).view(1, -1)  # (K, 1) * (1, K) -> (K, K), broadcast
        
        # Stack and reshape for grouped convolution
        filters = torch.stack([h0_2d, h0h1_2d, h1h0_2d, h1_2d], dim=0)  # (4, K, K)
        filters = filters.unsqueeze(1).repeat(channels, 1, 1, 1)  # (4, 1, K, K) -> (4*C, 1, K, K)
        
        # Cache with memory management
        self._manage_cache_memory()
        self._filter_cache[cache_key] = filters
        self._cache_access_count[cache_key] = 1
        
        return filters
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Optimized multi-level 2D DWT with minimal memory allocations.

        Args:
            x: Input tensor in NCHW format (B, C, H, W).

        Returns:
            yl: Low-frequency tensor (B, C, H/2^J, W/2^J) when H and W are divisible by 2^J.
            yh: List of high-frequency tensors, each (B, C, 3, H/2^j, W/2^j).
        """
        B, C, H, W = x.shape  # (B, C, H, W)
        device, dtype = x.device, x.dtype
        
        # Get optimized filters
        filters = self._get_conv_filters(C, device, dtype)  # (4*C, 1, K, K)
        
        yh = []
        current = x
        
        for j in range(self.J):
            kernel_size = filters.shape[-1]
            padding = (kernel_size - 1) // 2

            # Even-dimensions assertion: odd H or W would make the IDWT round-trip
            # come back at the wrong spatial size, breaking the residual addition
            # in the wavelet block. The encoder's downsamples preserve this, but
            # callers feeding arbitrary inputs would otherwise hit silent shape
            # mismatches downstream.
            H_in, W_in = current.shape[-2], current.shape[-1]
            assert H_in % 2 == 0 and W_in % 2 == 0, (
                f"OptimizedCNNWaveletTransform requires even H,W; got {H_in}x{W_in} "
                f"at DWT level {j} (wave={self.wave}, J={self.J})."
            )

            # Optimized grouped convolution with proper padding.
            # For mode='periodic' use circular F.pad so longer Daubechies filters
            # (db2/db3/db4) get a clean boundary instead of zero-bleed, which
            # noticeably improves DWT/IDWT round-trip fidelity. For db1/haar the
            # filter is 2-tap so padding is 0 either way (mode is moot).
            if self.mode == 'periodic' and padding > 0:
                current_padded = F.pad(current, [padding, padding, padding, padding], mode='circular')
                coeffs = F.conv2d(current_padded, filters, stride=2, padding=0, groups=C)  # (B, C, H, W) -> (B, 4*C, H/2, W/2)
            else:
                coeffs = F.conv2d(current, filters, stride=2, padding=padding, groups=C)  # (B, C, H, W) -> (B, 4*C, H/2, W/2)
            
            # Grouped conv output order: [ch0_f0..ch0_f3, ch1_f0..ch1_f3, ...]
            # Reshape as (B, C, 4, H2, W2) so dim=2 indexes subbands [LL, LH, HL, HH]
            coeffs = coeffs.view(B, C, 4, coeffs.shape[2], coeffs.shape[3])  # (B, 4*C, H2, W2) -> (B, C, 4, H2, W2)

            yl_new = coeffs[:, :, 0]  # (B, C, H2, W2) — LL subband
            yh_level = coeffs[:, :, 1:]  # (B, C, 3, H2, W2) — [LH, HL, HH]
            yh.append(yh_level)
            
            current = yl_new
        
        return current, yh

class OptimizedCNNInverseWaveletTransform(nn.Module):
    """Optimized CNN-based Inverse DWT with enhanced performance"""
    
    def __init__(self, wave: str = 'db1', mode: str = 'periodic') -> None:
        super().__init__()
        self.wave = wave
        self.mode = mode
        
        # Reuse forward transform for filter coefficients
        self.forward_transform = OptimizedCNNWaveletTransform(J=1, wave=wave, mode=mode)
        self._filter_cache = {}
        self._cache_access_count = defaultdict(int)
        # Bound the cache exactly like the forward transform. Without this the
        # inverse cache grew without limit across distinct (channels, device,
        # dtype) keys; benign for the canonical config but a latent leak for
        # multi-resolution / multi-dtype inference.
        self._cache_size_limit = 16
    
    def _get_conv_filters(self, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Get cached inverse convolution filters"""
        cache_key = f"{channels}_{self.wave}_inv_{device}_{dtype}"
        
        if cache_key in self._filter_cache:
            self._cache_access_count[cache_key] += 1
            return self._filter_cache[cache_key]
        
        # Get synthesis filters (same as analysis for orthogonal wavelets)
        h0 = self.forward_transform.h0.to(device=device, dtype=dtype)
        h1 = self.forward_transform.h1.to(device=device, dtype=dtype)
        
        # Create 2D synthesis filters
        h0_2d = h0.view(-1, 1) * h0.view(1, -1)  # (K, 1) * (1, K) -> (K, K), broadcast
        h0h1_2d = h0.view(-1, 1) * h1.view(1, -1)  # (K, 1) * (1, K) -> (K, K), broadcast
        h1h0_2d = h1.view(-1, 1) * h0.view(1, -1)  # (K, 1) * (1, K) -> (K, K), broadcast
        h1_2d = h1.view(-1, 1) * h1.view(1, -1)  # (K, 1) * (1, K) -> (K, K), broadcast
        
        filters = torch.stack([h0_2d, h0h1_2d, h1h0_2d, h1_2d], dim=0)  # (4, K, K)
        filters = filters.unsqueeze(1).repeat(channels, 1, 1, 1)  # (4, 1, K, K) -> (4*C, 1, K, K)

        # Evict least-recently-used entries before inserting (mirror forward).
        if len(self._filter_cache) >= self._cache_size_limit:
            valid_keys = set(self._filter_cache.keys()) & set(self._cache_access_count.keys())
            if valid_keys:
                sorted_keys = sorted(valid_keys, key=lambda k: self._cache_access_count[k])
                for key in sorted_keys[:len(self._filter_cache) - self._cache_size_limit + 1]:
                    self._filter_cache.pop(key, None)
                    self._cache_access_count.pop(key, None)
            else:
                self._filter_cache.clear()
                self._cache_access_count.clear()

        self._filter_cache[cache_key] = filters
        self._cache_access_count[cache_key] = 1

        return filters
    
    def forward(self, coeffs: Tuple[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """
        Optimized multi-level inverse DWT.

        Args:
            coeffs: Tuple of (yl, yh) where yl is (B, C, H/2^J, W/2^J) and
                yh is a list of (B, C, 3, H/2^j, W/2^j).

        Returns:
            Reconstructed tensor in NCHW format (B, C, H, W) for inputs divisible by 2^J.
        """
        yl, yh = coeffs
        B, C = yl.shape[0], yl.shape[1]  # (B, C, H_low, W_low)
        device, dtype = yl.device, yl.dtype
        
        filters = self._get_conv_filters(C, device, dtype)  # (4*C, 1, K, K)
        current = yl
        
        for j in range(len(yh) - 1, -1, -1):
            yh_level = yh[j]
            H_curr, W_curr = current.shape[-2:]

            # Efficient tensor concatenation and reshaping
            combined = torch.cat([current.unsqueeze(2), yh_level], dim=2)  # (B, C, 1, H, W) + (B, C, 3, H, W) -> (B, C, 4, H, W)
            combined = combined.view(B, 4*C, H_curr, W_curr)  # (B, C, 4, H, W) -> (B, 4*C, H, W)

            # Optimized transposed convolution
            kernel_size = filters.shape[-1]
            padding = (kernel_size - 1) // 2

            if self.mode == 'periodic' and padding > 0:
                # The analysis transform circularly pads by `padding` before the
                # valid stride-2 conv, so the exact inverse is the ADJOINT of
                # that operator: a zero-padding conv_transpose (padding=0)
                # followed by folding the circular-pad overhang back in.
                # Using conv_transpose2d(padding=p) instead (the old code)
                # implements the adjoint of a ZERO-padded analysis - interior
                # pixels reconstruct, but the border band of width ~2K had up
                # to ~67% (db2) / ~120% (db3) relative roundtrip error.
                t = F.conv_transpose2d(
                    combined, filters, stride=2, padding=0,
                    output_padding=0, groups=C
                )  # (B, 4*C, H, W) -> (B, C, 2*H + 2p, 2*W + 2p)
                p = padding
                H2, W2 = 2 * H_curr, 2 * W_curr
                # Fold height overhang (adjoint of circular pad): the rows that
                # came from wrapped padding are added back to the rows they
                # were copies of.
                t[:, :, p:2 * p, :] += t[:, :, H2 + p:H2 + 2 * p, :]
                t[:, :, H2:H2 + p, :] += t[:, :, 0:p, :]
                t = t[:, :, p:H2 + p, :]
                # Fold width overhang.
                t[:, :, :, p:2 * p] += t[:, :, :, W2 + p:W2 + 2 * p]
                t[:, :, :, W2:W2 + p] += t[:, :, :, 0:p]
                current = t[:, :, :, p:W2 + p]
            else:
                current = F.conv_transpose2d(
                    combined, filters, stride=2, padding=padding,
                    output_padding=0, groups=C
                )  # (B, 4*C, H, W) -> (B, C, 2*H, 2*W)

        return current

class WaveletDetailBlock(nn.Module):
    """Lightweight depthwise residual for the high-frequency wavelet bands.

    Operates on a (B, C, 3, H, W) detail tensor (the LH/HL/HH subbands) by
    folding the 3 orientations into the batch dim and applying a per-channel
    (depthwise) 3x3 convolution, added as a residual. Zero-initialized so the
    block is an exact identity at start: enabling it does not perturb a freshly
    initialized model or destabilize from-scratch training, and existing
    checkpoints (which simply lack these params) are unaffected when it is off.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=True)
        self.reset_identity()

    def reset_identity(self) -> None:
        nn.init.zeros_(self.dw.weight)
        if self.dw.bias is not None:
            nn.init.zeros_(self.dw.bias)

    def forward(self, yh_level: torch.Tensor) -> torch.Tensor:
        """yh_level: (B, C, 3, H, W) -> same shape, with a near-identity residual."""
        B, C, n, H, W = yh_level.shape  # (B, C, 3, H, W)
        # Fold the orientation axis into the batch so depthwise conv stays per-channel.
        x = yh_level.permute(0, 2, 1, 3, 4).reshape(B * n, C, H, W)  # (B*3, C, H, W)
        x = self.dw(x)  # (B*3, C, H, W)
        x = x.reshape(B, n, C, H, W).permute(0, 2, 1, 3, 4)  # (B, C, 3, H, W)
        return yh_level + x


# ===================== ENHANCED PERFORMANCE MONITORING =====================

class PerformanceMonitor:
    """Enhanced performance monitoring with detailed profiling"""
    
    def __init__(self, enabled: bool = True, rank: int = 0, profile_memory: bool = True, sync_cuda: bool = False) -> None:
        self.enabled = enabled and rank == 0
        self.sync_cuda = sync_cuda  # Only synchronize when explicit profiling is requested
        self.profile_memory = profile_memory and torch.cuda.is_available()
        self.reset()
    
    def reset(self) -> None:
        self.stage_times = {}
        self.memory_snapshots = {}
        self.operation_counts = defaultdict(int)
        self.tensor_stats = {}
        self._start_times = {}
        self._memory_baseline = None
        
        if self.profile_memory:
            # Do NOT reset CUDA peak-memory stats here: reset() runs at the top
            # of every model forward, which clobbered any external peak-memory
            # measurement (the trainer's end-of-epoch "Peak GPU memory" only
            # covered the final forward). Per-stage deltas below use
            # memory_allocated() snapshots and don't need the peak counter.
            self._memory_baseline = torch.cuda.memory_allocated()
    
    def start_stage(self, stage_name: str) -> None:
        if not self.enabled:
            return

        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

        self._start_times[stage_name] = time.perf_counter()
        
        if self.profile_memory:
            self.memory_snapshots[f"{stage_name}_start"] = {
                'allocated_mb': torch.cuda.memory_allocated() / 1024**2,
                'cached_mb': torch.cuda.memory_reserved() / 1024**2
            }
    
    def end_stage(self, stage_name: str) -> None:
        if not self.enabled or stage_name not in self._start_times:
            return

        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed = (time.perf_counter() - self._start_times[stage_name]) * 1000
        self.stage_times[stage_name] = elapsed
        
        if self.profile_memory:
            self.memory_snapshots[f"{stage_name}_end"] = {
                'allocated_mb': torch.cuda.memory_allocated() / 1024**2,
                'cached_mb': torch.cuda.memory_reserved() / 1024**2,
                'peak_mb': torch.cuda.max_memory_allocated() / 1024**2
            }
        
        del self._start_times[stage_name]
    
    def log_operation(self, op_name: str, tensor_shape: Optional[Tuple[int, ...]] = None) -> None:
        if not self.enabled:
            return
        
        self.operation_counts[op_name] += 1
        
        if tensor_shape:
            if op_name not in self.tensor_stats:
                self.tensor_stats[op_name] = []
            self.tensor_stats[op_name].append(tensor_shape)
    
    def get_summary(self) -> Dict[str, Any]:
        summary = {
            'stage_times_ms': dict(self.stage_times),
            'operation_counts': dict(self.operation_counts),
            'total_time_ms': sum(self.stage_times.values())
        }
        
        if self.profile_memory:
            summary['memory_snapshots'] = dict(self.memory_snapshots)
            if self._memory_baseline is not None:
                current_memory = torch.cuda.memory_allocated()
                summary['memory_delta_mb'] = (current_memory - self._memory_baseline) / 1024**2
        
        return summary

# ===================== ENHANCED CONFIGURATION =====================

@dataclass
class MSWRDualConfig:
    """Production-ready configuration with comprehensive validation"""
    
    # Core Architecture
    input_channels: int = 3
    output_channels: int = 31
    base_channels: int = 64
    channel_expansion: float = 2.0
    num_stages: int = 3
    
    # Attention Configuration
    attention_type: Literal['window', 'dual', 'landmark', 'hybrid'] = 'dual'
    num_heads: int = 8
    window_size: int = 8
    num_landmarks: int = 64
    landmark_pooling: Literal['learned', 'uniform', 'adaptive'] = 'learned'
    local_global_fusion: Literal['adaptive', 'concat', 'add', 'gated'] = 'adaptive'
    # MST++-style spectral (band-to-band) self-attention, added as a parallel
    # branch inside the dual-attention block. The original architecture attends
    # only spatially (window + landmark + SE channel gating) and has no
    # band-to-band attention, which is MST++'s defining inductive bias for
    # RGB->HSI. Default off preserves the original architecture and lets existing
    # checkpoints load unchanged.
    use_spectral_attn: bool = False
    # Number of heads for the spectral (band-to-band) S-MSA branch ONLY, decoupled
    # from num_heads. The shared num_heads=8 makes the spectral attention map
    # block-diagonal (per-head C/heads x C/heads), so at C=64 each of 8 heads sees
    # only an 8x8 sub-block of the 64x64 cross-band covariance. 0 = fall back to
    # num_heads (exact legacy behavior); 1 = FULL-RANK C x C band-to-band attention.
    # Must divide every stage channel width when nonzero. No projection/FFN params
    # are added; only the per-head rescale temperature changes shape (heads,1,1), so
    # full-rank (heads=1) is actually a few params LIGHTER than the 8-head default.
    spectral_attn_heads: int = 0

    # CNN Wavelet Configuration
    use_wavelet: bool = True
    wavelet_type: str = 'db1'
    wavelet_levels: Optional[List[int]] = None
    # When True, the high-frequency detail subbands (LH/HL/HH) are processed by a
    # lightweight depthwise residual block instead of being only multiplicatively
    # gated. The original wavelet path applies attention/FFN to the LL band only,
    # leaving detail bands under-modeled (audit ROUND5). The block is zero-init
    # (near-identity at start) and OFF by default, so it is checkpoint-safe and
    # changes nothing unless explicitly enabled for an ablation.
    wavelet_detail_processing: bool = False
    # NOTE: wavelet_gate_reuse caches a *content-dependent* gate keyed only by
    # spatial dims + stage. Reusing a cached gate across samples is a
    # correctness bug (wrong gate content, wrong batch size on reuse) and
    # freezes the gate after the first batch because it is .detach()'d before
    # caching. Default is now False; enable only with caution.
    wavelet_gate_reuse: bool = False
    
    # Network Architecture
    mlp_ratio: float = 4.0
    ffn_type: Literal['standard', 'gated'] = 'standard'
    # Backward-compatible config knob. The separate linear path was removed in
    # favor of always using qkv_conv so small crops and full-res validation train
    # the same parameters.
    fuse_qkv_small_maps: bool = False
    
    # Regularization
    dropout: float = 0.0
    attention_dropout: float = 0.0
    drop_path: float = 0.1
    layer_scale_init: float = 1e-4
    
    # Optimization & Performance
    use_checkpoint: bool = True
    checkpoint_blocks: Optional[List[int]] = None
    use_flash_attn: bool = True
    compile_model: bool = False
    mixed_precision: bool = True
    memory_efficient: bool = True
    
    # Advanced Features
    use_multi_scale_input: bool = True
    use_skip_init: bool = True
    norm_type: str = 'layer'  # Default to LayerNorm for better compatibility
    performance_monitoring: bool = True
    
    def __post_init__(self) -> None:
        """Enhanced validation with detailed error messages"""
        if self.attention_type not in {'window', 'dual', 'landmark', 'hybrid'}:
            raise ValueError(f"Unsupported attention_type: {self.attention_type!r}")

        if self.wavelet_levels is None:
            self.wavelet_levels = list(range(1, self.num_stages + 1))

        if self.channel_expansion <= 0:
            raise ValueError("channel_expansion must be positive")

        stage_channels = []
        channels = self.base_channels
        for stage_idx in range(self.num_stages):
            stage_channels.append(channels)
            if channels % self.num_heads != 0:
                raise ValueError(
                    f"stage {stage_idx} channels ({channels}) must be divisible by "
                    f"num_heads ({self.num_heads}); base_channels={self.base_channels}, "
                    f"channel_expansion={self.channel_expansion}."
                )
            if self.spectral_attn_heads and channels % self.spectral_attn_heads != 0:
                raise ValueError(
                    f"stage {stage_idx} channels ({channels}) must be divisible by "
                    f"spectral_attn_heads ({self.spectral_attn_heads}); set 0 to reuse "
                    f"num_heads, or 1 for full-rank band-to-band attention."
                )
            if stage_idx < self.num_stages - 1:
                channels = int(channels * self.channel_expansion)

        decoder_channels = stage_channels[-1]
        for expected_channels in reversed(stage_channels[:-1]):
            decoder_channels = int(decoder_channels / self.channel_expansion)
            if decoder_channels != expected_channels:
                raise ValueError(
                    "channel_expansion does not invert the encoder widths cleanly: "
                    f"expected decoder width {expected_channels}, got {decoder_channels} "
                    f"from encoder widths {stage_channels}."
                )
        
        if len(self.wavelet_levels) != self.num_stages:
            logger.warning(
                f"wavelet_levels length ({len(self.wavelet_levels)}) != num_stages ({self.num_stages}). "
                f"Adjusting wavelet_levels."
            )
            if len(self.wavelet_levels) == 1:
                self.wavelet_levels = self.wavelet_levels * self.num_stages
            else:
                self.wavelet_levels = self.wavelet_levels[:self.num_stages]
        
        if self.checkpoint_blocks is None and self.use_checkpoint:
            # Intelligent checkpoint block selection
            n_checkpoint = max(1, int(self.num_stages * 0.6))
            self.checkpoint_blocks = list(range(self.num_stages - n_checkpoint, self.num_stages))
        
        # Validate ranges
        assert 0.0 <= self.dropout <= 1.0, "dropout must be in [0, 1]"
        assert 0.0 <= self.attention_dropout <= 1.0, "attention_dropout must be in [0, 1]"
        assert 0.0 <= self.drop_path <= 1.0, "drop_path must be in [0, 1]"
        assert self.num_heads > 0, "num_heads must be positive"
        assert self.spectral_attn_heads >= 0, "spectral_attn_heads must be >= 0 (0 reuses num_heads)"
        assert self.base_channels > 0, "base_channels must be positive"
        assert self.num_stages > 0, "num_stages must be positive"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# ===================== CRITICAL FIX: LAYERNORM2D FOR CNN CONTEXTS =====================

class LayerNorm2d(nn.Module):
    """
    CRITICAL FIX: LayerNorm wrapper for 2D CNN contexts
    
    This handles the NCHW format from Conv2d layers and applies LayerNorm
    correctly by transposing to NHWC, applying norm, and transposing back.
    
    This is THE KEY FIX for the dimension mismatch error!
    """
    def __init__(self, num_channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)
        self.num_channels = num_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply LayerNorm over channel dimension for NCHW input.

        Args:
            x: Input tensor in NCHW format (B, C, H, W).

        Returns:
            Normalized tensor in NCHW format (B, C, H, W).
        """
        # x is expected to be in NCHW format from Conv2d
        # LayerNorm expects the normalized dimension to be last
        
        # Convert NCHW -> NHWC
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        # Apply LayerNorm (now channels are last)
        x = self.norm(x)  # (B, H, W, C) -> (B, H, W, C)
        # Convert back NHWC -> NCHW
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        
        return x

class AdaptiveNorm2d(nn.Module):
    """
    Adaptive normalization that automatically handles different formats
    and selects the appropriate normalization for CNN contexts.
    """
    def __init__(self, num_channels: int, norm_type: str = 'layer', eps: float = 1e-5) -> None:
        super().__init__()
        self.norm_type = norm_type
        self.num_channels = num_channels
        
        if norm_type == 'layer':
            # Use our fixed LayerNorm2d for CNN contexts
            self.norm = LayerNorm2d(num_channels, eps)
        elif norm_type == 'group':
            # GroupNorm works naturally with NCHW format
            num_groups = min(32, num_channels)
            while num_channels % num_groups != 0 and num_groups > 1:
                num_groups -= 1
            
            # Warn if groups collapsed to 1 (essentially LayerNorm behavior)
            if num_groups == 1:
                logger.warning(f"GroupNorm with {num_channels} channels collapsed to 1 group (LayerNorm-like behavior)")
            
            self.norm = nn.GroupNorm(num_groups, num_channels, eps)
        elif norm_type == 'batch':
            # BatchNorm2d for CNN contexts
            if dist.is_initialized():
                self.norm = nn.SyncBatchNorm(num_channels)
            else:
                self.norm = nn.BatchNorm2d(num_channels)
        else:
            self.norm = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply selected normalization to NCHW input.

        Args:
            x: Input tensor in NCHW format (B, C, H, W).

        Returns:
            Normalized tensor in NCHW format (B, C, H, W).
        """
        return self.norm(x)  # (B, C, H, W) -> (B, C, H, W)

def create_norm_layer(dim: int, norm_type: str = "layer", for_conv: bool = False) -> nn.Module:
    """
    Create normalization layer with proper handling for different contexts
    
    Args:
        dim: Number of channels/features
        norm_type: Type of normalization ('layer', 'group', 'batch', 'none')
        for_conv: If True, creates norm suitable for CNN contexts (NCHW format)
    
    Returns:
        Appropriate normalization module
    """
    if for_conv:
        # For CNN contexts, use AdaptiveNorm2d which handles formats correctly
        return AdaptiveNorm2d(dim, norm_type)
    else:
        # For attention/MLP contexts
        # FIX: BatchNorm and GroupNorm also need special handling for NCHW format
        # So we use AdaptiveNorm2d for them too, which handles the permutations
        if norm_type in ("batch", "group"):
            return AdaptiveNorm2d(dim, norm_type)
        elif norm_type == "layer":
            return nn.LayerNorm(dim)
        else:
            return nn.Identity()

# ===================== OPTIMIZED ATTENTION MODULES =====================

class OptimizedWindowAttention2D(nn.Module):
    """
    Highly optimized window attention with memory efficiency
    
    Key optimizations:
    - Fused QKV computation
    - Optimized memory layout
    - Flash attention integration
    - Reduced tensor operations
    """
    
    def __init__(self, dim: int, num_heads: int, window_size: int,
                 use_flash: bool = True, dropout: float = 0.0,
                 fuse_qkv_small_maps: bool = False, memory_efficient: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_flash = use_flash and hasattr(F, 'scaled_dot_product_attention')
        self.fuse_qkv_small_maps = False
        self.memory_efficient = memory_efficient
        
        # Optimized QKV projection. A 1x1 Conv2d is the same per-token affine
        # projection as Linear on NHWC-flattened pixels, but keeping one module
        # ensures all input sizes update the same parameters.
        self.qkv_conv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        
        self.proj = nn.Conv2d(dim, dim, 1)
        self.dropout = nn.Dropout(dropout)
        
        # Optimized relative position encoding
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)  # ((2*Ws-1)^2, Heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
        # Pre-compute relative position indices
        coords_h = torch.arange(window_size)  # (Ws,)
        coords_w = torch.arange(window_size)  # (Ws,)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # (2, Ws, Ws)
        coords_flatten = torch.flatten(coords, 1)  # (2, Ws*Ws)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, N, N), broadcast
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (N, N, 2)
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # (N, N)
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Windowed attention over NCHW input.

        Args:
            x: Input tensor in NCHW format (B, C, H, W).

        Returns:
            Output tensor in NCHW format (B, C, H, W).
        """
        B, C, H, W = x.shape  # (B, C, H, W)
        
        # Efficient padding for window partitioning
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        
        if pad_h > 0 or pad_w > 0:
            # Pad bottom/right ONLY: the unpad below crops [:H, :W] (top-left),
            # so symmetric padding would shift the attention output down/right
            # by pad//2 pixels relative to the residual branch (this triggered
            # at the deepest stage for default 128x128 crops: 4x4 LL -> 8x8).
            pad_mode = 'reflect'
            if H <= 1 or W <= 1 or pad_h >= H or pad_w >= W:
                pad_mode = 'replicate'
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=pad_mode)  # (B, C, H, W) -> (B, C, H_pad, W_pad)
            H_pad, W_pad = H + pad_h, W + pad_w
        else:
            H_pad, W_pad = H, W

        # SHAPE VALIDATION: Ensure padding produced divisible dimensions
        assert (H_pad % self.window_size == 0) and (W_pad % self.window_size == 0), \
            f"Padding failed: {H_pad}x{W_pad} not divisible by window_size={self.window_size}"

        qkv = self.qkv_conv(x)  # (B, C, H_pad, W_pad) -> (B, 3*C, H_pad, W_pad)
        
        # Efficient window partitioning with optimized tensor operations
        num_windows_h, num_windows_w = H_pad // self.window_size, W_pad // self.window_size
        
        qkv = rearrange(
            qkv,
            'b (k c) (nh wh) (nw ww) -> (b nh nw) (wh ww) k c',
            k=3, c=C, wh=self.window_size, ww=self.window_size,
            nh=num_windows_h, nw=num_windows_w
        )  # (B, 3*C, H_pad, W_pad) -> (B*nh*nw, Ws*Ws, 3, C)
        
        qkv = rearrange(qkv, 'bw n k (h d) -> k bw h n d', h=self.num_heads)  # (B*nh*nw, N, 3, C) -> (3, B*nh*nw, Heads, N, D)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B*nh*nw, Heads, N, D)
        
        # Compute relative position bias (used by both paths)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size * self.window_size, self.window_size * self.window_size, -1)  # (N, N, Heads)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (Heads, N, N)

        # Enhanced attention computation
        if self.use_flash:
            # Pass bias via attn_mask so SDPA can select the best kernel.
            # SDPA is equally valid for inference; previously this path was
            # gated on self.training which forced a slow fp32 manual softmax
            # at eval time.
            # Keep the mask broadcastable as (1, Heads, N, N): casting BEFORE
            # any expand avoids materializing a (B*nh*nw, Heads, N, N) copy
            # under fp16/bf16 autocast (.to() on an expanded view allocates the
            # full broadcast size).
            bias = relative_position_bias.to(dtype=q.dtype).unsqueeze(0)  # (1, Heads, N, N)
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=bias,
                dropout_p=self.dropout.p if self.training else 0.0,
                scale=self.scale
            )  # (B*nh*nw, Heads, N, D)
        else:
            # Manual attention with relative position bias
            # NUMERICAL STABILITY FIX: Always compute attention in float32 to avoid NaN in fp16/bf16
            q_fp32 = q.float()
            k_fp32 = k.float()
            v_fp32 = v.float()

            attn = (q_fp32 @ k_fp32.transpose(-2, -1)) * self.scale  # (B*nh*nw, Heads, N, D) @ (B*nh*nw, Heads, D, N) -> (B*nh*nw, Heads, N, N)

            # Add relative position bias (already computed above)
            attn = attn + relative_position_bias.unsqueeze(0).float()  # (1, Heads, N, N) broadcast across batch

            # NUMERICAL STABILITY FIX: Clamp attention scores to prevent overflow before softmax
            # This prevents NaN when attention scores become extremely large in low-precision modes
            attn = torch.clamp(attn, min=-65504.0, max=65504.0)  # fp16 max value

            attn = F.softmax(attn, dim=-1)  # (B*nh*nw, Heads, N, N), softmax over last dim
            attn = self.dropout(attn)
            attn_out = attn @ v_fp32  # (B*nh*nw, Heads, N, N) @ (B*nh*nw, Heads, N, D) -> (B*nh*nw, Heads, N, D)
            attn_out = attn_out.to(dtype=v.dtype)
        
        # Efficient reverse window partitioning
        attn_out = rearrange(attn_out, 'bw h n d -> bw n (h d)')  # (B*nh*nw, Heads, N, D) -> (B*nh*nw, N, C)
        attn_out = rearrange(
            attn_out,
            '(b nh nw) (wh ww) c -> b c (nh wh) (nw ww)',
            b=B, nh=num_windows_h, nw=num_windows_w,
            wh=self.window_size, ww=self.window_size
        )  # (B*nh*nw, N, C) -> (B, C, H_pad, W_pad)
        
        # Remove padding if applied
        if pad_h > 0 or pad_w > 0:
            attn_out = attn_out[:, :, :H, :W]  # (B, C, H_pad, W_pad) -> (B, C, H, W)
        
        return self.proj(attn_out)  # (B, C, H, W) -> (B, C, H, W)

class OptimizedLandmarkAttention2D(nn.Module):
    """Optimized landmark attention with enhanced efficiency"""
    
    def __init__(self, dim: int, num_heads: int, num_landmarks: int,
                 pooling_type: str = "learned", use_flash: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_landmarks = num_landmarks
        self.pooling_type = pooling_type
        self.scale = self.head_dim ** -0.5
        self.use_flash = use_flash and hasattr(F, 'scaled_dot_product_attention')
        
        # Optimized projections
        self.q_conv = nn.Conv2d(dim, dim, 1, bias=False)
        self.kv_linear = nn.Linear(dim, dim * 2, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.dropout = nn.Dropout(dropout)
        
        # Landmark generation
        if pooling_type == "learned":
            self.landmarks = nn.Parameter(torch.randn(1, num_landmarks, dim) * 0.02)  # (1, L, C)
        elif pooling_type == "adaptive":
            self.landmark_proj = nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.GELU(),
                nn.Linear(dim // 2, num_landmarks * dim)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Landmark attention over NCHW input.

        Args:
            x: Input tensor in NCHW format (B, C, H, W).

        Returns:
            Output tensor in NCHW format (B, C, H, W).
        """
        B, C, H, W = x.shape  # (B, C, H, W)

        # Generate queries efficiently
        q = self.q_conv(x)  # (B, C, H, W) -> (B, C, H, W)
        # Reshape: (B, C, H, W) -> (B, num_heads, H*W, head_dim)
        q = rearrange(q, 'b (h d) H W -> b h (H W) d', h=self.num_heads, H=H, W=W)  # (B, C, H, W) -> (B, Heads, H*W, D)
        
        # Efficient landmark generation
        if self.pooling_type == "learned":
            landmarks = self.landmarks.expand(B, -1, -1)  # (1, L, C) -> (B, L, C), broadcast
        elif self.pooling_type == "adaptive":
            x_pool = F.adaptive_avg_pool2d(x, 1).flatten(1)  # (B, C, 1, 1) -> (B, C)
            landmark_weights = self.landmark_proj(x_pool).view(B, self.num_landmarks, C)  # (B, C) -> (B, L, C)
            x_flat = rearrange(x, 'b c h w -> b (h w) c')  # (B, C, H, W) -> (B, H*W, C)
            
            # Efficient landmark selection
            attn_scores = torch.bmm(landmark_weights.float(), x_flat.transpose(1, 2).float())  # (B, L, C) @ (B, C, H*W) -> (B, L, H*W)
            attn_scores = attn_scores * (C ** -0.5)  # scale in fp32 for stability
            attn_weights = F.softmax(attn_scores, dim=-1)  # (B, L, H*W), softmax over tokens
            landmarks = torch.bmm(attn_weights, x_flat.float()).to(dtype=x_flat.dtype)  # (B, L, H*W) @ (B, H*W, C) -> (B, L, C)
        else:  # uniform
            if H * W < self.num_landmarks:
                raise ValueError(
                    f"uniform landmark pooling requires H*W >= num_landmarks "
                    f"({H*W} < {self.num_landmarks})"
                )
            step = max(1, (H * W) // self.num_landmarks)
            indices = torch.arange(0, H * W, step, device=x.device)[:self.num_landmarks]  # (L,)
            x_flat = rearrange(x, 'b c h w -> b (h w) c')  # (B, C, H, W) -> (B, H*W, C)
            landmarks = x_flat[:, indices]  # (B, L, C)
        
        # Generate keys and values
        kv = self.kv_linear(landmarks)  # (B, L, C) -> (B, L, 2*C)
        kv = kv.reshape(B, self.num_landmarks, 2, self.num_heads, self.head_dim)  # (B, L, 2*C) -> (B, L, 2, Heads, D)
        k, v = kv[:, :, 0].permute(0, 2, 1, 3), kv[:, :, 1].permute(0, 2, 1, 3)  # each (B, Heads, L, D)
        
        # Efficient attention computation
        # SDPA is correct in both training and eval; gating on self.training
        # previously forced an fp32 manual path at inference, doubling latency.
        if self.use_flash:
            dropout_p = self.dropout.p if self.training else 0.0
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, scale=self.scale)  # (B, Heads, H*W, D)
        else:
            attn = (q.float() @ k.float().transpose(-2, -1)) * self.scale  # (B, Heads, H*W, D) @ (B, Heads, D, L) -> (B, Heads, H*W, L)
            attn = F.softmax(attn, dim=-1)  # (B, Heads, H*W, L), softmax over landmarks
            attn = self.dropout(attn)
            out = attn @ v.float()  # (B, Heads, H*W, L) @ (B, Heads, L, D) -> (B, Heads, H*W, D)
            out = out.to(dtype=v.dtype)
        
        out = rearrange(out, 'b h (h_dim w_dim) d -> b (h d) h_dim w_dim', h_dim=H, w_dim=W)  # (B, Heads, H*W, D) -> (B, C, H, W)
        return self.proj(out)  # (B, C, H, W) -> (B, C, H, W)

class SpectralMSA2D(nn.Module):
    """
    Spectral-wise multi-head self-attention (MST++ S-MSA), NCHW in/out.

    Unlike the spatial window/landmark attention, this attends across the
    SPECTRAL (channel) dimension: tokens are pixels and the per-head C x C
    correlation between feature channels is what is modeled. This is the
    dominant inductive bias for RGB->HSI spectral reconstruction and the
    mechanism the original MSWR architecture lacked. A depthwise-conv
    positional embedding on V reinjects local spatial structure, as in MST++.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads}) for spectral attention"
            )
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # 1x1 conv == per-pixel Linear over channels (NCHW-friendly).
        self.to_q = nn.Conv2d(dim, dim, 1, bias=False)
        self.to_k = nn.Conv2d(dim, dim, 1, bias=False)
        self.to_v = nn.Conv2d(dim, dim, 1, bias=False)

        # Per-head learnable temperature (MST++ 'rescale').
        self.rescale = nn.Parameter(torch.ones(num_heads, 1, 1))  # (Heads, 1, 1)

        # Depthwise-conv positional embedding on V (local spatial cue), MST++-style.
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
        )

        self.proj = nn.Conv2d(dim, dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Spectral self-attention over NCHW input.

        Args:
            x: Input tensor in NCHW format (B, C, H, W).

        Returns:
            Output tensor in NCHW format (B, C, H, W).
        """
        B, C, H, W = x.shape  # (B, C, H, W)

        q = self.to_q(x)  # (B, C, H, W)
        k = self.to_k(x)  # (B, C, H, W)
        v = self.to_v(x)  # (B, C, H, W)

        # (B, C, H, W) -> (B, Heads, D, N) with N = H*W, D = C/Heads
        q = rearrange(q, 'b (h d) ph pw -> b h d (ph pw)', h=self.num_heads)  # (B, Heads, D, N)
        k = rearrange(k, 'b (h d) ph pw -> b h d (ph pw)', h=self.num_heads)  # (B, Heads, D, N)
        v_t = rearrange(v, 'b (h d) ph pw -> b h d (ph pw)', h=self.num_heads)  # (B, Heads, D, N)

        # NUMERICAL STABILITY: compute attention in float32 (matches the spatial
        # attention modules). Normalize along the spatial-token dim so the
        # attention is a cosine-similarity over channels (MST++ formulation).
        q_fp32 = F.normalize(q.float(), dim=-1)  # (B, Heads, D, N)
        k_fp32 = F.normalize(k.float(), dim=-1)  # (B, Heads, D, N)
        v_fp32 = v_t.float()  # (B, Heads, D, N)

        # (B, Heads, D, N) @ (B, Heads, N, D) -> (B, Heads, D, D): channel-channel attention
        attn = (q_fp32 @ k_fp32.transpose(-2, -1)) * self.rescale.float()  # (B, Heads, D, D)
        attn = torch.clamp(attn, min=-65504.0, max=65504.0)
        attn = F.softmax(attn, dim=-1)  # (B, Heads, D, D)
        attn = self.dropout(attn)

        out = attn @ v_fp32  # (B, Heads, D, D) @ (B, Heads, D, N) -> (B, Heads, D, N)
        out = out.to(dtype=x.dtype)

        # (B, Heads, D, N) -> (B, C, H, W)
        out = rearrange(out, 'b h d (ph pw) -> b (h d) ph pw', ph=H, pw=W)  # (B, C, H, W)

        # Reinject local spatial structure via depthwise positional embedding on V.
        out = out + self.pos_emb(v)  # (B, C, H, W)
        return self.proj(out)  # (B, C, H, W) -> (B, C, H, W)


class EnhancedDualAttention2D(nn.Module):
    """
    Enhanced dual attention with FIXED normalization handling
    """
    
    def __init__(self, dim: int, config: MSWRDualConfig) -> None:
        super().__init__()
        self.config = config
        self.dim = dim
        self.use_window_attn = config.attention_type in {'window', 'dual', 'hybrid'}
        self.use_landmark_attn = config.attention_type in {'landmark', 'dual', 'hybrid'}
        
        # Use create_norm_layer which now properly handles all norm types
        # For BatchNorm/GroupNorm, it returns AdaptiveNorm2d to handle NCHW format
        self.norm = create_norm_layer(dim, config.norm_type, for_conv=False)
        
        if self.use_window_attn:
            self.window_attn = OptimizedWindowAttention2D(
                dim, config.num_heads, config.window_size,
                config.use_flash_attn, config.attention_dropout,
                config.fuse_qkv_small_maps, config.memory_efficient
            )

        if self.use_landmark_attn:
            self.landmark_attn = OptimizedLandmarkAttention2D(
                dim, config.num_heads, config.num_landmarks,
                config.landmark_pooling, config.use_flash_attn,
                config.attention_dropout
            )

        # "hybrid" means dual spatial attention plus spectral attention.
        # use_spectral_attn remains an orthogonal opt-in for the other modes.
        self.use_spectral_attn = (
            config.attention_type == 'hybrid'
            or getattr(config, 'use_spectral_attn', False)
        )
        if self.use_spectral_attn:
            # Spectral head count is decoupled from the spatial num_heads: 0 reuses
            # num_heads (legacy block-diagonal map), 1 gives a full-rank C x C
            # band-to-band attention map. No added projection params (only the
            # per-head rescale temperature changes size).
            spectral_heads = getattr(config, 'spectral_attn_heads', 0) or config.num_heads
            self.spectral_attn = SpectralMSA2D(dim, spectral_heads, config.attention_dropout)

        # Enhanced fusion mechanisms
        if self.use_window_attn and self.use_landmark_attn and config.local_global_fusion == 'adaptive':
            self.fusion_gate = nn.Sequential(
                nn.Conv2d(dim * 2, dim // 2, 1),
                nn.GELU(),
                nn.Conv2d(dim // 2, dim, 1),
                nn.Sigmoid()
            )
        elif self.use_window_attn and self.use_landmark_attn and config.local_global_fusion == 'gated':
            self.gate_proj = nn.Sequential(
                nn.Conv2d(dim * 2, dim, 1),
                nn.GELU(),
                nn.Conv2d(dim, dim * 2, 1)
            )
        elif self.use_window_attn and self.use_landmark_attn and config.local_global_fusion == 'concat':
            self.fusion_proj = nn.Conv2d(dim * 2, dim, 1)
        
        self.proj = nn.Conv2d(dim, dim, 1)
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1) * config.layer_scale_init)  # (1, C, 1, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Dual attention over NCHW input.

        Args:
            x: Input tensor in NCHW format (B, C, H, W).

        Returns:
            Output tensor in NCHW format (B, C, H, W).
        """
        B, C, H, W = x.shape  # (B, C, H, W)
        identity = x
        
        # Apply normalization with proper format handling
        # FIX: Check if norm expects NCHW (AdaptiveNorm2d) or NHWC (LayerNorm)
        if isinstance(self.norm, AdaptiveNorm2d):
            # AdaptiveNorm2d handles NCHW format internally
            x_norm = self.norm(x)  # (B, C, H, W) -> (B, C, H, W)
        else:
            # LayerNorm expects NHWC
            x_norm = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
            x_norm = self.norm(x_norm)  # (B, H, W, C) -> (B, H, W, C)
            x_norm = x_norm.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        
        local_out = self.window_attn(x_norm) if self.use_window_attn else None
        global_out = self.landmark_attn(x_norm) if self.use_landmark_attn else None

        if local_out is None:
            fused = global_out
        elif global_out is None:
            fused = local_out
        elif self.config.local_global_fusion == 'adaptive':
            gate = self.fusion_gate(torch.cat([local_out, global_out], dim=1))  # (B, 2*C, H, W) -> (B, C, H, W)
            fused = gate * local_out + (1 - gate) * global_out  # broadcast is elementwise (B, C, H, W)
        elif self.config.local_global_fusion == 'gated':
            combined = torch.cat([local_out, global_out], dim=1)  # (B, 2*C, H, W)
            gates = self.gate_proj(combined).chunk(2, dim=1)  # 2x (B, C, H, W)
            gate1, gate2 = torch.sigmoid(gates[0]), torch.sigmoid(gates[1])  # each (B, C, H, W)
            fused = gate1 * local_out + gate2 * global_out  # elementwise (B, C, H, W)
        elif self.config.local_global_fusion == 'concat':
            fused = self.fusion_proj(torch.cat([local_out, global_out], dim=1))  # (B, 2*C, H, W) -> (B, C, H, W)
        else:  # 'add'
            fused = local_out + global_out  # (B, C, H, W)

        # Add the spectral (band-to-band) attention branch in parallel with the
        # spatial fusion. The shared proj + tiny layer-scale gamma below keep the
        # whole sub-layer near-identity at init, so enabling this does not
        # destabilize from-scratch training.
        if self.use_spectral_attn:
            fused = fused + self.spectral_attn(x_norm)  # (B, C, H, W)

        out = self.proj(fused)  # (B, C, H, W) -> (B, C, H, W)
        out = self.gamma * out  # (1, C, 1, 1) broadcast over (B, C, H, W)
        
        return identity + out  # (B, C, H, W)

# ===================== OPTIMIZED FFN =====================

class OptimizedFFN2D(nn.Module):
    """Memory-efficient FFN with gating support"""
    
    def __init__(self, dim: int, mlp_ratio: float = 4.0, ffn_type: str = 'standard',
                 dropout: float = 0.0, memory_efficient: bool = True) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.ffn_type = ffn_type
        self.memory_efficient = memory_efficient
        
        if ffn_type == 'standard':
            layers = [
                nn.Conv2d(dim, hidden_dim, 1),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv2d(hidden_dim, dim, 1),
                nn.Dropout(dropout)
            ]
            self.net = nn.Sequential(*layers)
        
        elif ffn_type == 'gated':
            self.w1 = nn.Conv2d(dim, hidden_dim, 1)
            self.w2 = nn.Conv2d(dim, hidden_dim, 1)
            self.w3 = nn.Conv2d(hidden_dim, dim, 1)
            self.act = nn.GELU()
            self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Feed-forward network over NCHW input.

        Args:
            x: Input tensor in NCHW format (B, C, H, W).

        Returns:
            Output tensor in NCHW format (B, C, H, W).
        """
        if self.ffn_type == 'standard':
            if self.memory_efficient and self.training:
                return checkpoint.checkpoint(self.net, x, use_reentrant=False)  # (B, C, H, W) -> (B, C, H, W)
            else:
                return self.net(x)  # (B, C, H, W) -> (B, C, H, W)
        else:
            # Gated FFN
            if self.memory_efficient and self.training:
                def gated_forward(x):
                    w1 = self.w1(x)  # (B, C, H, W) -> (B, hidden, H, W)
                    w2 = self.w2(x)  # (B, C, H, W) -> (B, hidden, H, W)
                    gated = self.act(w1) * w2  # (B, hidden, H, W), elementwise
                    out = self.w3(gated)  # (B, hidden, H, W) -> (B, C, H, W)
                    return self.dropout(out)  # (B, C, H, W)
                return checkpoint.checkpoint(gated_forward, x, use_reentrant=False)
            else:
                w1 = self.w1(x)  # (B, C, H, W) -> (B, hidden, H, W)
                w2 = self.w2(x)  # (B, C, H, W) -> (B, hidden, H, W)
                gated = self.act(w1) * w2  # (B, hidden, H, W), elementwise
                out = self.w3(gated)  # (B, hidden, H, W) -> (B, C, H, W)
                return self.dropout(out)  # (B, C, H, W)

# ===================== ENHANCED TRANSFORMER BLOCK =====================

class EnhancedWaveletDualTransformerBlock(nn.Module):
    """
    Production-ready transformer block with CNN wavelets and optimizations
    """
    
    def __init__(self, dim: int, config: MSWRDualConfig, stage_idx: int = 0,
                 drop_path: float = 0.0, wavelet_gate_cache: Optional[Dict] = None) -> None:
        super().__init__()
        self.config = config
        self.stage_idx = stage_idx
        self.dim = dim
        self.wavelet_gate_cache = wavelet_gate_cache if wavelet_gate_cache is not None else {}
        
        # CNN Wavelet components
        if (config.use_wavelet and 
            stage_idx < len(config.wavelet_levels) and 
            config.wavelet_levels[stage_idx] > 0):
            
            self.wavelet_level = config.wavelet_levels[stage_idx]
            self.dwt = OptimizedCNNWaveletTransform(
                J=self.wavelet_level, 
                wave=config.wavelet_type
            )
            self.idwt = OptimizedCNNInverseWaveletTransform(
                wave=config.wavelet_type
            )
            
            # Enhanced wavelet gate with better initialization
            self.wavelet_gate = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 1),
                nn.GELU(),
                nn.Conv2d(dim // 4, dim, 1),
                nn.Sigmoid()
            )
            
            # Better initialization for wavelet gate
            nn.init.zeros_(self.wavelet_gate[-2].weight)
            nn.init.constant_(self.wavelet_gate[-2].bias, _WAVELET_GATE_INIT_BIAS)

            # Optional high-frequency detail processing (audit ROUND5). Default
            # off => exact legacy behavior (detail bands only gated).
            self.wavelet_detail = (
                WaveletDetailBlock(dim)
                if getattr(config, 'wavelet_detail_processing', False)
                else None
            )
        else:
            self.dwt = None
            self.wavelet_level = 0
            self.wavelet_detail = None
        
        # Core transformer components
        self.attn = EnhancedDualAttention2D(dim, config)
        block_is_checkpointed = (
            config.use_checkpoint
            and stage_idx in (config.checkpoint_blocks or [])
        )
        self.ffn = OptimizedFFN2D(
            dim, config.mlp_ratio, config.ffn_type, 
            config.dropout,
            config.memory_efficient and config.use_checkpoint and not block_is_checkpointed
        )
        
        # Use standard LayerNorm for FFN (we handle format conversion)
        # Now properly handles BatchNorm/GroupNorm with AdaptiveNorm2d
        self.norm2 = create_norm_layer(dim, config.norm_type, for_conv=False)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.gamma2 = nn.Parameter(torch.ones(1, dim, 1, 1) * config.layer_scale_init)  # (1, C, 1, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transformer block over NCHW input with optional wavelet branch.

        Args:
            x: Input tensor in NCHW format (B, C, H, W).

        Returns:
            Output tensor in NCHW format (B, C, H, W).
        """
        B, C, H, W = x.shape  # (B, C, H, W)
        
        # Input validation
        if C != self.dim:
            raise ValueError(f"Expected {self.dim} channels, got {C}")
        
        # Wavelet processing branch
        if self.dwt is not None:
            min_size = 2 ** self.wavelet_level
            if H >= min_size and W >= min_size:
                return self._wavelet_forward(x)
        
        # Standard transformer path
        x = self.attn(x)  # (B, C, H, W)
        x = x + self.drop_path(self.gamma2 * self._ffn_forward(x))  # gamma2 (1, C, 1, 1) broadcast
        return x
    
    def _wavelet_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Wavelet processing branch.

        Args:
            x: Input tensor in NCHW format (B, C, H, W).

        Returns:
            Output tensor in NCHW format (B, C, H, W).
        """
        # No try/except here on purpose: a silent fallback to the standard path
        # masks real bugs (the prior B-1 reshape error trained for months under
        # such a fallback). If wavelets fail, training should fail loudly so the
        # cause is investigated, not bypassed.
        B, C, H, W = x.shape  # (B, C, H, W)
        cache_key = f"{H}x{W}_stage{self.stage_idx}"

        # Apply wavelet transform
        yl, yh = self.dwt(x)  # yl: (B, C, H/2^J, W/2^J), yh: list of (B, C, 3, H/2^j, W/2^j)

        # Generate or retrieve wavelet gate
        if self.config.wavelet_gate_reuse and cache_key in self.wavelet_gate_cache:
            gate = self.wavelet_gate_cache[cache_key]  # (B, C, H_low, W_low)
        else:
            gate = self.wavelet_gate(yl)  # (B, C, H_low, W_low)
            if self.config.wavelet_gate_reuse:
                self.wavelet_gate_cache[cache_key] = gate.detach()

        # Apply gating to high-frequency components
        yh_gated = []
        for h_coeffs in yh:
            B_h, C_h, n_bands, H_level, W_level = h_coeffs.shape

            # Ensure gate matches spatial dimensions
            if gate.shape[-2:] != (H_level, W_level):
                gate_resized = F.interpolate(
                    gate, size=(H_level, W_level),
                    mode='bilinear', align_corners=False
                )  # (B, C, H_low, W_low) -> (B, C, H_level, W_level)
            else:
                gate_resized = gate  # (B, C, H_level, W_level)

            # Apply gate with proper broadcasting
            gate_expanded = gate_resized.unsqueeze(2)  # (B, C, H_level, W_level) -> (B, C, 1, H_level, W_level)
            h_gated = h_coeffs * gate_expanded  # (B, C, 3, H_level, W_level), broadcast along band dim

            # Optional lightweight detail-band processing (near-identity residual
            # at init). When disabled this branch is skipped entirely.
            if self.wavelet_detail is not None:
                h_gated = self.wavelet_detail(h_gated)  # (B, C, 3, H_level, W_level)

            yh_gated.append(h_gated)

        # Process low-frequency component with attention
        yl_processed = self.attn(yl)  # (B, C, H_low, W_low)
        yl_processed = yl_processed + self.drop_path(self.gamma2 * self._ffn_forward(yl_processed))  # gamma2 broadcast

        # Reconstruct signal
        x_reconstructed = self.idwt((yl_processed, yh_gated))  # (B, C, H, W)

        return x_reconstructed
    
    def _ffn_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply FFN with proper normalization.

        Args:
            x: Input tensor in NCHW format (B, C, H, W).

        Returns:
            Output tensor in NCHW format (B, C, H, W).
        """
        # FIX: Check if norm expects NCHW (AdaptiveNorm2d) or NHWC (LayerNorm)
        if isinstance(self.norm2, AdaptiveNorm2d):
            # AdaptiveNorm2d handles NCHW format internally
            x_norm = self.norm2(x)  # (B, C, H, W)
        else:
            # LayerNorm expects NHWC
            x_norm = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
            x_norm = self.norm2(x_norm)  # (B, H, W, C)
            x_norm = x_norm.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        
        return self.ffn(x_norm)  # (B, C, H, W)

# ===================== DROP PATH =====================

class DropPath(nn.Module):
    """Optimized Drop paths (Stochastic Depth) with better performance"""
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply stochastic depth to NCHW input.

        Args:
            x: Input tensor (B, C, H, W) or higher-rank tensor.

        Returns:
            Output tensor with same shape as input.
        """
        if self.drop_prob == 0. or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)  # (B, 1, ..., 1)
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor  # broadcast over non-batch dims

# ===================== INPUT/OUTPUT MODULES =====================

class EnhancedMultiScaleInputProjection(nn.Module):
    """Optimized multi-scale input processing with fixed grouped convolution"""
    
    def __init__(self, in_channels: int, out_channels: int, memory_efficient: bool = True) -> None:
        super().__init__()
        mid_channels = out_channels // 2
        
        # Fix grouped convolution: ensure groups divides both in_channels and mid_channels
        max_groups = min(in_channels, 8)
        # Find the largest group size that divides both in_channels and mid_channels
        groups = 1
        for g in range(max_groups, 0, -1):
            if in_channels % g == 0 and mid_channels % g == 0:
                groups = g
                break
        
        # Multi-scale convolutions with fixed depthwise separable design
        self.scale1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, groups=groups),
            nn.Conv2d(mid_channels, out_channels, 1),
            nn.GELU()
        )
        
        self.scale2 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 5, padding=2, groups=groups),
            nn.Conv2d(mid_channels, out_channels, 1),
            nn.GELU()
        )
        
        self.scale3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 7, padding=3, groups=groups),
            nn.Conv2d(mid_channels, out_channels, 1),
            nn.GELU()
        )
        
        # Efficient fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, 1),
            nn.GELU()
        )
        
        # Use AdaptiveNorm2d for CNN context
        self.norm = create_norm_layer(out_channels, 'layer', for_conv=True)
        self.memory_efficient = memory_efficient
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Multi-scale projection for NCHW input.

        Args:
            x: Input tensor in NCHW format (B, C_in, H, W).

        Returns:
            Projected tensor in NCHW format (B, C_out, H, W).
        """
        if self.memory_efficient and self.training:
            # Use gradient checkpointing for memory efficiency
            s1 = checkpoint.checkpoint(self.scale1, x, use_reentrant=False)  # (B, C_in, H, W) -> (B, C_out, H, W)
            s2 = checkpoint.checkpoint(self.scale2, x, use_reentrant=False)  # (B, C_in, H, W) -> (B, C_out, H, W)
            s3 = checkpoint.checkpoint(self.scale3, x, use_reentrant=False)  # (B, C_in, H, W) -> (B, C_out, H, W)
        else:
            s1 = self.scale1(x)  # (B, C_in, H, W) -> (B, C_out, H, W)
            s2 = self.scale2(x)  # (B, C_in, H, W) -> (B, C_out, H, W)
            s3 = self.scale3(x)  # (B, C_in, H, W) -> (B, C_out, H, W)
        
        fused = self.fusion(torch.cat([s1, s2, s3], dim=1))  # (B, 3*C_out, H, W) -> (B, C_out, H, W)
        
        # Apply normalization (AdaptiveNorm2d handles format automatically)
        fused = self.norm(fused)  # (B, C_out, H, W)
        
        return fused

class EnhancedOutputProjection(nn.Module):
    """Enhanced output projection with attention and efficiency"""
    
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        
        self.proj1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, in_channels * 2, 1),
            nn.GELU()
        )
        
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1)
        )
        
        # Enhanced channel attention
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, max(in_channels // 16, 8), 1),
            nn.GELU(),
            nn.Conv2d(max(in_channels // 16, 8), in_channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(in_channels, 1, 7, padding=3),
            nn.Sigmoid()
        )
        
        self.proj2 = nn.Conv2d(in_channels, out_channels, 1)
        
        # Better initialization
        nn.init.constant_(self.proj2.weight, 0.01)
        if self.proj2.bias is not None:
            nn.init.zeros_(self.proj2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Output projection for NCHW input.

        Args:
            x: Input tensor in NCHW format (B, C_in, H, W).

        Returns:
            Output tensor in NCHW format (B, C_out, H, W).
        """
        feat = self.proj1(x)  # (B, C_in, H, W) -> (B, 2*C_in, H, W)
        refined = self.refine(feat)  # (B, 2*C_in, H, W) -> (B, C_in, H, W)
        refined = refined + x  # (B, C_in, H, W)
        
        # Dual attention
        ca = self.channel_attn(refined)  # (B, C_in, H, W) -> (B, C_in, 1, 1)
        sa = self.spatial_attn(refined)  # (B, C_in, H, W) -> (B, 1, H, W)
        refined = refined * ca * sa  # broadcast over spatial and channel dims
        
        return self.proj2(refined)  # (B, C_in, H, W) -> (B, C_out, H, W)

# ===================== MAIN ENHANCED MODEL =====================

class IntegratedMSWRNet(nn.Module):
    """
    MSWR-Net v2.1.2 - Production-Ready Dual Transformer with CNN Wavelets (FULLY FIXED)
    
    CRITICAL FIXES IN THIS VERSION:
    - Fixed LayerNorm dimension mismatch in Sequential blocks
    - Added LayerNorm2d and AdaptiveNorm2d for proper CNN context handling
    - Enhanced memory efficiency and error handling
    - Optimized CNN wavelet implementation
    - Better gradient checkpointing strategy
    - Comprehensive performance monitoring
    """
    
    def __init__(self, config: Optional[MSWRDualConfig] = None) -> None:
        super().__init__()
        
        if config is None:
            config = MSWRDualConfig()
        
        self.config = config
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        
        if self.rank == 0:
            logger.info("Initializing Enhanced MSWR-Net v2.1.2 with CNN wavelets (FULLY FIXED)")
        
        # Enhanced performance monitoring
        self.perf_monitor = PerformanceMonitor(
            config.performance_monitoring, self.rank, profile_memory=True
        )
        
        # Shared wavelet gate cache for memory efficiency
        self.wavelet_gate_cache = {} if config.wavelet_gate_reuse else None
        
        # Enhanced input projection with fallback for compatibility
        if config.use_multi_scale_input:
            try:
                self.input_proj = EnhancedMultiScaleInputProjection(
                    config.input_channels,
                    config.base_channels,
                    config.memory_efficient and config.use_checkpoint,
                )
            except ValueError as e:
                # Fallback to simple projection if grouped conv fails
                logger.warning(f"Multi-scale input projection failed ({e}), using simple projection")
                self.input_proj = nn.Sequential(
                    nn.Conv2d(config.input_channels, config.base_channels, 3, padding=1),
                    nn.GELU()
                )
        else:
            self.input_proj = nn.Sequential(
                nn.Conv2d(config.input_channels, config.base_channels, 3, padding=1),
                nn.GELU()
            )
        
        # Build optimized encoder
        self.encoder_stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        channels = config.base_channels
        dpr = [x.item() for x in torch.linspace(0, config.drop_path, config.num_stages * 2)]  # (2*num_stages,)
        
        for i in range(config.num_stages):
            block = EnhancedWaveletDualTransformerBlock(
                channels, config, stage_idx=i, drop_path=dpr[i],
                wavelet_gate_cache=self.wavelet_gate_cache
            )
            
            # Intelligent gradient checkpointing
            if config.use_checkpoint and i in (config.checkpoint_blocks or []):
                block = _wrap_block_with_checkpoint(block)
            
            self.encoder_stages.append(block)
            
            # CRITICAL FIX: Downsampling layers with proper normalization for CNN context
            if i < config.num_stages - 1:
                next_channels = int(channels * config.channel_expansion)
                downsample = nn.Sequential(
                    nn.Conv2d(channels, next_channels, 2, stride=2),
                    # Use AdaptiveNorm2d for CNN context which handles NCHW format correctly
                    create_norm_layer(next_channels, config.norm_type, for_conv=True)
                )
                self.downsamples.append(downsample)
                channels = next_channels
        
        # Build optimized decoder
        self.decoder_stages = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        
        for i in range(config.num_stages - 1):
            in_ch = channels
            out_ch = int(channels / config.channel_expansion)
            
            # Enhanced upsampling with proper normalization for CNN context
            upsample = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
                create_norm_layer(out_ch, config.norm_type, for_conv=True),
                nn.GELU()
            )
            self.upsamples.append(upsample)
            
            # Optimized skip connections with proper normalization
            skip_conv = nn.Sequential(
                nn.Conv2d(out_ch * 2, out_ch, 1),
                create_norm_layer(out_ch, config.norm_type, for_conv=True),
                nn.GELU()
            )
            self.skip_connections.append(skip_conv)
            
            # Decoder blocks
            decoder_idx = config.num_stages + i
            block = EnhancedWaveletDualTransformerBlock(
                out_ch, config, stage_idx=decoder_idx,
                drop_path=dpr[decoder_idx],
                wavelet_gate_cache=self.wavelet_gate_cache
            )
            
            if config.use_checkpoint and decoder_idx in (config.checkpoint_blocks or []):
                block = _wrap_block_with_checkpoint(block)
            
            self.decoder_stages.append(block)
            channels = out_ch
        
        # Enhanced output projection
        self.output_proj = EnhancedOutputProjection(config.base_channels, config.output_channels)

        # Learnable input skip connection
        self.input_skip = nn.Conv2d(config.input_channels, config.output_channels, 1)

        # Apply enhanced (Kaiming) initialization FIRST so it does not overwrite
        # the per-layer soft-identity inits applied below. Previously this call
        # ran AFTER EnhancedOutputProjection.__init__ had set proj2.weight=0.01
        # and after we set input_skip.weight=0.01, silently wiping both — so the
        # model never actually started from the documented near-identity head.
        self.apply(self._init_weights)

        # Re-apply soft-identity init on the output head and the input skip.
        # These small initial weights make the model start close to a learned
        # linear RGB->HSI mapping and avoid wild early-training outputs.
        nn.init.constant_(self.output_proj.proj2.weight, 0.01)
        if self.output_proj.proj2.bias is not None:
            nn.init.zeros_(self.output_proj.proj2.bias)
        if config.use_skip_init:
            nn.init.constant_(self.input_skip.weight, 0.01)
            if self.input_skip.bias is not None:
                nn.init.zeros_(self.input_skip.bias)

        # Re-apply the near-identity wavelet-gate init after Kaiming.
        for module in self.modules():
            gate = getattr(module, "wavelet_gate", None)
            if gate is not None:
                nn.init.zeros_(gate[-2].weight)
                nn.init.constant_(gate[-2].bias, _WAVELET_GATE_INIT_BIAS)
            # Restore the zero (identity) init of any wavelet detail block, which
            # the global Kaiming apply() above would otherwise overwrite.
            detail = getattr(module, "wavelet_detail", None)
            if isinstance(detail, WaveletDetailBlock):
                detail.reset_identity()
        
        # Model compilation for optimization
        if config.compile_model and hasattr(torch, 'compile'):
            try:
                # Keep reference to original forward for debugging
                self._original_forward = self.forward
                self._compiled_forward = torch.compile(self.forward, mode='default')
                self.forward = self._compiled_forward
                if self.rank == 0:
                    logger.info("Model compiled successfully")
            except Exception as e:
                if self.rank == 0:
                    logger.warning(f"Model compilation failed: {e}")
                # Keep original forward on failure
                pass
        
        if self.rank == 0:
            self._log_model_info()

    def _required_spatial_multiple(self) -> int:
        """Return the input multiple needed by encoder wavelet/downsample stages."""
        required_power = 0

        if self.config.use_wavelet:
            for stage_idx, level in enumerate(self.config.wavelet_levels or []):
                if stage_idx >= self.config.num_stages:
                    break
                if level > 0:
                    required_power = max(required_power, stage_idx + level)

        return 2 ** required_power

    def _pad_to_required_multiple(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Pad H/W on the bottom/right so all wavelet stages receive valid sizes."""
        multiple = self._required_spatial_multiple()
        if multiple <= 1:
            return x, (0, 0)

        H, W = x.shape[-2:]
        pad_h = (multiple - H % multiple) % multiple
        pad_w = (multiple - W % multiple) % multiple

        if pad_h == 0 and pad_w == 0:
            return x, (0, 0)

        pad_mode = "reflect" if pad_h < H and pad_w < W else "replicate"
        x = F.pad(x, (0, pad_w, 0, pad_h), mode=pad_mode)
        return x, (pad_h, pad_w)
    
    def _init_weights(self, m: nn.Module) -> None:
        """Enhanced weight initialization"""
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.ones_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def _log_model_info(self) -> None:
        """Log comprehensive model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        
        logger.info("="*70)
        logger.info("ENHANCED MSWR-NET v2.1.2 MODEL INFORMATION (FULLY FIXED)")
        logger.info("="*70)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Model memory: {(param_size + buffer_size) / (1024**2):.2f} MB")
        logger.info(f"Architecture: {self.config.num_stages} stages, {self.config.base_channels} base channels")
        logger.info(f"Attention: {self.config.attention_type}")
        logger.info(f"Wavelet: {self.config.use_wavelet} ({self.config.wavelet_type})")
        logger.info(f"Flash Attention: {self.config.use_flash_attn}")
        logger.info(f"Normalization: {self.config.norm_type}")
        logger.info("✅ LayerNorm dimension mismatch FIXED")
        logger.info("✅ Single qkv_conv projection path (no small-map divergence)")
        logger.info("✅ BatchNorm/GroupNorm in attention/FFN blocks FIXED")
        logger.info("✅ All normalization layers properly configured")
        logger.info("="*70)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for MSWR-Net.

        Args:
            x: Input tensor in NCHW format (B, C_in, H, W).

        Returns:
            Output tensor in NCHW format (B, C_out, H, W).
        """
        # Reset performance monitor for clean per-forward metrics
        if self.config.performance_monitoring:
            self.perf_monitor.reset()
        
        # Performance monitoring
        self.perf_monitor.start_stage("total")
        
        # Input validation with detailed error messages
        B, C, H, W = x.shape  # (B, C_in, H, W)
        if C != self.config.input_channels:
            raise ValueError(
                f"Input channels mismatch: expected {self.config.input_channels}, got {C}"
            )

        min_size = 2 ** (self.config.num_stages + 2)
        if H < min_size or W < min_size:
            raise ValueError(
                f"Input size ({H}, {W}) too small for {self.config.num_stages} stages. "
                f"Minimum required: ({min_size}, {min_size})"
            )

        orig_H, orig_W = H, W
        x, (pad_h, pad_w) = self._pad_to_required_multiple(x)
        H, W = x.shape[-2:]
        
        # Clear cache for new forward pass
        if self.wavelet_gate_cache is not None:
            self.wavelet_gate_cache.clear()
        
        # Generate input skip connection
        input_skip = self.input_skip(x)  # (B, C_in, H, W) -> (B, C_out, H, W)
        
        # Input projection
        self.perf_monitor.start_stage("input_proj")
        x = self.input_proj(x)  # (B, C_in, H, W) -> (B, C_base, H, W)
        self.perf_monitor.end_stage("input_proj")
        self.perf_monitor.log_operation("input_proj", x.shape)
        
        # Encoder path
        encoder_features = []
        for i, stage in enumerate(self.encoder_stages):
            self.perf_monitor.start_stage(f"encoder_{i}")
            
            try:
                x = stage(x)  # (B, C_i, H_i, W_i)
                encoder_features.append(x)  # list of (B, C_i, H_i, W_i)
                self.perf_monitor.log_operation(f"encoder_{i}", x.shape)
                
                if i < len(self.downsamples):
                    x = self.downsamples[i](x)  # (B, C_i, H_i, W_i) -> (B, C_{i+1}, H_i/2, W_i/2)
                    self.perf_monitor.log_operation(f"downsample_{i}", x.shape)
                
            except Exception as e:
                logger.error(f"Error in encoder stage {i}: {e}")
                raise RuntimeError(f"Encoder stage {i} failed: {e}") from e
            
            self.perf_monitor.end_stage(f"encoder_{i}")
        
        # Decoder path
        for i, (upsample, skip_conv, stage) in enumerate(
            zip(self.upsamples, self.skip_connections, self.decoder_stages)
        ):
            self.perf_monitor.start_stage(f"decoder_{i}")
            
            try:
                # Upsampling
                x = upsample(x)  # (B, C_{i+1}, H_i/2, W_i/2) -> (B, C_i, H_i, W_i)
                
                # Skip connection with proper size matching
                skip_idx = -(i + 2)
                skip_feat = encoder_features[skip_idx]  # (B, C_i, H_i, W_i)
                
                if x.shape[-2:] != skip_feat.shape[-2:]:
                    x = F.interpolate(
                        x, size=skip_feat.shape[-2:],
                        mode='bilinear', align_corners=False
                    )  # (B, C_i, H_i, W_i) -> (B, C_i, H_i, W_i)
                
                # Combine and process
                x = skip_conv(torch.cat([x, skip_feat], dim=1))  # (B, 2*C_i, H_i, W_i) -> (B, C_i, H_i, W_i)
                x = stage(x)  # (B, C_i, H_i, W_i)
                
                self.perf_monitor.log_operation(f"decoder_{i}", x.shape)
                
            except Exception as e:
                logger.error(f"Error in decoder stage {i}: {e}")
                raise RuntimeError(f"Decoder stage {i} failed: {e}") from e
            
            self.perf_monitor.end_stage(f"decoder_{i}")
        
        # Output projection
        self.perf_monitor.start_stage("output")
        x = self.output_proj(x)  # (B, C_base, H, W) -> (B, C_out, H, W)
        
        # Add input skip connection
        if x.shape[-2:] != input_skip.shape[-2:]:
            input_skip = F.interpolate(
                input_skip, size=x.shape[-2:],
                mode='bilinear', align_corners=False
            )  # (B, C_out, H_in, W_in) -> (B, C_out, H, W)
        
        x = x + input_skip  # (B, C_out, H, W)
        if pad_h or pad_w:
            x = x[:, :, :orig_H, :orig_W].contiguous()

        self.perf_monitor.end_stage("output")
        self.perf_monitor.end_stage("total")
        
        return x
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get detailed performance summary"""
        return self.perf_monitor.get_summary()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'total_memory_mb': (param_size + buffer_size) / (1024 * 1024),
            'architecture': {
                'num_stages': self.config.num_stages,
                'base_channels': self.config.base_channels,
                'attention_type': self.config.attention_type,
                'use_spectral_attn': (
                    self.config.attention_type == 'hybrid'
                    or getattr(self.config, 'use_spectral_attn', False)
                ),
                'use_wavelet': self.config.use_wavelet,
                'wavelet_type': self.config.wavelet_type,
                'wavelet_levels': list(self.config.wavelet_levels or []),
                'use_flash_attn': self.config.use_flash_attn,
                'norm_type': self.config.norm_type
            },
            'optimization': {
                'use_checkpoint': self.config.use_checkpoint,
                'memory_efficient': self.config.memory_efficient,
                'mixed_precision': self.config.mixed_precision
            }
        }
    
    def __repr__(self) -> str:
        return (
            f"IntegratedMSWRNet v2.1.2 (FULLY FIXED)\n"
            f"  Stages: {self.config.num_stages}\n"
            f"  Channels: {self.config.base_channels}\n"
            f"  Attention: {self.config.attention_type}\n"
            f"  Wavelet: {self.config.use_wavelet} ({self.config.wavelet_type})\n"
            f"  Parameters: {sum(p.numel() for p in self.parameters()):,}\n"
            f"  Memory Efficient: {self.config.memory_efficient}\n"
            f"  ✅ All normalization issues FIXED\n"
            f"  ✅ Single qkv_conv projection path\n"
            f"  ✅ BatchNorm/GroupNorm compatibility FIXED\n"
        )

# ===================== FACTORY FUNCTIONS =====================

def create_mswr_tiny(**kwargs: Any) -> IntegratedMSWRNet:
    """Create tiny MSWR model optimized for speed"""
    config = MSWRDualConfig(
        base_channels=32, num_stages=2, num_heads=4,
        window_size=4, num_landmarks=32,
        **kwargs
    )
    return IntegratedMSWRNet(config)

def create_mswr_small(**kwargs: Any) -> IntegratedMSWRNet:
    """Create small MSWR model balanced for speed/quality"""
    config = MSWRDualConfig(
        base_channels=48, num_stages=3, num_heads=6,
        window_size=8, num_landmarks=49,
        **kwargs
    )
    return IntegratedMSWRNet(config)

def create_mswr_base(**kwargs: Any) -> IntegratedMSWRNet:
    """Create base MSWR model (recommended)"""
    config = MSWRDualConfig(
        base_channels=64, num_stages=3, num_heads=8,
        window_size=8, num_landmarks=64,
        **kwargs
    )
    return IntegratedMSWRNet(config)

def create_mswr_large(**kwargs: Any) -> IntegratedMSWRNet:
    """Create large MSWR model for maximum quality"""
    config = MSWRDualConfig(
        base_channels=96, num_stages=4, num_heads=12,
        window_size=12, num_landmarks=128,
        **kwargs
    )
    return IntegratedMSWRNet(config)

# Public API
__all__ = [
    'MSWRDualConfig',
    'IntegratedMSWRNet', 
    'OptimizedCNNWaveletTransform',
    'OptimizedCNNInverseWaveletTransform',
    'create_mswr_tiny',
    'create_mswr_small',
    'create_mswr_base', 
    'create_mswr_large',
    'PerformanceMonitor',
]

if __name__ == "__main__":
    print("="*80)
    print("MSWR-Net v2.1.2 with CNN Wavelets - FULLY FIXED VERSION")
    print("="*80)
    print("✅ Fixed LayerNorm dimension mismatch in Sequential blocks")
    print("✅ Added LayerNorm2d wrapper for CNN contexts")
    print("✅ Added AdaptiveNorm2d for automatic format handling")
    print("✅ Fixed all downsampling and upsampling normalization")
    print("✅ Fixed grouped convolution parameter errors")
    print("✅ Removed divergent QKV linear path for small feature maps")
    print("✅ Enhanced memory efficiency")
    print("✅ Production-quality error handling")
    print("="*80)
    
    # Comprehensive functionality test
    try:
        print("\nRunning comprehensive functionality tests...")
        
        # Determine device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Testing on device: {device}")
        
        # Test 1: LayerNorm with exact scenario from the error
        print("\nTest 1: LayerNorm with batch size 20 (from error scenario)")
        model_layer = create_mswr_base(use_wavelet=True, norm_type='layer')
        model_layer = model_layer.to(device)
        x = torch.randn(20, 3, 128, 128, device=device)  # (B=20, C=3, H=128, W=128)
        
        with torch.no_grad():
            y = model_layer(x)  # (B, C_out, H, W)
        
        print(f"✅ LayerNorm test passed: Input {x.shape} -> Output {y.shape}")
        
        # Test 2: GroupNorm (FIXED)
        print("\nTest 2: GroupNorm configuration (FIXED)")
        model_group = create_mswr_base(use_wavelet=True, norm_type='group')
        model_group = model_group.to(device)
        
        with torch.no_grad():
            y = model_group(x)  # (B, C_out, H, W)
        
        print(f"✅ GroupNorm test passed: Input {x.shape} -> Output {y.shape}")
        
        # Test 3: BatchNorm (FIXED)
        print("\nTest 3: BatchNorm configuration (FIXED)")
        model_batch = create_mswr_base(use_wavelet=True, norm_type='batch')
        model_batch = model_batch.to(device)
        model_batch.eval()  # BatchNorm requires eval mode for inference
        
        with torch.no_grad():
            y = model_batch(x)  # (B, C_out, H, W)
        
        print(f"✅ BatchNorm test passed: Input {x.shape} -> Output {y.shape}")
        
        # Test 4: Different input sizes INCLUDING PROBLEMATIC SIZES
        print("\nTest 4: Various input sizes (including small feature maps)")
        test_sizes = [
            (1, 3, 64, 64),    # Small
            (4, 3, 128, 128),  # Medium (triggers 16x16 in stage 2)
            (2, 3, 256, 256),  # Large (triggers 32x32 in stage 2)
        ]
        
        # Add extra large only if enough memory
        if device.type == 'cuda':
            try:
                # Test with small tensor first
                test_tensor = torch.randn(1, 3, 512, 512, device=device)  # (B=1, C=3, H=512, W=512)
                del test_tensor
                test_sizes.append((1, 3, 512, 512))  # Extra large (reduced batch)
            except torch.cuda.OutOfMemoryError:
                print("  (Skipping 512x512 test due to memory constraints)")
        
        for size in test_sizes:
            x_test = torch.randn(*size, device=device)  # size = (B, C, H, W)
            with torch.no_grad():
                y_test = model_layer(x_test)  # (B, C_out, H, W)
            print(f"✅ Size {size} -> {y_test.shape} (stage 2 would be {size[2]//8}x{size[3]//8})")
            del x_test, y_test  # Free memory
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Test 5: Specific test for the former small-map divergence case
        print("\nTest 5: Specific test for small feature maps (16x16 case)")
        x_problem = torch.randn(1, 3, 128, 128, device=device)  # (B=1, C=3, H=128, W=128)
        model_test = create_mswr_base(
            use_wavelet=True, 
            fuse_qkv_small_maps=True  # Backward-compatible no-op; qkv_conv is always used
        )
        model_test = model_test.to(device)
        
        with torch.no_grad():
            y_problem = model_test(x_problem)  # (B, C_out, H, W)
        print(f"✅ Small feature map test passed: {x_problem.shape} -> {y_problem.shape}")
        
        # Test 6: Training mode simulation (only on CPU or small batch)
        print("\nTest 6: Training mode with gradient flow")
        model_layer.train()
        batch_size = 1 if device.type == 'cuda' else 2
        x_train = torch.randn(batch_size, 3, 128, 128, device=device, requires_grad=True)  # (B, C=3, H=128, W=128)
        y_train = model_layer(x_train)  # (B, C_out, H, W)
        loss = y_train.mean()  # () scalar
        loss.backward()
        print("✅ Training mode test passed with gradient flow")
        
        # Clean up GPU memory
        del model_layer, model_group, model_batch, model_test
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Model info (create fresh model for stats)
        model_info = create_mswr_base()
        info = model_info.get_model_info()
        print("\n📊 Model Statistics:")
        print(f"   Total parameters: {info['total_parameters']:,}")
        print(f"   Model memory: {info['total_memory_mb']:.1f} MB")
        
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
        print("="*80)
        print("\nThe model is now fully fixed and ready for training.")
        print("Key fixes applied:")
        print("1. LayerNorm2d wrapper for CNN contexts")
        print("2. AdaptiveNorm2d for automatic format handling")
        print("3. Proper normalization in all Sequential blocks")
        print("4. Removed divergent QKV linear branch")
        print("5. Fixed BatchNorm/GroupNorm in attention/FFN blocks")
        print("6. Enhanced error handling and validation")
        print("\nYou can now use this model in your training script without errors.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease check:")
        print("1. PyTorch version compatibility (>= 1.12 recommended)")
        print("2. Required dependencies are installed (einops)")
        print("3. CUDA availability if using GPU")
