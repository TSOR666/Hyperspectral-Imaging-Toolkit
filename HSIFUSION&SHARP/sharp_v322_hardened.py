"""
SHARP v3.2.2 - Production-hardened with Fixed Local Window Attention
Author: Thierry Silvio Claude Soreze

Fixed in this version:
- Local window attention now handles different q/k/v dimensions correctly
- Proper dimension tracking for RBF-transformed queries/keys
- Separate index tensors for k and v gathering operations

All v3.2.2 features maintained:
- Streaming sparse attention with O(B * H * N * k * D) footprint
- FP16-safe padding sentinels with dtype awareness
- Query tiling for bounded memory O(BH * q_block * k_block)
- Configurable sparsity_ratio and rbf_centers_per_head
- All previous stability and performance fixes
"""

import logging
import math
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def robust_version_parse(version_str: str) -> Tuple[int, int, int]:
    """Robustly parse a version string into ``(major, minor, patch)``."""

    try:
        base_version = version_str.split('+')[0].split('-')[0]
        parts = [p for p in base_version.split('.') if p.isdigit()]
        parts = (parts + ['0', '0', '0'])[:3]
        return tuple(int(p) for p in parts)
    except Exception:
        logger.exception("Failed to parse version string '%s'", version_str)
        return (0, 0, 0)

# Version detection
TORCH_VERSION = robust_version_parse(torch.__version__)
HAS_TORCH_2_0 = TORCH_VERSION >= (2, 0, 0)
HAS_TORCH_2_2 = TORCH_VERSION >= (2, 2, 0)

# Feature gates
HAS_DYNAMO = HAS_TORCH_2_0 and hasattr(torch, '_dynamo')
HAS_COMPILE = HAS_TORCH_2_0 and hasattr(torch, 'compile')

# ROCm detection
IS_ROCM = getattr(torch.version, "hip", None) is not None
HAS_ROCM_5_7_PLUS = False
if IS_ROCM:
    try:
        _hip_major, _hip_minor = map(int, torch.version.hip.split(".")[:2])
        HAS_ROCM_5_7_PLUS = (_hip_major > 5) or (_hip_major == 5 and _hip_minor >= 7)
    except Exception:
        HAS_ROCM_5_7_PLUS = False

def has_cuda_11_8_or_newer():
    """Check if CUDA 11.8+ is available for stable compilation."""
    if not torch.cuda.is_available() or IS_ROCM:
        return False
    
    try:
        cuda_version = torch.version.cuda
        if cuda_version:
            major, minor = map(int, cuda_version.split('.')[:2])
            return (major > 11) or (major == 11 and minor >= 8)
    except:
        pass
    return False

HAS_CUDA_11_8 = has_cuda_11_8_or_newer()

# ============================================================================
# Enums for configuration
# ============================================================================

class KeyRBFMode(str, Enum):
    """RBF key projection modes."""
    MEAN = "mean"      # Original mean+expand (rank-1, strong regularization)
    LINEAR = "linear"  # Learned linear projection (more capacity)
    NONE = "none"      # No RBF transform on keys

# ============================================================================
# Fixed Sparse Attention Components
# ============================================================================

def sparse_attention_topk_streaming(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    sparsity_ratio: float = 0.9, scale: Optional[float] = None, 
    block_size: int = 2048, max_tokens: int = 8192,
    window_size: int = 49, k_cap: Optional[int] = 1024,
    q_block_size: int = 1024
) -> torch.Tensor:
    """
    Streaming top-k attention that avoids allocating the full (N x N) scores.
    
    v3.2.2: 
    - Auto-disables k_cap when sparsity_ratio=0 for true dense attention
    - Short-circuits to efficient dense SDPA when k_keep >= N
    - Fixed to handle different q/k/v dimensions from RBF transforms
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.size(-1))

    # Input validation
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")
    if window_size <= 0:
        raise ValueError(f"window_size must be > 0, got {window_size}")
    if q_block_size <= 0:
        raise ValueError(f"q_block_size must be > 0, got {q_block_size}")
    
    # Clamp sparsity_ratio to valid range
    if not (0.0 <= sparsity_ratio <= 1.0):
        logger.warning(f"sparsity_ratio={sparsity_ratio} out of [0,1]; clamping.")
        sparsity_ratio = float(min(1.0, max(0.0, sparsity_ratio)))

    BH, N, D_q = q.shape
    D_k = k.shape[-1]  # May differ from D_q
    D_v = v.shape[-1]  # Original dimension, may differ from both D_q and D_k
    
    # Handle edge case where sparsity_ratio >= 1.0 (keep nothing)
    if sparsity_ratio >= 1.0:
        # Return zeros with v's dimension
        return torch.zeros(BH, N, D_v, device=q.device, dtype=q.dtype)
    
    # v3.2.2: Auto-disable k_cap when sparsity_ratio=0 for true dense attention
    if sparsity_ratio == 0.0:
        k_cap = None
        logger.debug("Auto-disabled k_cap for sparsity_ratio=0 (dense attention)")
    
    k_keep = max(1, int(N * (1 - sparsity_ratio)))
    
    # Cap k_keep to prevent memory spikes (graceful handling of k_cap)
    if k_cap is not None and k_cap > 0:
        k_keep = min(k_keep, k_cap)
    
    # v3.2.2: Short-circuit to dense attention when k_keep == N
    if k_keep >= N and N <= max_tokens:
        # Use efficient dense attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(scores.to(torch.float32), dim=-1).to(scores.dtype)
        return torch.matmul(attn, v)
    
    # Safety cap for very large sequences
    if N > max_tokens:
        # Fallback to local window attention (ensure odd window_size)
        logger.warning(
            f"Sequence length N={N} exceeds max_tokens={max_tokens}. "
            f"Falling back to local window attention with window_size={window_size}. "
            f"This changes the attention pattern from sparse to local windowed. "
            f"Consider increasing max_tokens or using a different attention mechanism "
            f"for very long sequences."
        )

        adjusted_window_size = window_size if window_size % 2 == 1 else window_size - 1
        if adjusted_window_size != window_size:
            logger.debug(f"Adjusted window_size from {window_size} to {adjusted_window_size} (must be odd)")

        return local_window_attention(q, k, v, adjusted_window_size, scale)
    
    # Initialize output tensor with v's dimension
    out = q.new_zeros(BH, N, D_v)
    
    # v3.2.2: Create BH_device once, outside the loop
    BH_device = torch.arange(BH, device=v.device, dtype=torch.long)
    v_flat = v.reshape(BH * N, D_v)  # Use D_v here, not D_q
    
    # Process queries in blocks to further reduce memory
    for q_start in range(0, N, q_block_size):
        q_end = min(q_start + q_block_size, N)
        q_chunk = q[:, q_start:q_end, :]  # (BH, Q, D_q)
        q_len = q_end - q_start
        
        # Initialize running top-k for this query chunk
        topk_vals = q.new_full((BH, q_len, k_keep), float("-inf"))
        topk_idx = q.new_zeros((BH, q_len, k_keep), dtype=torch.long)

        # Process key blocks to avoid NxN allocation
        for k_start in range(0, N, block_size):
            k_end = min(k_start + block_size, N)
            block_len = k_end - k_start

            # Scores for this key block: (BH, q_len, block_len)
            scores_blk = torch.matmul(q_chunk, k[:, k_start:k_end, :].transpose(-2, -1)) * scale
            
            # Top-k within this block first
            k_blk = min(k_keep, block_len)
            blk_vals, blk_pos = torch.topk(scores_blk, k_blk, dim=-1, sorted=False)
            blk_idx = blk_pos + k_start
            
            # Merge previous top-k with block top-k
            merge_vals = torch.cat([topk_vals, blk_vals], dim=-1)
            merge_idx = torch.cat([topk_idx, blk_idx], dim=-1)
            
            # Select overall top-k from merged candidates
            new_vals, sel = torch.topk(merge_vals, k_keep, dim=-1, sorted=False)
            new_idx = torch.gather(merge_idx, -1, sel)
            
            topk_vals, topk_idx = new_vals, new_idx

        # Softmax over the selected top-k values (stabilized)
        topk_vals_stable = topk_vals - topk_vals.max(dim=-1, keepdim=True).values
        attn = F.softmax(topk_vals_stable.to(torch.float32), dim=-1).to(topk_vals_stable.dtype)

        # v3.2.2: Optimized gather without 4D expand (BH_device created outside loop)
        # Create flat indices for gathering
        idx_offset = BH_device.view(BH, 1, 1) * N  # (BH, 1, 1)
        idx_flat = (topk_idx + idx_offset).reshape(-1)  # (BH*q_len*k_keep,)
        
        # Gather and reshape with v's dimension
        v_gathered = v_flat.index_select(0, idx_flat).view(BH, q_len, k_keep, D_v)

        # Compute output for this query chunk
        out_chunk = torch.einsum('bnk,bnkd->bnd', attn, v_gathered)
        
        # Write to output
        out[:, q_start:q_end, :] = out_chunk
    
    return out


def local_window_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                          window_size: int, scale: float) -> torch.Tensor:
    """Vectorized local window attention for very large sequences.
    
    Fixed to handle different q/k/v dimensions that can occur with RBF transforms.
    """
    BH, N, D_q = q.shape
    D_k = k.shape[-1]  # k dimension (may differ from q due to RBF)
    D_v = v.shape[-1]  # v dimension (original head_dim)
    
    # Early return for trivial window size
    if window_size <= 1:
        return v
    
    # Ensure window_size is odd for symmetric window
    if window_size % 2 == 0:
        window_size -= 1
    half = window_size // 2
    
    # Build (N, W) neighbor indices
    base = torch.arange(N, device=q.device).unsqueeze(1)
    offsets = torch.arange(-half, half + 1, device=q.device).unsqueeze(0)
    idx = (base + offsets).clamp_(0, N - 1)
    
    # Gather k and v windows
    W = idx.size(1)
    
    # Create separate index tensors for k and v with their respective dimensions
    idx_b_k = idx.unsqueeze(0).unsqueeze(-1).expand(BH, -1, -1, D_k)
    idx_b_v = idx.unsqueeze(0).unsqueeze(-1).expand(BH, -1, -1, D_v)
    
    k_src = k.unsqueeze(2).expand(-1, -1, W, -1)
    v_src = v.unsqueeze(2).expand(-1, -1, W, -1)
    
    k_win = torch.gather(k_src, 1, idx_b_k)
    v_win = torch.gather(v_src, 1, idx_b_v)
    
    # Compute scores using q's dimension
    scores = torch.einsum('bnd,bnwd->bnw', q, k_win) * scale
    attn = F.softmax(scores.to(torch.float32), dim=-1).to(scores.dtype)
    
    # Apply attention to values
    out = torch.einsum('bnw,bnwd->bnd', attn, v_win)
    return out

# ============================================================================
# Fixed Components with v3.2.2 improvements
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization with fixed caching."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.register_buffer('_weight_fp16', None, persistent=False)
        self.register_buffer('_weight_bf16', None, persistent=False)
        
    def train(self, mode=True):
        """Override train to reset caches when switching modes."""
        super().train(mode)
        if mode:
            self._weight_fp16 = None
            self._weight_bf16 = None
        return self
        
    def clear_eval_caches(self):
        """Clear evaluation caches. Call after loading new weights in eval mode."""
        self._weight_fp16 = None
        self._weight_bf16 = None
        return self
        
    def forward(self, x):
        out_dtype = x.dtype
        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        
        if self.training or out_dtype == torch.float32:
            return (x * self.weight).to(out_dtype)
        else:
            # Keep multiply in float32 for better accuracy
            if out_dtype == torch.float16:
                if self._weight_fp16 is None:
                    self._weight_fp16 = self.weight.detach().half()
                weight_dtype = self._weight_fp16
            elif out_dtype == torch.bfloat16:
                if self._weight_bf16 is None:
                    self._weight_bf16 = self.weight.detach().bfloat16()
                weight_dtype = self._weight_bf16
            else:
                weight_dtype = self.weight.to(out_dtype)
            return (x * weight_dtype.to(x.dtype)).to(out_dtype)

class ChannelRMSNorm(nn.Module):
    """Channel RMSNorm with fixed caching."""
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(dim, 1, 1))
        self.register_buffer('_weight_fp16', None, persistent=False)
        self.register_buffer('_weight_bf16', None, persistent=False)
        
    def train(self, mode=True):
        """Override train to reset caches when switching modes."""
        super().train(mode)
        if mode:
            self.clear_eval_caches()
        return self
        
    def clear_eval_caches(self):
        """Clear evaluation caches."""
        self._weight_fp16 = None
        self._weight_bf16 = None
        return self
        
    def forward(self, x):
        dtype = x.dtype
        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.float()
        var = x.pow(2).mean(1, keepdim=True)
        x = x * torch.rsqrt(var.clamp(min=self.eps))
        
        if self.weight.shape[0] != x.shape[1]:
            raise ValueError(f"ChannelRMSNorm expects {self.weight.shape[0]} channels, got {x.shape[1]}")
        
        if self.training or dtype == torch.float32:
            return (x * self.weight).to(dtype)
        else:
            if dtype == torch.float16:
                if self._weight_fp16 is None:
                    self._weight_fp16 = self.weight.detach().half()
                weight_dtype = self._weight_fp16
            elif dtype == torch.bfloat16:
                if self._weight_bf16 is None:
                    self._weight_bf16 = self.weight.detach().bfloat16()
                weight_dtype = self._weight_bf16
            else:
                weight_dtype = self.weight.to(dtype)
            return (x * weight_dtype.to(x.dtype)).to(dtype)

# ============================================================================
# Optimized Attention Mechanisms with v3.2.2 fixes
# ============================================================================

class VectorizedWindowedSparsemax(nn.Module):
    """Windowed Sparsemax with v3.2.2 fixed pad sentinel."""
    def __init__(self, window_size: int = 64, dim: int = -1, pad_value: Optional[float] = None):
        super().__init__()
        self.window_size = window_size
        self.dim = dim
        self.pad_value = pad_value
        
    def forward(self, scores):
        if self.dim != -1:
            scores = scores.transpose(self.dim, -1)
    
        *batch_dims, seq_len = scores.shape
    
        if seq_len <= self.window_size:
            output = linear_sparsemax(scores, dim=-1)
            if self.dim != -1:
                output = output.transpose(self.dim, -1)
            return output
    
        pad_len = (self.window_size - seq_len % self.window_size) % self.window_size
        if pad_len > 0:
            # v3.2.2: Fixed pad sentinel for fp16/bf16
            if self.pad_value is not None:
                pad_val = self.pad_value
            else:
                pad_val = -1e4 if scores.dtype in (torch.float16, torch.bfloat16) else -1e9
            
            scores = F.pad(scores, (0, pad_len), value=pad_val)
            padded_len = seq_len + pad_len
        else:
            padded_len = seq_len
        
        scores = scores.reshape(-1, padded_len // self.window_size, self.window_size)
        output = linear_sparsemax(scores, dim=-1)
        output = output.reshape(*batch_dims, padded_len)
        
        if pad_len > 0:
            output = output[..., :seq_len]
        
        row_sums = output.sum(dim=-1, keepdim=True)
        mask = row_sums > 1e-12
        output = torch.where(mask, output / row_sums.clamp_min(1e-12), output)
        
        if self.dim != -1:
            output = output.transpose(self.dim, -1)
        
        return output

class MultiScaleAttention(nn.Module):
    """Multi-scale attention with local and global branches."""
    def __init__(self, dim: int, num_heads: int = 8, 
                 local_window: int = 7, use_conv_proj: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.local_window = local_window
        
        groups = get_optimal_groups(dim, dim * 3)
        if use_conv_proj:
            self.qkv = nn.Conv2d(dim, dim * 3, 1, groups=groups, bias=False)
            self.proj = nn.Conv2d(dim, dim, 1)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=False)
            self.proj = nn.Linear(dim, dim)
            
        self.local_attn = nn.Sequential(
            nn.Conv2d(dim, dim, local_window, padding=local_window//2, groups=dim),
            ChannelRMSNorm(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1)
        )
        
        self.mix = nn.Parameter(torch.zeros(1))
        nn.init.uniform_(self.mix, -0.1, 0.1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        local_out = self.local_attn(x)
        
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, H*W)
        q, k, v = qkv.unbind(1)
        q, k, v = [t.permute(0, 1, 3, 2) for t in (q, k, v)]
        
        global_out = sdpa_unified(q, k, v, scale=self.scale)
        global_out = global_out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        global_out = self.proj(global_out)
        
        mix = torch.sigmoid(self.mix)
        return local_out * mix + global_out * (1 - mix)

class OptimizedSparseAttention(nn.Module):
    """Sparse attention with v3.2.2 improvements including configurable key RBF mode."""
    def __init__(self, dim: int, num_heads: int = 8, 
                 sparsity_ratio: float = 0.9, use_topk: bool = True,
                 use_sparsemax: bool = True, use_rbf: bool = True,
                 block_size: int = 2048, max_tokens: int = 8192,
                 window_size: int = 49, k_cap: int = 1024,
                 q_block_size: int = 1024, rbf_centers_per_head: int = 32,
                 key_rbf_mode: str = KeyRBFMode.MEAN,
                 sparsemax_pad_value: Optional[float] = None):
        super().__init__()

        # Proper input validation (not assert - works with python -O)
        if not (0.0 <= sparsity_ratio <= 1.0):
            raise ValueError(
                f"sparsity_ratio must be in range [0.0, 1.0], got {sparsity_ratio}. "
                f"Use 0.0 for dense attention, 1.0 for maximum sparsity."
            )

        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")

        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")

        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads}). "
                f"Consider using dim={dim - (dim % num_heads) + num_heads}"
            )

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sparsity_ratio = sparsity_ratio
        self.use_topk = use_topk
        self.use_sparsemax = use_sparsemax
        self.use_rbf = use_rbf
        self.block_size = block_size
        self.max_tokens = max_tokens
        self.window_size = window_size if window_size % 2 == 1 else window_size - 1
        self.key_rbf_mode = KeyRBFMode(key_rbf_mode)
        
        # Graceful k_cap handling
        if k_cap is not None and k_cap <= 0:
            k_cap = None
        self.k_cap = k_cap
        self.q_block_size = q_block_size
        
        groups = get_optimal_groups(dim, dim * 3)
        self.qkv = nn.Conv2d(dim, dim * 3, 1, groups=groups, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1)
        
        self.threshold = nn.Parameter(torch.zeros(1, num_heads, 1))
        
        if use_sparsemax:
            self.sparsemax = VectorizedWindowedSparsemax(
                window_size=64, 
                pad_value=sparsemax_pad_value
            )
            
        if use_rbf:
            self.centers_per_head = rbf_centers_per_head
            self.rbf_kernel = RBFKernel(
                input_dim=self.head_dim,
                num_centers=self.centers_per_head,
                output_scale=True,
                learnable_centers=True
            )
            
            # v3.2.2: Configurable key projection
            if self.key_rbf_mode == KeyRBFMode.LINEAR:
                self.k_proj = nn.Linear(self.head_dim, self.centers_per_head, bias=False)
                nn.init.orthogonal_(self.k_proj.weight)
        
    def _apply_rbf(self, q, k):
        """Apply RBF transform with configurable key handling (v3.2.2)."""
        B, H, N, D = q.shape
        
        # Process keys based on mode
        if self.key_rbf_mode == KeyRBFMode.NONE:
            # No RBF at all - return originals
            return q, k
        
        # Process queries
        q_flat = q.reshape(B * H, N, D).reshape(-1, D)
        rbf_out = self.rbf_kernel(q_flat)
        q_rbf = rbf_out.view(B, H, N, self.centers_per_head)
        
        if self.key_rbf_mode == KeyRBFMode.LINEAR:
            # Learned linear projection
            k_flat = k.reshape(B * H * N, D)
            k_proj = self.k_proj(k_flat).view(B, H, N, self.centers_per_head)
            return q_rbf, k_proj
        else:  # KeyRBFMode.MEAN
            # Original mean+expand (rank-1 regularization)
            k_proj = k.mean(dim=-1, keepdim=True).expand(-1, -1, -1, self.centers_per_head)
            return q_rbf, k_proj
        
    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, N)
        q, k, v = qkv.unbind(1)
        q, k, v = [t.permute(0, 1, 3, 2).contiguous() for t in (q, k, v)]
        
        if self.use_rbf and self.key_rbf_mode != KeyRBFMode.NONE:
            q, k = self._apply_rbf(q, k)
        
        if self.use_topk:
            # Vectorized sparse attention with streaming top-k
            q_flat = q.reshape(B * self.num_heads, N, -1)
            k_flat = k.reshape(B * self.num_heads, N, -1)
            v_flat = v.reshape(B * self.num_heads, N, -1)
            
            out_flat = sparse_attention_topk_streaming(
                q_flat, k_flat, v_flat, 
                sparsity_ratio=self.sparsity_ratio, 
                scale=None,  # Let it compute scale based on q's dimension
                block_size=self.block_size,
                max_tokens=self.max_tokens,
                window_size=self.window_size,
                k_cap=self.k_cap,
                q_block_size=self.q_block_size
            )
            out = out_flat.reshape(B, self.num_heads, N, -1)
        else:
            # Dynamic scale based on q's dimension after RBF
            scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(q.size(-1)))
            
            if self.use_sparsemax:
                scores_flat = scores.reshape(B * self.num_heads, N, -1)
                attn_flat = self.sparsemax(scores_flat)
                attn = attn_flat.reshape(B, self.num_heads, N, -1)
            else:
                threshold = self.threshold.sigmoid() * scores.abs().mean(dim=-1, keepdim=True)
                mask = scores.abs() > threshold
                scores_masked = scores.masked_fill(~mask, float('-inf'))
                
                # v3.2.2: Fix diagonal for rows with no valid entries (safe in-place write)
                row_has_valid = mask.any(dim=-1)
                N = scores.shape[-1]
                idx = torch.arange(N, device=scores.device, dtype=torch.long)
                
                # Replace diagonal only where a row had no valid entries
                scores_masked[..., idx, idx] = torch.where(
                    row_has_valid,
                    scores_masked[..., idx, idx],
                    scores[..., idx, idx]
                )
                
                attn = F.softmax(scores_masked.to(torch.float32), dim=-1).to(scores_masked.dtype)
            
            out = torch.matmul(attn, v)
        
        out = out.permute(0, 1, 3, 2).contiguous().reshape(B, C, H, W)
        return self.proj(out)

# ============================================================================
# Enhanced Blocks
# ============================================================================

class EnhancedDualAttentionBlock(nn.Module):
    """Dual attention with optimized memory usage."""
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.,
                 drop_path: float = 0., use_checkpoint: bool = False,
                 sparse_config: Optional[dict] = None):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        self.norm = ChannelRMSNorm(dim)
        
        self.attn1 = MultiScaleAttention(dim, num_heads)
        
        # Use sparse config if provided
        sparse_kwargs = sparse_config or {}
        self.attn2 = OptimizedSparseAttention(
            dim, num_heads, 
            use_sparsemax=True, 
            use_rbf=True,
            use_topk=True,
            sparsity_ratio=sparse_kwargs.get('sparsity_ratio', 0.9),
            rbf_centers_per_head=sparse_kwargs.get('rbf_centers_per_head', 32),
            key_rbf_mode=sparse_kwargs.get('key_rbf_mode', KeyRBFMode.MEAN),
            sparsemax_pad_value=sparse_kwargs.get('sparsemax_pad_value', None),
            **{k: v for k, v in sparse_kwargs.items() 
               if k not in ['sparsity_ratio', 'rbf_centers_per_head', 'key_rbf_mode', 'sparsemax_pad_value']}
        )
        
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            ChannelRMSNorm(dim),
            nn.Conv2d(dim, mlp_hidden * 2, 1),
            nn.GELU(),
            nn.Conv2d(mlp_hidden * 2, mlp_hidden, 1, groups=get_optimal_groups(mlp_hidden * 2, mlp_hidden)),
            nn.GELU(), 
            nn.Conv2d(mlp_hidden, dim, 1)
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def _forward_impl(self, x):
        x = x + self.drop_path(self.attn1(self.norm(x)))
        x = x + self.drop_path(self.attn2(x))
        x = x + self.drop_path(self.mlp(x))
        return x
        
    def forward(self, x):
        if self.use_checkpoint and self.training:
            if HAS_TORCH_2_0:
                return checkpoint.checkpoint(self._forward_impl, x, use_reentrant=False)
            else:
                return checkpoint.checkpoint(self._forward_impl, x)
        return self._forward_impl(x)

class ImprovedCrossAttentionFusion(nn.Module):
    """Cross-attention with gated residual."""
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.norm1 = ChannelRMSNorm(dim)
        self.norm2 = ChannelRMSNorm(dim)
        
        groups = get_optimal_groups(dim, dim)
        self.q = nn.Conv2d(dim, dim, 1, groups=groups, bias=False)
        self.kv = nn.Conv2d(dim, dim * 2, 1, groups=groups, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1)
        
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, skip):
        B, C, H, W = x.shape
        
        x_norm = self.norm1(x)
        skip_norm = self.norm2(skip)
        
        q = self.q(x_norm).reshape(B, self.num_heads, self.head_dim, H*W).permute(0, 1, 3, 2)
        kv = self.kv(skip_norm).reshape(B, 2, self.num_heads, self.head_dim, H*W)
        k, v = kv.unbind(1)
        k, v = [t.permute(0, 1, 3, 2) for t in (k, v)]
        
        out = sdpa_unified(q, k, v, scale=self.scale)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        out = self.proj(out)
        
        gate = self.gate(torch.cat([x, out], dim=1))
        return x * (1 - gate) + out * gate

# ============================================================================
# Model Configuration with v3.2.2 additions
# ============================================================================

@dataclass
class SHARPv32Config:
    in_channels: int = 3
    out_channels: int = 31
    base_dim: int = 64
    depths: List[int] = (2, 2, 6, 2)
    heads: List[int] = (4, 8, 16, 32)
    mlp_ratios: List[float] = (4., 4., 4., 4.)
    drop_path_rate: float = 0.1
    use_checkpoint: bool = False
    compile_mode: Optional[str] = "reduce-overhead"
    # Streaming attention parameters
    sparse_block_size: int = 2048
    sparse_max_tokens: int = 8192
    sparse_window_size: int = 49
    sparse_k_cap: int = 1024
    sparse_q_block_size: int = 1024
    sparse_sparsity_ratio: float = 0.9
    # RBF kernel parameters
    rbf_centers_per_head: int = 32
    # v3.2.2 additions
    key_rbf_mode: str = KeyRBFMode.MEAN
    sparsemax_pad_value: Optional[float] = None
    # EMA parameters
    ema_update_every: int = 1

    def __post_init__(self):
        self.base_dim = (self.base_dim + 7) // 8 * 8
        for i, (d, h) in enumerate(zip([self.base_dim * (2**i) for i in range(len(self.depths))], self.heads)):
            if d % h != 0:
                raise ValueError(f"Stage {i}: dim {d} not divisible by heads {h}")
        
        # v3.2.2: Auto-disable k_cap when sparsity_ratio=0
        if self.sparse_sparsity_ratio == 0.0:
            self.sparse_k_cap = None
            logger.info("Auto-disabled sparse_k_cap for sparsity_ratio=0 (dense attention)")
        elif self.sparse_k_cap is not None and self.sparse_k_cap <= 0:
            self.sparse_k_cap = None

# ============================================================================
# Improved SHARP Model with v3.2.2 enhancements
# ============================================================================

class SHARPv32(nn.Module):
    """SHARP v3.2.2 with all audit fixes and optimizations applied."""
    def __init__(self, config: SHARPv32Config):
        super().__init__()
        self.config = config
        
        # v3.2.2: Cache for spectral basis by device/dtype
        self._spectral_basis_cache = {}
        
        # Fixed: Apply get_optimal_groups to stem
        groups_stem = get_optimal_groups(config.in_channels, config.base_dim // 2) if config.in_channels > 1 else 1
        self.stem = nn.Sequential(
            nn.Conv2d(config.in_channels, config.base_dim // 2, 3, padding=1, groups=groups_stem),
            ChannelRMSNorm(config.base_dim // 2),
            nn.GELU(),
            nn.Conv2d(config.base_dim // 2, config.base_dim, 3, padding=1, 
                     groups=get_optimal_groups(config.base_dim // 2, config.base_dim)),
            ChannelRMSNorm(config.base_dim),
            nn.GELU()
        )
        
        dpr = torch.linspace(0, config.drop_path_rate, sum(config.depths)).tolist()
        
        self.stages = nn.ModuleList()
        self.downsample = nn.ModuleList()
        
        # Prepare sparse attention config
        sparse_config = {
            'block_size': config.sparse_block_size,
            'max_tokens': config.sparse_max_tokens,
            'window_size': config.sparse_window_size,
            'k_cap': config.sparse_k_cap,
            'q_block_size': config.sparse_q_block_size,
            'sparsity_ratio': config.sparse_sparsity_ratio,
            'rbf_centers_per_head': config.rbf_centers_per_head,
            'key_rbf_mode': config.key_rbf_mode,
            'sparsemax_pad_value': config.sparsemax_pad_value
        }
        
        cur_depth = 0
        for i in range(len(config.depths)):
            dim = config.base_dim * (2 ** i)
            
            blocks = nn.ModuleList([
                EnhancedDualAttentionBlock(
                    dim=dim,
                    num_heads=config.heads[i],
                    mlp_ratio=config.mlp_ratios[i],
                    drop_path=dpr[cur_depth + j],
                    use_checkpoint=config.use_checkpoint,
                    sparse_config=sparse_config
                )
                for j in range(config.depths[i])
            ])
            self.stages.append(blocks)
            cur_depth += config.depths[i]
            
            if i < len(config.depths) - 1:
                downsample = nn.Sequential(
                    nn.Conv2d(dim, dim * 2, 2, stride=2),
                    ChannelRMSNorm(dim * 2)
                )
                self.downsample.append(downsample)
        
        self.upsample = nn.ModuleList()
        self.fusion = nn.ModuleList()
        
        for i in range(len(config.depths) - 1, 0, -1):
            up_dim = config.base_dim * (2 ** (i - 1))
            down_dim = config.base_dim * (2 ** i)
            
            self.upsample.append(nn.Sequential(
                nn.ConvTranspose2d(down_dim, up_dim, 2, stride=2),
                ChannelRMSNorm(up_dim),
                nn.GELU()
            ))
            
            self.fusion.append(ImprovedCrossAttentionFusion(up_dim, config.heads[i-1]))
        
        self.head = nn.Sequential(
            nn.Conv2d(config.base_dim, config.base_dim, 3, padding=1),
            ChannelRMSNorm(config.base_dim),
            nn.GELU(),
            nn.Conv2d(config.base_dim, config.out_channels * 2, 1)
        )
        
        # Register spectral basis on CPU
        self.register_buffer(
            'spectral_basis',
            self._generate_spectral_basis(config.out_channels),
            persistent=False
        )
        
        self._init_weights()
        
    def _generate_spectral_basis(self, num_channels: int) -> torch.Tensor:
        """Generate smooth spectral basis functions."""
        x = torch.linspace(0, 1, num_channels)
        n_freq = (num_channels + 1) // 2
        
        sin_part = torch.stack([
            torch.sin(2 * math.pi * i * x) for i in range(n_freq)
        ])
        cos_part = torch.stack([
            torch.cos(2 * math.pi * i * x) for i in range(n_freq)
        ])
        
        basis = torch.cat([sin_part, cos_part], dim=0)[:num_channels]
        return basis
    
    def _get_spectral_basis(self, device, dtype):
        """v3.2.2: Get cached spectral basis for specific device/dtype."""
        cache_key = (str(device), str(dtype))
        
        if cache_key not in self._spectral_basis_cache:
            self._spectral_basis_cache[cache_key] = self.spectral_basis.to(device=device, dtype=dtype)
        
        return self._spectral_basis_cache[cache_key]
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def clear_caches(self):
        """v3.2.2: Clear all caches (spectral basis, RMSNorm weights)."""
        self._spectral_basis_cache.clear()
        for m in self.modules():
            if hasattr(m, 'clear_eval_caches'):
                m.clear_eval_caches()
                
    def forward_features(self, x):
        """Extract hierarchical features."""
        x = self.stem(x)
        
        features = []
        for i, stage in enumerate(self.stages):
            for block in stage:
                x = block(x)
            features.append(x)
            if i < len(self.stages) - 1:
                x = self.downsample[i](x)
            
        return features
        
    def forward(self, x):
        features = self.forward_features(x)
        
        x = features[-1]
        for i, (up, fuse, skip) in enumerate(zip(self.upsample, 
                                                  self.fusion, 
                                                  reversed(features[:-1]))):
            if HAS_DYNAMO:
                with torch._dynamo.disable():
                    x = up(x)
            else:
                x = up(x)
            x = fuse(x, skip)
        
        out = self.head(x)
        out, gate = out.chunk(2, dim=1)
        out = out * torch.sigmoid(gate)
        
        return torch.tanh(out)
    
    def compute_loss(self, pred, target, alpha: float = 0.1):
        """Memory-efficient loss with spectral regularization using cached basis."""
        rec_loss = F.l1_loss(pred, target)
        
        if alpha > 0:
            # v3.2.2: Use cached spectral basis
            basis = self._get_spectral_basis(pred.device, pred.dtype)
            
            # Memory-efficient computation using einsum
            proj = torch.einsum('bchw,dc->bdhw', pred, basis)
            smooth_loss = F.mse_loss(
                torch.einsum('bdhw,dc->bchw', proj, basis), 
                pred
            )
            
            return rec_loss + alpha * smooth_loss
        
        return rec_loss
    
    @property
    def num_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_mb = total * 4 / 1024 / 1024
        return {
            'total': total, 
            'trainable': trainable,
            'size_mb': total_mb,
            'size_description': 'float32-equivalent MB'
        }

# ============================================================================
# Model Factory with v3.2.2 enhancements
# ============================================================================

def create_sharp_v32(
    model_size: str = "base",
    in_channels: int = 3,
    out_channels: int = 31,
    compile_model: bool = True,
    compile_kwargs: Optional[dict] = None,
    verbose: bool = True,
    **kwargs
) -> SHARPv32:
    """Create SHARP v3.2.2 model with all audit fixes and optimizations."""
    
    configs = {
        "tiny": {
            "base_dim": 48,
            "depths": [2, 2, 2, 2],
            "heads": [3, 6, 12, 24],
            "mlp_ratios": [4., 4., 4., 4.]
        },
        "small": {
            "base_dim": 64,
            "depths": [2, 2, 4, 2],
            "heads": [4, 8, 16, 32],
            "mlp_ratios": [4., 4., 4., 4.]
        },
        "base": {
            "base_dim": 96,
            "depths": [2, 2, 6, 2],
            "heads": [6, 12, 24, 48],
            "mlp_ratios": [4., 4., 4., 4.]
        },
        "large": {
            "base_dim": 128,
            "depths": [2, 2, 8, 2],
            "heads": [8, 16, 32, 64],
            "mlp_ratios": [4., 4., 4., 4.]
        }
    }
    
    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}")
    
    model_config = configs[model_size]
    config = SHARPv32Config(
        in_channels=in_channels,
        out_channels=out_channels,
        **model_config,
        **kwargs
    )
    
    validate_model_config(config, f"SHARP-{model_size}")
    
    model = SHARPv32(config)
    
    # v3.2.2: Use logging instead of print
    if verbose:
        logger.info(f"Creating SHARP v3.2.2-{model_size}")
        logger.info(f"Sparse attention config:")
        logger.info(f"  block_size: {config.sparse_block_size}")
        logger.info(f"  q_block_size: {config.sparse_q_block_size}")
        logger.info(f"  max_tokens: {config.sparse_max_tokens}")
        logger.info(f"  window_size: {config.sparse_window_size}")
        logger.info(f"  k_cap: {config.sparse_k_cap}")
        logger.info(f"  sparsity_ratio: {config.sparse_sparsity_ratio}")
        logger.info(f"  rbf_centers_per_head: {config.rbf_centers_per_head}")
        logger.info(f"  key_rbf_mode: {config.key_rbf_mode}")
        logger.info(f"  ema_update_every: {config.ema_update_every}")
        
        sample_n = 512
        sample_k = max(1, int(sample_n * (1 - config.sparse_sparsity_ratio)))
        sample_k_capped = min(sample_k, config.sparse_k_cap) if config.sparse_k_cap else sample_k
        logger.info(f"  For N={sample_n}, sparsity={config.sparse_sparsity_ratio}: k_keep={sample_k_capped}")
    
    # Enhanced compilation with proper version and CUDA/ROCm checks
    if compile_model and HAS_COMPILE:
        can_compile = False
        compile_reason = ""
        
        if torch.cuda.is_available():
            if IS_ROCM:
                if HAS_TORCH_2_2 and HAS_ROCM_5_7_PLUS:
                    can_compile = True
                    compile_reason = f"ROCm {torch.version.hip} and PyTorch {torch.__version__}"
            else:
                if HAS_TORCH_2_2 and HAS_CUDA_11_8:
                    can_compile = True
                    compile_reason = f"CUDA {torch.version.cuda} and PyTorch {torch.__version__}"
        else:
            if HAS_TORCH_2_2:
                can_compile = True
                compile_reason = f"CPU with PyTorch {torch.__version__}"
        
        if can_compile:
            try:
                ck = {"mode": config.compile_mode or "reduce-overhead"}
                
                if HAS_TORCH_2_2:
                    ck.setdefault("dynamic", True)
                
                if compile_kwargs:
                    ck.update(compile_kwargs)
                
                if not config.use_checkpoint:
                    model = torch.compile(model, **ck)
                    if verbose:
                        logger.info(f"Model compiled ({compile_reason}) with: {ck}")
                elif verbose:
                    logger.info("Compilation skipped due to checkpoint usage")
            except Exception as e:
                if verbose:
                    logger.warning(f"Model compilation failed: {e}")
        else:
            if verbose:
                if torch.cuda.is_available() and not IS_ROCM and not HAS_CUDA_11_8:
                    logger.info(f"Compilation skipped: CUDA {torch.version.cuda} < 11.8")
                elif torch.cuda.is_available() and IS_ROCM and not HAS_ROCM_5_7_PLUS:
                    logger.info(f"Compilation skipped: ROCm {torch.version.hip} < 5.7")
                elif not HAS_TORCH_2_2:
                    logger.info(f"Compilation skipped: PyTorch {torch.__version__} < 2.2")
    
    model_ref = getattr(model, '_orig_mod', model)
    params = model_ref.num_parameters
    if verbose:
        logger.info(f"SHARP v3.2.2-{model_size}: {params['trainable']/1e6:.2f}M parameters ({params['size_mb']:.1f} MB)")
    
    return model

# ============================================================================
# Enhanced Trainer with v3.2.2 EMA throttling
# ============================================================================

class SHARPv32Trainer:
    """Optimized trainer for SHARP v3.2.2 with EMA update throttling."""
    def __init__(
        self,
        model: SHARPv32,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.05,
        warmup_ratio: float = 0.1,
        total_steps: int = 100000,
        gradient_clip: float = 1.0,
        ema_decay: float = 0.999,
        use_amp: bool = True,
        ema_update_every: Optional[int] = None,
    ):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Store original model reference for compiled models
        self._orig_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        
        # v3.2.2: Get EMA update frequency from config if not specified
        if ema_update_every is None:
            ema_update_every = getattr(self._orig_model.config, 'ema_update_every', 1)
        self.ema_update_every = ema_update_every
        self._step = 0
        
        self.optimizer = self._create_optimizer(learning_rate, weight_decay)
        
        warmup_steps = int(total_steps * warmup_ratio)
        self.scheduler = self._create_scheduler(warmup_steps, total_steps)
        
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        
        self.ema_decay = ema_decay
        self.ema_state = self._create_ema_state() if ema_decay > 0 else None
        
        self.gradient_clip = gradient_clip
        
    def _create_optimizer(self, lr: float, wd: float):
        """Module-aware weight decay grouping."""
        NORM_TYPES = (
            nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
            RMSNorm, ChannelRMSNorm
        )
        decay, no_decay = [], []
        
        for mod_name, mod in self.model.named_modules():
            for p_name, p in mod.named_parameters(recurse=False):
                if not p.requires_grad:
                    continue
                if isinstance(mod, NORM_TYPES) or p_name.endswith('bias') or p.dim() == 1:
                    no_decay.append(p)
                else:
                    decay.append(p)
        
        return torch.optim.AdamW(
            [{'params': decay, 'weight_decay': wd},
             {'params': no_decay, 'weight_decay': 0.0}],
            lr=lr, betas=(0.9, 0.999)
        )
        
    def _create_scheduler(self, warmup_steps: int, total_steps: int):
        def lr_lambda(step: int) -> float:
            step = min(step, total_steps - 1)
            
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
            
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
    def _create_ema_state(self):
        """Create EMA state on CPU to save VRAM."""
        state = {}
        
        # Parameters
        for k, p in self.model.named_parameters():
            if p.requires_grad:
                state[k] = p.detach().to('cpu', non_blocking=True).float()
        
        # Buffers
        for mod_name, mod in self.model.named_modules():
            non_persist = getattr(mod, "_non_persistent_buffers_set", set())
            for buf_name, b in mod.named_buffers(recurse=False):
                if b is None or buf_name.startswith('_') or 'num_batches_tracked' in buf_name:
                    continue
                if buf_name in non_persist:
                    continue
                full_name = ".".join([m for m in [mod_name, buf_name] if m])
                state[full_name] = b.detach().to('cpu', non_blocking=True)
                
        return state
        
    @torch.no_grad()
    def update_ema(self):
        """v3.2.2: Update EMA weights with throttling."""
        if self.ema_state is None:
            return
        
        # v3.2.2: Only update every N steps
        if self._step % self.ema_update_every != 0:
            return
            
        # Update EMA parameters
        for name, param in self.model.named_parameters():
            if name in self.ema_state:
                param_cpu = param.detach().to('cpu', non_blocking=True).float()
                self.ema_state[name].mul_(self.ema_decay).add_(
                    param_cpu, alpha=1 - self.ema_decay
                )
        
        # Update buffers
        for mod_name, mod in self.model.named_modules():
            non_persist = getattr(mod, "_non_persistent_buffers_set", set())
            for buf_name, buffer in mod.named_buffers(recurse=False):
                if buffer is None or buf_name in non_persist:
                    continue
                full_name = ".".join([m for m in [mod_name, buf_name] if m])
                if full_name in self.ema_state:
                    if buffer.dtype.is_floating_point:
                        buffer_cpu = buffer.detach().to(device='cpu', dtype=torch.float32, non_blocking=True)
                        self.ema_state[full_name] = self.ema_state[full_name].to(torch.float32)
                        self.ema_state[full_name].mul_(self.ema_decay).add_(
                            buffer_cpu, alpha=1 - self.ema_decay
                        )
                    else:
                        self.ema_state[full_name] = buffer.detach().to('cpu', non_blocking=True)
            
    def train_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        self._step += 1
        
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            outputs = self.model(inputs)
            loss = self.model.compute_loss(outputs, targets)
        
        self.scaler.scale(loss).backward()
        
        self.scaler.unscale_(self.optimizer)
        
        if self.gradient_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip
            )
        else:
            grad_norm = 0.0
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        
        self.scheduler.step()
        self.update_ema()  # v3.2.2: Throttled by ema_update_every
        
        return {
            'loss': loss.item(),
            'grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            'lr': self.scheduler.get_last_lr()[0],
            'step': self._step
        }
        
    @torch.no_grad()
    def evaluate(self, dataloader, psnr_max: float = 2.0) -> Dict[str, float]:
        """Evaluate model."""
        assert psnr_max > 0, "psnr_max must be positive"
        
        self.model.eval()
        
        total_loss = 0.0
        total_psnr = 0.0
        num_batches = 0
        
        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.model.compute_loss(outputs, targets)
            
            # PSNR calculation with configurable max value
            mse = F.mse_loss(outputs, targets)
            psnr = 10 * torch.log10((psnr_max ** 2) / mse.clamp(min=1e-8))
            
            total_loss += loss.item()
            total_psnr += psnr.item()
            num_batches += 1
        
        if num_batches == 0:
            return {'loss': 0.0, 'psnr': 0.0}
        
        return {
            'loss': total_loss / num_batches,
            'psnr': total_psnr / num_batches
        }
    
    def get_ema_model(self, device=None, eval_mode=True):
        """Get model with EMA weights on specified device."""
        if self.ema_state is None:
            return self.model
        
        if device is None:
            device = next(self.model.parameters()).device
        
        # Use stored original model reference
        ema_model = type(self._orig_model)(self._orig_model.config).to(device)
        
        # v3.2.2: Clear caches for fresh eval state
        ema_model.clear_caches()
        
        with torch.no_grad():
            # Load parameters
            for name, param in ema_model.named_parameters():
                if name in self.ema_state:
                    param.copy_(self.ema_state[name].to(device=device, dtype=param.dtype))
            
            # Load buffers
            for mod_name, mod in ema_model.named_modules():
                for buf_name, buffer in mod.named_buffers(recurse=False):
                    if buffer is not None:
                        full_name = ".".join([m for m in [mod_name, buf_name] if m])
                        if full_name in self.ema_state:
                            buffer.copy_(self.ema_state[full_name].to(device=device, dtype=buffer.dtype))
        
        if eval_mode:
            ema_model.eval()
                    
        return ema_model




