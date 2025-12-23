"""
Memory-Efficient Attention Modules for HSI Reconstruction

CRITICAL MEMORY OPTIMIZATIONS (v3.0):
==========================================
1. CSWinAttentionBlock Bias Fix (SAVES ~50GB):
   - Bias mask scales with long-axis length (W/H), not s*W/H
   - Before: bias scaled with s*W/H, inflating attention masks
   - After: bias scales with W/H, reducing mask size by ~s^2
   - Memory reduction: ~s^2 on bias masks (attention path dependent)

2. EfficientSpectralAttention:
   - Uses global channel attention applied to spatial features
   - Avoids memory-intensive (H*W) × (H*W) attention matrix
   - Memory: O(B*C^2/num_heads + B*C*H*W)

3. PyTorch Version Compatibility (v3.0):
   - Added version check for use_reentrant parameter
   - Works with PyTorch 1.x and 2.x

4. Bias Table Precision (v3.2):
   - Default to FP32 for relative position bias (FP16 optional via config)
   - FP16 saves ~50% memory on bias tables when enabled

v3.1 NUMERICAL STABILITY FIXES:
==========================================
- Added eps to F.normalize to prevent NaN on zero-norm vectors
- Added softmax stabilization (subtract max before exp) for non-SDPA path
- Cast attention computation to fp32 for mixed precision safety
- Removed inplace=True from ReLU to prevent autograd issues with checkpointing

Additional Memory Tips:
- Set PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
- Use torch.cuda.amp.autocast(dtype=torch.bfloat16) on Ampere/Hopper GPUs
- Enable use_fp16_bias=True in config to store bias tables in FP16
- Monitor memory with torch.cuda.memory_summary() during training

Version History:
- v1.0: Initial implementation with memory-hungry global attention
- v2.0: Fixed critical bias tiling bug that caused 76GB OOM on A100
- v3.0: Added PyTorch compatibility and FP16 bias tables
- v3.1: Added numerical stability fixes (eps, softmax stabilization, fp32 casting)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Optional, Mapping
from einops import rearrange, repeat
from torch.utils.checkpoint import checkpoint
from packaging import version

logger = logging.getLogger(__name__)

ConfigDict = Mapping[str, object]

# Numerical stability epsilon - defined once and reused
EPS = 1e-8


class LePEAttention(nn.Module):
    """
    Locally enhanced positional encoding for attention.
    
    Applies a depth-wise convolution to capture local spatial information
    as a form of relative positional encoding.
    
    Args:
        channels: Number of input/output channels
    """
    def __init__(self, channels: int) -> None:
        super(LePEAttention, self).__init__()
        self.pe_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.GELU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Position-enhanced tensor of shape (B, C, H, W)
        """
        return x + self.pe_conv(x)


class CSWinAttentionBlock(nn.Module):
    """
    Cross-Shaped Window Attention module with FIXED memory-efficient bias computation.
    
    This module performs attention along horizontal and vertical stripes separately,
    then combines the results. Critical v2.0 fix: bias is now correctly tiled by
    the number of windows, not raw pixel dimensions.
    
    Memory Impact of Fix:
    - Before: bias matched s*W/H (e.g., 3584x3584 for 512px with s=7)
    - After: bias matches long-axis length (e.g., 512x512 for 512px)
    - Savings scale with s^2 on the bias mask size
    
    Args:
        dim: Number of input/output channels
        num_heads: Number of attention heads
        split_size: Size of window for splitting input
        qkv_bias: Whether to include bias in QKV projections
        config: Model configuration dict
    """
    def __init__(
        self, 
        dim: int, 
        num_heads: int = 8, 
        split_size: int = 7, 
        qkv_bias: bool = True, 
        config: Optional[ConfigDict] = None
    ) -> None:
        super(CSWinAttentionBlock, self).__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.dim = dim
        self.num_heads = num_heads
        self.split_size = split_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # QKV projections
        self.qkv_h = nn.Conv2d(dim, dim*3, kernel_size=1, bias=qkv_bias)
        self.qkv_v = nn.Conv2d(dim, dim*3, kernel_size=1, bias=qkv_bias)
        
        # Output projection
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        
        # Locally enhanced positional encoding
        self.lepe_h = LePEAttention(dim)
        self.lepe_v = LePEAttention(dim)

        torch_version = version.parse(torch.__version__.split('+')[0]) if hasattr(torch, "__version__") else version.parse("0")
        self._supports_non_reentrant_ckpt = torch_version >= version.parse("2.1")
        
        # Default to FP32 bias tables for numerical stability; allow FP16 opt-in.
        # Config overrides, otherwise fall back to environment variable for compatibility.
        use_fp16_bias = False
        if config is not None:
            use_fp16_bias = config.get('use_fp16_bias', False)
        elif os.environ.get('MST_USE_FP32_BIAS', '').lower() == 'true':
            use_fp16_bias = False

        if use_fp16_bias:
            logger.warning("use_fp16_bias=True may reduce attention precision; enable only if memory constrained.")
        
        bias_dtype = torch.float16 if use_fp16_bias else torch.float32
        
        table_shape = (2 * split_size - 1, 2 * split_size - 1, num_heads)
        self.relative_position_bias_table_h = nn.Parameter(
            torch.zeros(table_shape, dtype=bias_dtype))
        self.relative_position_bias_table_v = nn.Parameter(
            torch.zeros(table_shape, dtype=bias_dtype))

        nn.init.trunc_normal_(self.relative_position_bias_table_h, std=0.02)
        nn.init.trunc_normal_(self.relative_position_bias_table_v, std=0.02)

        idx = torch.arange(split_size, dtype=torch.long)
        relative_index = idx[:, None] - idx[None, :] + split_size - 1  # (s, s)
        self.register_buffer(
            "_relative_position_index", relative_index, persistent=False
        )
        self.register_buffer(
            "_relative_center_index",
            torch.tensor(split_size - 1, dtype=torch.long),
            persistent=False,
        )

        logger.debug(f"CSWinAttentionBlock initialized with bias dtype: {bias_dtype}")
        
    def _expand_bias(self, bias_ss_head: torch.Tensor, tiles_long: int) -> torch.Tensor:
        """
        Inflate a (s, s, H) bias that covers one split-size window into a full
        (H, s·tiles_long, s·tiles_long) bias.
        
        Args:
            bias_ss_head: Small bias tensor of shape (s, s, H)
            tiles_long: Number of windows/repetitions to use when expanding the bias
                in both row and column dimensions
                
        Returns:
            Expanded bias tensor of shape (H, s*tiles_long, s*tiles_long)
        """
        if tiles_long <= 0:
            raise ValueError("tiles_long must be a positive integer")
        H = bias_ss_head.shape[-1]
        s = self.split_size
        bias = bias_ss_head.permute(2, 0, 1)  # (H, s, s)
        bias = (bias
                .unsqueeze(2).unsqueeze(4)  # (H, s, 1, s, 1)
                .expand(-1, s, tiles_long, s, tiles_long)
                .reshape(H, s * tiles_long, s * tiles_long))
        return bias
        
    def _compute_horizontal_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute attention along horizontal stripes with FIXED position bias.
        
        Critical v2.0 fix: Now correctly computes window count for bias expansion.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Attention-processed tensor of same shape
        """
        B, C, padded_H, padded_W = x.shape
        if padded_H % self.split_size != 0 or padded_W % self.split_size != 0:
            raise ValueError("Input must be divisible by split_size after padding")
        h_windows = padded_H // self.split_size
        head_dim = self.head_dim
        
        # Process QKV with convolution
        qkv_h = self.qkv_h(x)  # Shape: [B, 3*C, padded_H, padded_W]
        
        # Reshape to separate QKV, heads, and windows
        qkv_h = rearrange(
            qkv_h, 
            'b (three h d) (hw s) w -> three b hw s w h d', 
            three=3, 
            h=self.num_heads, 
            d=head_dim, 
            hw=h_windows, 
            s=self.split_size
        )
        
        q_h, k_h, v_h = qkv_h[0], qkv_h[1], qkv_h[2]  # Each shape: [B, h_windows, split_size, W, num_heads, head_dim]
        
        # Reshape for attention computation (treat split rows as batch for width attention)
        q_h = rearrange(q_h, 'b hw s w h d -> (b hw s) h w d')
        k_h = rearrange(k_h, 'b hw s w h d -> (b hw s) h w d')
        v_h = rearrange(v_h, 'b hw s w h d -> (b hw s) h w d')

        use_sdpa = hasattr(F, "scaled_dot_product_attention")

        rel_cols = self._relative_position_index
        bias_ss = self.relative_position_bias_table_h[
            self._relative_center_index, rel_cols, :
        ]
        w_windows = padded_W // self.split_size
        bias = self._expand_bias(bias_ss, tiles_long=w_windows)

        if use_sdpa:
            # SDPA handles numerical stability internally
            attn_mask = bias.unsqueeze(0).to(q_h.dtype)  # (1, num_heads, seq, seq)
            out_h = F.scaled_dot_product_attention(
                q_h, k_h, v_h, attn_mask=attn_mask, scale=self.scale
            )
        else:
            # Manual attention with numerical stability (v3.1 fix)
            # (B*hw*s, H, seq, d) @ (B*hw*s, H, d, seq) -> (B*hw*s, H, seq, seq)
            attn_h = (q_h @ k_h.transpose(-2, -1)) * self.scale
            attn_h = attn_h + bias.unsqueeze(0).to(attn_h.dtype)
            # Cast to fp32 and stabilize softmax to prevent overflow
            attn_h = attn_h.float()
            attn_h = attn_h - attn_h.amax(dim=-1, keepdim=True)
            attn_h = F.softmax(attn_h, dim=-1)
            attn_h = attn_h.to(q_h.dtype)  # Cast back to compute dtype
            out_h = attn_h @ v_h
        
        # Reshape back to original format
        out_h = rearrange(
            out_h, 
            '(b hw s) h w d -> b (h d) (hw s) w', 
            b=B, 
            hw=h_windows, 
            s=self.split_size, 
            h=self.num_heads,
            w=padded_W
        )
        
        # Add locally enhanced positional encoding
        out_h = self.lepe_h(out_h)
        
        return out_h
        
    def _compute_vertical_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute attention along vertical stripes with FIXED position bias.
        
        Critical v2.0 fix: Now correctly computes window count for bias expansion.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Attention-processed tensor of same shape
        """
        B, C, padded_H, padded_W = x.shape
        if padded_H % self.split_size != 0 or padded_W % self.split_size != 0:
            raise ValueError("Input must be divisible by split_size after padding")
        w_windows = padded_W // self.split_size
        head_dim = self.head_dim
        
        # Process vertical stripes 
        # Reshape input for vertical attention
        x_v = rearrange(
            x, 
            'b c h (ww s) -> (b ww) c h s', 
            ww=w_windows, 
            s=self.split_size
        )  # [B*w_windows, C, padded_H, split_size]
        
        # Apply QKV projection
        qkv_v = self.qkv_v(x_v)  # [B*w_windows, 3*C, padded_H, split_size]
        
        # Reshape to separate QKV, heads, and dimensions
        qkv_v = rearrange(
            qkv_v, 
            '(b ww) (three h d) ph s -> three (b ww) ph s h d', 
            b=B, 
            ww=w_windows, 
            three=3, 
            h=self.num_heads, 
            d=head_dim, 
            ph=padded_H
        )
        
        q_v, k_v, v_v = qkv_v[0], qkv_v[1], qkv_v[2]  # Each shape: [B*w_windows, padded_H, split_size, num_heads, head_dim]
        
        # Reshape for attention computation (treat split columns as batch for height attention)
        q_v = rearrange(q_v, 'bw ph s h d -> (bw s) h ph d')
        k_v = rearrange(k_v, 'bw ph s h d -> (bw s) h ph d')
        v_v = rearrange(v_v, 'bw ph s h d -> (bw s) h ph d')

        use_sdpa = hasattr(F, "scaled_dot_product_attention")

        rel_rows = self._relative_position_index
        bias_ss = self.relative_position_bias_table_v[
            rel_rows, self._relative_center_index, :
        ]
        h_windows = padded_H // self.split_size
        bias = self._expand_bias(bias_ss, tiles_long=h_windows)

        if use_sdpa:
            # SDPA handles numerical stability internally
            attn_mask = bias.unsqueeze(0).to(q_v.dtype)
            out_v = F.scaled_dot_product_attention(
                q_v, k_v, v_v, attn_mask=attn_mask, scale=self.scale
            )
        else:
            # Manual attention with numerical stability (v3.1 fix)
            # (B*ww*s, H, seq, d) @ (B*ww*s, H, d, seq) -> (B*ww*s, H, seq, seq)
            attn_v = (q_v @ k_v.transpose(-2, -1)) * self.scale
            attn_v = attn_v + bias.unsqueeze(0).to(attn_v.dtype)
            # Cast to fp32 and stabilize softmax to prevent overflow
            attn_v = attn_v.float()
            attn_v = attn_v - attn_v.amax(dim=-1, keepdim=True)
            attn_v = F.softmax(attn_v, dim=-1)
            attn_v = attn_v.to(q_v.dtype)  # Cast back to compute dtype
            out_v = attn_v @ v_v
        
        # Reshape back to original format
        out_v = rearrange(
            out_v, 
            '(b ww s) h ph d -> b (h d) ph (ww s)', 
            b=B, 
            ww=w_windows, 
            s=self.split_size, 
            ph=padded_H, 
            h=self.num_heads
        )
        
        # Add locally enhanced positional encoding
        out_v = self.lepe_v(out_v)
        
        return out_v
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform cross-shaped window attention.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Attention-processed tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Pad input if needed to make it divisible by split_size
        pad_h = (self.split_size - H % self.split_size) % self.split_size
        pad_w = (self.split_size - W % self.split_size) % self.split_size
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            padded_H, padded_W = H + pad_h, W + pad_w
        else:
            padded_H, padded_W = H, W
        
        # v3.0: Version-aware gradient checkpointing
        if self.training:
            if self._supports_non_reentrant_ckpt:
                out_h = checkpoint(self._compute_horizontal_attention, x, use_reentrant=False)
                out_v = checkpoint(self._compute_vertical_attention, x, use_reentrant=False)
            else:
                # Fallback for older PyTorch versions
                out_h = checkpoint(self._compute_horizontal_attention, x)
                out_v = checkpoint(self._compute_vertical_attention, x)
        else:
            out_h = self._compute_horizontal_attention(x)
            out_v = self._compute_vertical_attention(x)
        
        # Combine horizontal and vertical attention and apply projection
        out = self.proj((out_h + out_v) / 2)
        
        # Remove padding if necessary
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]
            
        return out


class EfficientSpectralAttention(nn.Module):
    """
    Spectral Attention module optimized for hyperspectral data with noise robustness.
    
    This version computes global channel attention and applies it to spatial features,
    avoiding the memory-intensive (H*W) x (H*W) attention matrix.
    
    Memory complexity: O(B*C^2/num_heads + B*C*H*W)
    
    Args:
        channels: Number of input/output channels
        num_heads: Number of attention heads
        config: Model configuration
    """
    def __init__(
        self, 
        channels: int, 
        num_heads: int = 4, 
        config: Optional[ConfigDict] = None
    ) -> None:
        super(EfficientSpectralAttention, self).__init__()
        self.channels = channels
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        # Handle None config
        norm_groups = config.get("norm_groups", 8) if config else 8
        
        # Linear projections for Q, K, V with group normalization for better stability
        self.to_q = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(norm_groups, channels)
        )
        self.to_k = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(norm_groups, channels)
        )
        self.to_v = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        
        # Output projection with group normalization
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GroupNorm(norm_groups, channels)
        )
        
        # Position embedding
        self.pos_embed = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute global channel attention and apply it to spatial features.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Attention-processed tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Compute Q, K, V
        q = self.to_q(x)  # (B, C, H, W)
        k = self.to_k(x)  # (B, C, H, W)
        v = self.to_v(x)  # (B, C, H, W)
        
        # Global channel descriptors
        q_global = q.mean(dim=(2, 3))  # (B, C)
        k_global = k.mean(dim=(2, 3))  # (B, C)
        
        # Reshape for multi-head attention
        q_global = rearrange(q_global, 'b (h d) -> b h d', h=self.num_heads)
        k_global = rearrange(k_global, 'b (h d) -> b h d', h=self.num_heads)
        
        # Normalize for cosine similarity (more robust to noise)
        q_norm = F.normalize(q_global, dim=-1, eps=EPS)
        k_norm = F.normalize(k_global, dim=-1, eps=EPS)
        if not torch.isfinite(q_norm).all() or not torch.isfinite(k_norm).all():
            logger.warning("Non-finite values in normalized Q/K; replacing with zeros")
            q_norm = torch.nan_to_num(q_norm)
            k_norm = torch.nan_to_num(k_norm)
        
        # Compute channel attention (B, H, d, d)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.einsum('b h i, b h j -> b h i j', q_norm, k_norm) * scale
        attn = attn - attn.amax(dim=-1, keepdim=True)  # Stabilize softmax
        attn_dtype = attn.dtype
        if attn_dtype in (torch.float16, torch.bfloat16):
            attn = F.softmax(attn.float(), dim=-1).to(attn_dtype)
        else:
            attn = F.softmax(attn, dim=-1)
        
        # Apply attention to spatial features
        v_reshaped = rearrange(v, 'b (h d) H W -> b h d (H W)', h=self.num_heads)
        if attn.dtype != v_reshaped.dtype:
            attn = attn.to(v_reshaped.dtype)
        out = torch.einsum('b h i j, b h j s -> b h i s', attn, v_reshaped)
        
        # Reshape back to image format
        out = rearrange(out, 'b h d (H W) -> b (h d) H W', H=H, W=W)
        
        # Apply position embedding and projection
        pos = self.pos_embed(x)
        out = out + pos
        out = self.proj(out)
        
        return out


# Optional: Alternative lightweight spectral attention for extreme memory constraints
class ChannelAttention(nn.Module):
    """
    Lightweight channel attention module (SE-style) for extreme memory constraints.
    Can be used as a drop-in replacement for EfficientSpectralAttention.
    
    Args:
        channels: Number of channels
        reduction: Channel reduction factor
        config: Model configuration (optional)
    """
    def __init__(
        self, 
        channels: int, 
        reduction: int = 4,
        config: Optional[ConfigDict] = None
    ) -> None:
        super(ChannelAttention, self).__init__()
        self.channels = channels
        
        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Channel attention with two FC layers
        # v3.1: Removed inplace=True to prevent autograd issues with checkpointing
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
        # Optional learnable temperature
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel-wise attention.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Channel-attention weighted tensor of shape (B, C, H, W)
        """
        B, C, _, _ = x.shape
        
        # Global average pooling
        y = self.avg_pool(x).view(B, C)  # (B, C)
        
        # Channel attention weights with temperature
        y = self.fc(y).view(B, C, 1, 1) * self.temperature
        
        # Apply attention
        return x * y.expand_as(x)
