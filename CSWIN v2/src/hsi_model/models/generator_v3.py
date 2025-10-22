"""
Noise-Robust CSWin Generator for HSI Reconstruction v5

Final production generator with all architectural fixes:
- Adaptive GroupNorm (no channel mismatches) 
- NaN-safe attention blocks
- Configurable output activation
- torch.jit safe operations
- Configurable clamping with iteration-based disable

No changes from v4 - the generator architecture was already optimal.
The v5 fixes are all in the training script.

Key architectural features:
- U-Net structure with CSWin transformer blocks
- Dual attention (spectral + spatial)
- Noise-aware processing
- Dynamic up/downsampling for flexible resolutions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Any, Optional

from .attention import CSWinAttentionBlock, EfficientSpectralAttention


def adaptive_group_norm(channels: int, base_groups: int = 8) -> nn.GroupNorm:
    """Create GroupNorm with adaptive number of groups to avoid channel mismatches."""
    # Find the largest divisor of channels that is <= base_groups
    for groups in range(min(base_groups, channels), 0, -1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    # Fallback to 1 group (equivalent to LayerNorm)
    return nn.GroupNorm(1, channels)


class NaNSafeAttention(nn.Module):
    """Wrapper for attention modules with NaN protection."""
    def __init__(self, attention_module):
        super().__init__()
        self.attention = attention_module
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store input for potential recovery
        x_input = x.clone()
        
        # Forward through attention
        out = self.attention(x)
        
        # Check for NaN/Inf
        if torch.isnan(out).any() or torch.isinf(out).any():
            # Fallback to input (skip connection)
            return x_input
        
        return out


class DepthwiseConvBlock(nn.Module):
    """
    Efficient Depthwise Convolution block using separable convolutions.
    Now with adaptive GroupNorm.
    """
    def __init__(self, in_channels: int, out_channels: int, config: Dict[str, Any]) -> None:
        super(DepthwiseConvBlock, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # Use adaptive GroupNorm
        base_groups = config.get("norm_groups", 8)
        self.norm = adaptive_group_norm(out_channels, base_groups)
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class NoiseAwareBlock(nn.Module):
    """
    Block that adaptively processes features based on estimated noise levels.
    Fixed with proper channel handling.
    """
    def __init__(self, channels: int, config: Dict[str, Any]) -> None:
        super(NoiseAwareBlock, self).__init__()
        
        # Noise estimation produces 1 channel
        self.noise_est_conv = nn.Sequential(
            nn.Conv2d(channels, channels//4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels//4, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Feature processing
        self.features_conv = nn.Sequential(
            DepthwiseConvBlock(channels, channels, config),
            DepthwiseConvBlock(channels, channels, config)
        )
        
        # Gate takes channels + 1 (for noise map)
        self.gate = nn.Sequential(
            nn.Conv2d(channels + 1, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Estimate noise level
        noise_map = self.noise_est_conv(x)
        
        # Process features
        features = self.features_conv(x)
        
        # Apply adaptive gating based on noise
        gate_input = torch.cat([features, noise_map], dim=1)
        gate = self.gate(gate_input)
        
        # Apply gate and add residual connection
        return x + gate * features


class FeedForwardNetwork(nn.Module):
    """
    Feed-forward network with GELU activation and adaptive normalization.
    """
    def __init__(self, channels: int, expansion_factor: int = 4, config: Dict[str, Any] = None) -> None:
        super(FeedForwardNetwork, self).__init__()
        
        if config is None:
            raise ValueError("config cannot be None for FeedForwardNetwork")
            
        hidden_features = channels * expansion_factor
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden_features, kernel_size=1),
            nn.GELU(),
            DepthwiseConvBlock(hidden_features, hidden_features, config),
            nn.Conv2d(hidden_features, channels, kernel_size=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DualTransformerBlock(nn.Module):
    """
    Dual Transformer Block with NaN-safe attention.
    """
    def __init__(
        self, 
        channels: int, 
        split_size: int = 7, 
        num_heads: int = 4, 
        config: Dict[str, Any] = None
    ) -> None:
        super(DualTransformerBlock, self).__init__()
        
        if config is None:
            raise ValueError("config cannot be None for DualTransformerBlock")
            
        base_groups = config.get("norm_groups", 8)
        
        if num_heads is None:
            num_heads = config.get("num_heads", 4)
            
        # Use adaptive GroupNorm
        self.norm1 = adaptive_group_norm(channels, base_groups)
        self.norm2 = adaptive_group_norm(channels, base_groups)
        self.norm3 = adaptive_group_norm(channels, base_groups)
        
        # Wrap attention modules with NaN protection
        spectral_attn = EfficientSpectralAttention(channels, num_heads=num_heads)
        self.spectral_attn = NaNSafeAttention(spectral_attn)
        
        spatial_attn = CSWinAttentionBlock(channels, num_heads=num_heads, split_size=split_size)
        self.spatial_attn = NaNSafeAttention(spatial_attn)
        
        self.ffn = FeedForwardNetwork(channels, config=config)
        self.noise_block = NoiseAwareBlock(channels, config=config)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spectral attention branch
        x = x + self.spectral_attn(self.norm1(x))
        
        # Spatial attention branch with CSWin
        x = x + self.spatial_attn(self.norm2(x))
        
        # Feed-forward
        x = x + self.ffn(self.norm3(x))
        
        # Noise-aware processing
        x = self.noise_block(x)
        
        return x


class DynamicDownsampleBlock(nn.Module):
    """Downsampling block with adaptive normalization."""
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        scale_factor: int = 2, 
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        super(DynamicDownsampleBlock, self).__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Adaptive normalization
        base_groups = config.get("norm_groups", 8) if config else 8
        self.norm = adaptive_group_norm(out_channels, base_groups)
        self.act = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = F.interpolate(
            x, 
            size=(h//self.scale_factor, w//self.scale_factor), 
            mode='bilinear', 
            align_corners=False
        )
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class DynamicUpsampleBlock(nn.Module):
    """Upsampling block with adaptive normalization."""
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        scale_factor: int = 2, 
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        super(DynamicUpsampleBlock, self).__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Adaptive normalization
        base_groups = config.get("norm_groups", 8) if config else 8
        self.norm = adaptive_group_norm(out_channels, base_groups)
        self.act = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = F.interpolate(
            x, 
            size=(h*self.scale_factor, w*self.scale_factor), 
            mode='bilinear', 
            align_corners=False
        )
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class NoiseRobustCSWinGenerator(nn.Module):
    """
    Noise-Robust U-Net Generator with CSWin transformer blocks.
    
    v3.0 Key Fixes:
    - No sigmoid by default - better gradient flow
    - Adaptive GroupNorm throughout
    - NaN-safe attention blocks
    - Optional output activation modes
    
    Args:
        config: Model configuration
            - output_activation: 'none', 'sigmoid', 'tanh', 'delayed_sigmoid'
            - activation_delay_iters: iterations before activating output (for delayed_sigmoid)
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        super(NoiseRobustCSWinGenerator, self).__init__()
        
        # Extract parameters
        in_channels = config.get("in_channels", 3)
        out_channels = config.get("out_channels", 31)
        base_channels = config.get("base_channels", 64)
        split_sizes = config.get("split_sizes", [7, 7, 7])
        base_groups = config.get("norm_groups", 8)
        num_heads = config.get("num_heads", 4)
        
        # Output activation configuration
        self.output_activation = config.get("output_activation", "none")
        self.activation_delay_iters = config.get("activation_delay_iters", 20000)
        self.clamp_range = config.get("generator_clamp_range", 10.0)  # Configurable clamp
        self.clamp_after_iters = config.get("clamp_after_iters", 0)  # Can disable clamping after warmup
        self.register_buffer('iteration_count', torch.tensor(0))
        
        # Initial denoising
        self.denoising = nn.Sequential(
            nn.Conv2d(in_channels, base_channels//2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels//2, base_channels//2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels//2, in_channels, kernel_size=3, padding=1)
        )
        
        # Input embedding
        self.embedding = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            adaptive_group_norm(base_channels, base_groups),
            nn.GELU()
        )
        
        # Encoder blocks
        self.encoder1 = nn.Sequential(
            DualTransformerBlock(base_channels, split_size=split_sizes[0], num_heads=num_heads, config=config),
            DualTransformerBlock(base_channels, split_size=split_sizes[0], num_heads=num_heads, config=config)
        )
        self.down1 = DynamicDownsampleBlock(base_channels, base_channels*2, config=config)
        
        self.encoder2 = nn.Sequential(
            DualTransformerBlock(base_channels*2, split_size=split_sizes[1], num_heads=num_heads, config=config),
            DualTransformerBlock(base_channels*2, split_size=split_sizes[1], num_heads=num_heads, config=config)
        )
        self.down2 = DynamicDownsampleBlock(base_channels*2, base_channels*4, config=config)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            DualTransformerBlock(base_channels*4, split_size=split_sizes[2], num_heads=num_heads, config=config),
            DualTransformerBlock(base_channels*4, split_size=split_sizes[2], num_heads=num_heads, config=config)
        )
        
        # Decoder blocks
        self.up1 = DynamicUpsampleBlock(base_channels*4, base_channels*2, config=config)
        self.decoder1 = nn.Sequential(
            DualTransformerBlock(base_channels*4, split_size=split_sizes[1], num_heads=num_heads, config=config),
            DualTransformerBlock(base_channels*4, split_size=split_sizes[1], num_heads=num_heads, config=config)
        )
        self.compressor1 = nn.Conv2d(base_channels*4, base_channels*2, kernel_size=1)
        
        self.up2 = DynamicUpsampleBlock(base_channels*2, base_channels, config=config)
        self.compressor2 = nn.Conv2d(base_channels*2, base_channels, kernel_size=1)
        self.decoder2 = nn.Sequential(
            DualTransformerBlock(base_channels, split_size=split_sizes[0], num_heads=num_heads, config=config),
            DualTransformerBlock(base_channels, split_size=split_sizes[0], num_heads=num_heads, config=config)
        )
        
        # Output layer
        self.to_spectral = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Update iteration count during training (torch.jit safe)
        if self.training:
            self.iteration_count.add_(1)
        
        # Initial denoising with residual connection
        x_denoised = self.denoising(x)
        x = x + x_denoised
        
        # Embedding
        x = self.embedding(x)
        emb = x
        
        # Encoder
        e1 = self.encoder1(x)
        x = self.down1(e1)
        
        e2 = self.encoder2(x)
        x = self.down2(e2)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections
        x = self.up1(x)
        if x.shape[2:] != e2.shape[2:]:
            x = F.interpolate(x, size=e2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, e2], dim=1)
        x = self.decoder1(x)
        x = self.compressor1(x)
        
        x = self.up2(x)
        if x.shape[2:] != e1.shape[2:]:
            x = F.interpolate(x, size=e1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, e1], dim=1)
        x = self.compressor2(x)  # Apply before decoder2
        x = self.decoder2(x)
        
        # Handle dynamic spatial dimensions for residual connection
        if x.shape[2:] != emb.shape[2:]:
            x = F.interpolate(x, size=emb.shape[2:], mode='bilinear', align_corners=False)
        
        # Output with residual connection
        x = self.to_spectral(x + emb)
        
        # Apply output activation based on configuration
        if self.output_activation == "sigmoid":
            x = torch.sigmoid(x)
        elif self.output_activation == "tanh":
            x = 0.5 * (torch.tanh(x) + 1.0)  # Map to [0, 1]
        elif self.output_activation == "delayed_sigmoid":
            # Only apply sigmoid after specified iterations
            if self.iteration_count > self.activation_delay_iters:
                x = torch.sigmoid(x)
        # else: no activation (linear output)
        
        # Optional: Soft clipping to prevent extreme values
        # FIX: Actually use the config values instead of hardcoded ±10!
        if self.training and self.output_activation == "none":
            if self.clamp_after_iters == 0 or self.iteration_count < self.clamp_after_iters:
                x = torch.clamp(x, -self.clamp_range, self.clamp_range)
        
        return x
