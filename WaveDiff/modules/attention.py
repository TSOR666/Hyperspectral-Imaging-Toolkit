"""Spectral and spatial attention modules for HSI processing."""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Numerical stability constant used across attention modules
_EPS = 1e-6

class SpectralAttention(nn.Module):
    """
    Spectral attention module that focuses on different frequency components
    
    Applies attention mechanism across channels/bands to capture 
    spectral relationships.
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        
        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # MLP for channel attention
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Calculate spectral profile (channel-wise importance)
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        
        # Apply attention
        return x * y.expand_as(x)


class CrossSpectralAttention(nn.Module):
    """
    Cross-spectral attention module that models relationships between different frequency bands
    using a multi-head attention mechanism
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert self.head_dim * num_heads == channels, "channels must be divisible by num_heads"
        
        # Projection layers
        self.query = nn.Conv2d(channels, channels, kernel_size=1)
        self.key = nn.Conv2d(channels, channels, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        
        # Output projection
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)
        
        # Scale factor for attention
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with multi-head cross-spectral attention.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Attended tensor [B, C, H, W]
        """
        b, c, h, w = x.size()

        # Calculate query, key, value: [B, num_heads, head_dim, H*W]
        q = self.query(x).view(b, self.num_heads, self.head_dim, h * w)
        k = self.key(x).view(b, self.num_heads, self.head_dim, h * w)
        v = self.value(x).view(b, self.num_heads, self.head_dim, h * w)

        # Transpose for attention: [B, num_heads, H*W, head_dim]
        q = q.transpose(2, 3)
        k = k.transpose(2, 3)
        v = v.transpose(2, 3)

        # Calculate attention scores: [B, num_heads, H*W, H*W]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Stabilized softmax: subtract max for numerical stability (prevents overflow in float16)
        attn = attn - attn.max(dim=-1, keepdim=True).values
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values: [B, num_heads, H*W, head_dim]
        out = attn @ v
        out = out.transpose(2, 3).contiguous().view(b, c, h, w)

        return self.out_proj(out)


class SpectralSpatialAttention(nn.Module):
    """
    Combined spectral and spatial attention module
    
    Applies both channel attention and spatial attention sequentially
    to capture both spectral and spatial relationships.
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        
        # Spectral attention
        self.spectral_attn = SpectralAttention(channels, reduction)
        
        # Spatial attention
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Apply spectral attention
        x_spec = self.spectral_attn(x)
        
        # Calculate spatial attention
        avg_out = torch.mean(x_spec, dim=1, keepdim=True)
        max_out, _ = torch.max(x_spec, dim=1, keepdim=True)
        spatial_features = torch.cat([avg_out, max_out], dim=1)
        spatial_attn = self.spatial_conv(spatial_features)
        
        # Apply spatial attention
        return x_spec * spatial_attn


class WaveletAttention(nn.Module):
    """
    Attention module applied to wavelet coefficients
    
    Applies different attention weights to approximation and detail coefficients
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.channels = channels
        
        # Separate attention for approximation and detail coefficients
        self.ll_attention = SpectralAttention(channels, reduction)
        self.detail_attention = nn.ModuleList([
            SpectralAttention(channels, reduction) for _ in range(3)  # LH, HL, HH
        ])
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, C, 4, H, W] containing wavelet coefficients
        
        Returns:
            Tensor of shape [B, C, 4, H, W] with attention applied
        """
        B, C, _, H, W = x.shape
        
        # Extract wavelet components
        components = [x[:, :, i] for i in range(4)]  # LL, LH, HL, HH
        
        # Apply appropriate attention to each component
        components[0] = self.ll_attention(components[0])  # LL - approximation
        
        for i in range(3):
            components[i+1] = self.detail_attention[i](components[i+1])  # Detail coefficients
            
        # Recombine components
        return torch.stack(components, dim=2)


class MultiscaleSpectralAttention(nn.Module):
    """
    Multi-scale attention that operates at different resolutions

    Captures both local and global spectral-spatial relationships
    """
    def __init__(self, channels, reduction=8, scales=(1, 2, 4)):
        super().__init__()
        self.scales = scales

        # Attention at different scales
        self.attentions = nn.ModuleList([
            SpectralAttention(channels, reduction) for _ in scales
        ])

        # Fusion layer
        self.fusion = nn.Conv2d(channels * len(scales), channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape

        # Create multi-scale features
        multi_scale_features = []

        for i, scale in enumerate(self.scales):
            # Skip if scale is too large for input
            if h // scale < 1 or w // scale < 1:
                continue

            # Downsample
            if scale > 1:
                downsample = F.avg_pool2d(x, kernel_size=scale)
            else:
                downsample = x

            # Apply attention
            attn_feat = self.attentions[i](downsample)

            # Upsample back to original size if needed
            if scale > 1:
                attn_feat = F.interpolate(attn_feat, size=(h, w), mode='bilinear', align_corners=False)

            multi_scale_features.append(attn_feat)

        # Concatenate and fuse multi-scale features
        if len(multi_scale_features) > 1:
            fused = self.fusion(torch.cat(multi_scale_features, dim=1))
        else:
            fused = multi_scale_features[0]

        return fused


class MultiHeadSpectralAttention(nn.Module):
    """
    Enhanced multi-head spectral attention for robust feature extraction
    Helps model generalize by capturing diverse spectral patterns
    """
    def __init__(self, channels, num_heads=8, reduction=4, dropout=0.1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        # Multi-head projections
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)

        # Attention dropout for regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) * (self.head_dim ** -0.5))

        # Channel mixing MLP for better representation
        hidden_dim = channels // reduction
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, channels, kernel_size=1),
            nn.Dropout(dropout)
        )

        # Layer normalization
        self.norm1 = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
        self.norm2 = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)

    def forward(self, x):
        B, C, H, W = x.shape
        identity = x

        # Normalize input
        x = self.norm1(x)

        # Compute QKV
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        # Reshape for multi-head attention
        q = q.reshape(B, self.num_heads, self.head_dim, H * W)
        k = k.reshape(B, self.num_heads, self.head_dim, H * W)
        v = v.reshape(B, self.num_heads, self.head_dim, H * W)

        # Compute attention scores: [B, num_heads, H*W, H*W]
        attn = torch.matmul(q.transpose(-2, -1), k)

        # Clamp temperature to prevent extreme scaling (numerical stability)
        clamped_temp = torch.clamp(self.temperature, min=1e-4, max=10.0)
        attn = attn * clamped_temp

        # Stabilized softmax: subtract max for float16 safety
        attn = attn - attn.max(dim=-1, keepdim=True).values
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # Apply attention to values
        out = torch.matmul(v, attn.transpose(-2, -1))  # B, num_heads, head_dim, H*W
        out = out.reshape(B, C, H, W)

        # Output projection
        out = self.out_proj(out)
        out = self.proj_dropout(out)

        # First residual connection
        x = identity + out

        # Channel mixing with second residual connection
        out = self.norm2(x)
        out = self.channel_mlp(out)
        x = x + out

        return x


class DomainAdaptiveAttention(nn.Module):
    """
    Domain-adaptive attention mechanism for cross-dataset generalization
    Learns to adapt attention patterns based on input domain characteristics
    """
    def __init__(self, channels, num_domains=4, reduction=8):
        super().__init__()
        self.channels = channels
        self.num_domains = num_domains

        # Domain classifier (predicts domain from features)
        self.domain_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, num_domains)
        )

        # Domain-specific attention modules
        self.domain_attentions = nn.ModuleList([
            SpectralSpatialAttention(channels, reduction) for _ in range(num_domains)
        ])

    def forward(self, x, get_domain_weights=False):
        """
        Args:
            x: Input tensor [B, C, H, W]
            get_domain_weights: Whether to return domain weights for analysis

        Returns:
            Attended features (and optionally domain weights)
        """
        # Predict domain weights
        domain_logits = self.domain_encoder(x)  # B, num_domains
        domain_weights = F.softmax(domain_logits, dim=1)  # B, num_domains

        # Apply domain-specific attention with weighted combination
        outputs = []
        for i, attn_module in enumerate(self.domain_attentions):
            domain_out = attn_module(x)  # B, C, H, W
            # Weight by domain probability
            weight = domain_weights[:, i:i+1, None, None]  # B, 1, 1, 1
            outputs.append(domain_out * weight)

        # Combine domain-specific outputs
        attended = torch.stack(outputs, dim=0).sum(dim=0)  # B, C, H, W

        if get_domain_weights:
            return attended, domain_weights
        return attended