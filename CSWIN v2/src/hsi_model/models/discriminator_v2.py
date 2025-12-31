# src/hsi_model/models/sn_transformer_discriminator_v5.py
"""
Spectral Normalized Transformer Discriminator v5 - Final Production

Features:
- Spectral normalization on all layers for stability
- NaN-protected attention with clamped logits  
- Adaptive normalization (no channel mismatches)
- Better initialization (Xavier with reduced gain)
- Residual scaling for gradient stability
- Compatible with Sinkhorn loss

No changes from v4 - the discriminator was already optimal.
The v5 fixes (R1 on HSI, update timing) are in the training script.

Architecture:
- Input: RGB (3) + HSI (31) = 34 channels
- Progressive downsampling with transformer blocks
- No global pooling - outputs spatial feature maps
- Suitable for both standard and Wasserstein losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Mapping, Callable, Union, overload, Literal
import math
from einops import rearrange
import logging

logger = logging.getLogger(__name__)

ConfigDict = Mapping[str, object]


def spectral_norm(module: nn.Module, n_power_iterations: int = 1, eps: float = 1e-6) -> nn.Module:
    """Apply spectral normalization to a module."""
    return nn.utils.spectral_norm(module, n_power_iterations=n_power_iterations, eps=eps)


def _get_spectral_weight(module: nn.Module) -> torch.Tensor:
    return module.weight_orig if hasattr(module, "weight_orig") else module.weight


class SNConvBlock(nn.Module):
    """Spectral Normalized Convolution Block."""
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        use_bias: bool = True
    ):
        super().__init__()
        
        # Spectral normalized convolution
        self.conv = spectral_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=use_bias)
        )
        
        # GELU activation
        self.activation = nn.GELU()
        
        # Better initialization
        self._init_weights()
    
    def _init_weights(self):
        # Xavier initialization (better for GELU)
        nn.init.xavier_uniform_(_get_spectral_weight(self.conv), gain=math.sqrt(2))
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(x))


class SpectralSelfAttention(nn.Module):
    """
    Spectral Self-Attention Block with NaN protection.
    """
    def __init__(
        self, 
        dim: int, 
        num_heads: int = 8,
        bias: bool = False,
        dropout: float = 0.0,
        attention_clamp: float = 50.0
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.attention_clamp = attention_clamp
        
        # Spectral normalized projections
        self.qkv = spectral_norm(nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias))
        self.qkv_dwconv = spectral_norm(nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias))
        self.project_out = spectral_norm(nn.Conv2d(dim, dim, kernel_size=1, bias=bias))
        
        self.attn_drop = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        # Smaller initialization for attention layers
        nn.init.xavier_uniform_(_get_spectral_weight(self.qkv), gain=1.0)
        nn.init.xavier_uniform_(_get_spectral_weight(self.qkv_dwconv), gain=1.0)
        nn.init.xavier_uniform_(_get_spectral_weight(self.project_out), gain=1.0)
        if self.qkv.bias is not None:
            nn.init.constant_(self.qkv.bias, 0)
            nn.init.constant_(self.qkv_dwconv.bias, 0)
            nn.init.constant_(self.project_out.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        
        # L2 normalize Q and K for stability
        q = F.normalize(q, dim=-1, eps=1e-12)
        k = F.normalize(k, dim=-1, eps=1e-12)
        
        # Compute attention with clamping
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        
        # Clamp attention scores to prevent overflow
        attn = torch.clamp(attn, -self.attention_clamp, self.attention_clamp)
        
        # Apply softmax
        attn = attn.softmax(dim=-1)
        
        # Check for NaN in attention
        if torch.isnan(attn).any():
            logger.warning("NaN detected in attention weights, using identity mapping")
            return x
        
        attn = self.attn_drop(attn)
        
        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        
        # Final NaN check
        if torch.isnan(out).any():
            logger.warning("NaN detected in attention output, using identity mapping")
            return x
        
        return out


class SNTransformerBlock(nn.Module):
    """Transformer block with spectral normalization and NaN protection."""
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        norm_type: str = 'layernorm'
    ):
        super().__init__()
        
        # Adaptive normalization
        if norm_type == 'layernorm':
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
        else:
            # Adaptive group norm
            groups = min(8, dim)
            while dim % groups != 0 and groups > 1:
                groups -= 1
            self.norm1 = nn.GroupNorm(groups, dim)
            self.norm2 = nn.GroupNorm(groups, dim)
        
        # Self-attention with NaN protection
        self.attn = SpectralSelfAttention(dim, num_heads, dropout=dropout)
        
        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            spectral_norm(nn.Conv2d(dim, mlp_hidden_dim, 1)),
            nn.GELU(),
            nn.Dropout(dropout),
            spectral_norm(nn.Conv2d(mlp_hidden_dim, dim, 1)),
            nn.Dropout(dropout)
        )
        
        # Apply LayerNorm in spatial domain if needed
        self.use_spatial_norm = norm_type == 'layernorm'
        
        # Residual scaling for stability
        self.residual_scale = 0.1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        
        # Store input for NaN recovery
        x_input = x.clone()
        
        if self.use_spatial_norm:
            # Apply LayerNorm in spatial domain
            B, C, H, W = x.shape
            x_norm = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
            x_norm = self.norm1(x_norm)
            x_norm = x_norm.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        else:
            x_norm = self.norm1(x)
        
        # Self-attention with residual
        attn_out = self.attn(x_norm)
        x = x + self.residual_scale * attn_out
        
        # Check for NaN after attention
        if torch.isnan(x).any():
            logger.warning("NaN after attention block, reverting to input")
            x = x_input
        
        if self.use_spatial_norm:
            # Apply LayerNorm in spatial domain
            x_norm = x.permute(0, 2, 3, 1).contiguous()
            x_norm = self.norm2(x_norm)
            x_norm = x_norm.permute(0, 3, 1, 2).contiguous()
        else:
            x_norm = self.norm2(x)
        
        # MLP with residual
        mlp_out = self.mlp(x_norm)
        x = x + self.residual_scale * mlp_out
        
        # Final NaN check
        if torch.isnan(x).any():
            logger.warning("NaN after transformer block, reverting to input")
            x = x_input
        
        return x


class SNTransformerDiscriminator(nn.Module):
    """
    Spectral Normalized Transformer Discriminator v2.
    
    Enhanced with:
    - NaN protection in attention layers
    - Better initialization
    - Adaptive normalization
    - Gradient stability improvements
    """
    def __init__(self, config: ConfigDict):
        super().__init__()
        
        # Configuration
        in_channels = 3 + 31  # RGB + HSI
        base_dim = config.get('discriminator_base_dim', 64)
        num_heads = config.get('discriminator_num_heads', 8)
        num_blocks = config.get('discriminator_num_blocks', [2, 2, 2])  # Per resolution
        mlp_ratio = config.get('discriminator_mlp_ratio', 2.0)
        dropout = config.get('discriminator_dropout', 0.0)
        
        # Initial projection with careful initialization
        self.input_proj = SNConvBlock(in_channels, base_dim, kernel_size=7, stride=1, padding=3)
        
        # Encoder stages
        self.encoder_stages = nn.ModuleList()
        dims = [base_dim, base_dim * 2, base_dim * 4]
        
        for i, (in_dim, out_dim, n_blocks) in enumerate(zip([base_dim] + dims[:-1], dims, num_blocks)):
            stage = nn.ModuleDict({
                'downsample': SNConvBlock(in_dim, out_dim, kernel_size=3, stride=2, padding=1) if i > 0 else nn.Identity(),
                'blocks': nn.ModuleList([
                    SNTransformerBlock(out_dim, num_heads, mlp_ratio, dropout)
                    for _ in range(n_blocks)
                ])
            })
            self.encoder_stages.append(stage)
        
        # Output projection - no global pooling!
        self.output_proj = nn.Sequential(
            spectral_norm(nn.Conv2d(dims[-1], dims[-1] // 2, kernel_size=3, padding=1)),
            nn.GELU(),
            spectral_norm(nn.Conv2d(dims[-1] // 2, 1, kernel_size=1))
        )
        
        # Initialize output layer with smaller weights
        nn.init.xavier_uniform_(_get_spectral_weight(self.output_proj[0]), gain=0.1)
        nn.init.xavier_uniform_(_get_spectral_weight(self.output_proj[2]), gain=0.1)
        
        logger.info(f"Initialized SNTransformerDiscriminator v2 with {sum(num_blocks)} transformer blocks")
    
    def forward(self, rgb: torch.Tensor, hsi: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through discriminator.
        
        Args:
            rgb: RGB input (B, 3, H, W)
            hsi: HSI input (B, 31, H, W) - real or generated
            
        Returns:
            Feature maps (B, 1, H', W') for loss computation
        """
        # Concatenate inputs
        x = torch.cat([rgb, hsi], dim=1)  # (B, 34, H, W)
        
        # Check input validity
        if torch.isnan(x).any():
            logger.error("NaN in discriminator input!")
            # Return zeros to avoid propagating NaN
            B = x.shape[0]
            return torch.zeros(B, 1, 8, 8, device=x.device, dtype=x.dtype)
        
        # Initial projection
        x = self.input_proj(x)  # (B, base_dim, H, W)
        
        # Encoder stages
        for stage in self.encoder_stages:
            # Downsample
            x = stage['downsample'](x)
            
            # Transformer blocks
            for block in stage['blocks']:
                x_before = x.clone()
                x = block(x)
                
                # Emergency NaN handling
                if torch.isnan(x).any():
                    logger.warning("NaN in discriminator encoder, using pre-block features")
                    x = x_before
        
        # Output projection
        output = self.output_proj(x)  # (B, 1, H', W')

        # Validate output spatial dimensions
        if output.shape[2] < 1 or output.shape[3] < 1:
            raise ValueError(
                f"Discriminator output too small: {output.shape}. "
                f"Increase input size or reduce downsampling. "
                f"Input shape was: rgb={rgb.shape}, hsi={hsi.shape}"
            )

        # Final clamp to prevent extreme values
        output = torch.clamp(output, -10, 10)
        
        return output


def compute_gradient_penalty(
    discriminator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    real_rgb: torch.Tensor,
    real_hsi: torch.Tensor,
    fake_hsi: torch.Tensor,
    lambda_gp: float = 10.0
) -> torch.Tensor:
    """
    Compute gradient penalty for WGAN-GP.
    
    Args:
        discriminator: The discriminator network
        real_rgb: Real RGB images
        real_hsi: Real HSI images
        fake_hsi: Generated HSI images
        lambda_gp: Gradient penalty coefficient
        
    Returns:
        Gradient penalty loss
    """
    batch_size = real_rgb.shape[0]
    device = real_rgb.device
    
    # Random interpolation coefficient
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    
    # Interpolate between real and fake
    interpolated_hsi = alpha * real_hsi + (1 - alpha) * fake_hsi
    interpolated_hsi.requires_grad_(True)
    
    # Get discriminator output for interpolated data
    disc_interpolated = discriminator(real_rgb, interpolated_hsi)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=disc_interpolated,
        inputs=interpolated_hsi,
        grad_outputs=torch.ones_like(disc_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Compute gradient penalty
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()
    
    return gradient_penalty


class DiscriminatorWithSinkhorn(nn.Module):
    """Wrapper that combines discriminator with Sinkhorn loss computation."""
    def __init__(self, config: ConfigDict):
        super().__init__()
        self.discriminator = SNTransformerDiscriminator(config)

    @overload
    def forward(
        self,
        rgb: torch.Tensor,
        hsi: torch.Tensor,
        return_features: Literal[True] = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    @overload
    def forward(
        self,
        rgb: torch.Tensor,
        hsi: torch.Tensor,
        return_features: Literal[False]
    ) -> torch.Tensor:
        ...

    def forward(
        self,
        rgb: torch.Tensor,
        hsi: torch.Tensor,
        return_features: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with optional feature extraction.

        Args:
            rgb: RGB input
            hsi: HSI input
            return_features: If True, return features for Sinkhorn loss

        Returns:
            If return_features: (disc_output, sinkhorn_features)
            Else: disc_output
        """
        disc_output = self.discriminator(rgb, hsi)

        if return_features:
            # Extract features for Sinkhorn loss
            B, C, H, W = disc_output.shape
            features = disc_output.view(B, -1)  # (B, H'*W')

            # L2 normalize for stability
            features = F.normalize(features, p=2, dim=1, eps=1e-12)

            return disc_output, features

        return disc_output


# Alias for backward compatibility and consistent naming
SpatialSpectralDiscriminator = SNTransformerDiscriminator


__all__ = [
    'SNTransformerDiscriminator',
    'SpatialSpectralDiscriminator',
    'DiscriminatorWithSinkhorn',
    'compute_gradient_penalty',
]
