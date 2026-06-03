"""
Noise-Robust CSWin Generator for HSI Reconstruction v5.1

Final production generator with all architectural fixes:
- Adaptive GroupNorm (no channel mismatches)
- NaN-safe attention blocks
- Configurable output activation
- torch.jit safe operations
- Configurable clamping with iteration-based disable

v5.1 Fixes:
- Added missing type hints
- Added torch.no_grad() around buffer updates for safety

Key architectural features:
- U-Net structure with CSWin transformer blocks
- Dual attention (spectral + spatial)
- Noise-aware processing
- Dynamic up/downsampling for flexible resolutions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union, Mapping

from .attention import CSWinAttentionBlock, EfficientSpectralAttention, SpectralMSA

ConfigDict = Mapping[str, object]


def adaptive_group_norm(channels: int, base_groups: int = 8) -> nn.GroupNorm:
    """Create GroupNorm with adaptive number of groups to avoid channel mismatches."""
    # Prefer group counts that divide both channels and base_groups for consistency.
    for groups in range(min(base_groups, channels), 0, -1):
        if channels % groups == 0 and base_groups % groups == 0:
            return nn.GroupNorm(groups, channels)
    # Fallback to 1 group (equivalent to LayerNorm)
    return nn.GroupNorm(1, channels)


class NaNSafeAttention(nn.Module):
    """Wrapper for attention modules with NaN protection.

    The `torch.isnan/isinf` check forces a GPU→CPU sync on every call.
    With 20+ instances per generator forward this dominated inference
    latency.  The check is now frequency-gated:
    - Training: check every `check_freq` calls (default 100).  NaN still
      raises immediately so instability is caught within one epoch.
    - Eval: skip entirely.  NaN in inference is detected downstream by
      the `isfinite` guards in the training/validation loop.
    """

    def __init__(self, attention_module: nn.Module, check_freq: int = 100) -> None:
        super().__init__()
        self.attention = attention_module
        self.nan_count = 0
        self._call_count = 0
        self._check_freq = check_freq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.attention(x)

        # Only pay the GPU→CPU sync cost periodically in training;
        # never during eval (sync-free inference path).
        self._call_count += 1
        if self.training and (self._call_count % self._check_freq == 0):
            if torch.isnan(out).any() or torch.isinf(out).any():
                self.nan_count += 1
                raise RuntimeError(
                    f"NaN/Inf in attention (occurrence {self.nan_count}). "
                    f"Input range: [{x.min():.4f}, {x.max():.4f}]"
                )

        return out


class DepthwiseConvBlock(nn.Module):
    """
    Efficient Depthwise Convolution block using separable convolutions.
    Now with adaptive GroupNorm.
    """
    def __init__(self, in_channels: int, out_channels: int, config: ConfigDict) -> None:
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
    def __init__(self, channels: int, config: ConfigDict) -> None:
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
    def __init__(self, channels: int, expansion_factor: int = 4, config: Optional[ConfigDict] = None) -> None:
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


class ChannelLayerNorm(nn.Module):
    """LayerNorm over the channel dimension for (B, C, H, W) tensors
    (Restormer-style pre-norm used inside the SSTB)."""
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)          # (B, H, W, C)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


class CBAMChannelGate(nn.Module):
    """CBAM-style channel attention gate (HSIFormer Eq. 3): concat of avg- and
    max-pooled channel descriptors -> shared 1x1 MLP -> sigmoid -> scales X."""
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(2 * channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=(2, 3), keepdim=True)
        mx = x.amax(dim=(2, 3), keepdim=True)
        gate = torch.sigmoid(self.mlp(torch.cat([avg, mx], dim=1)))
        return x * gate


class GDFN(nn.Module):
    """Gated-Dconv feed-forward network (Restormer / HSIFormer GDFN, Fig 3c):
    1x1 expand -> 3x3 depthwise -> GELU(half) * (other half) -> 1x1 project."""
    def __init__(self, channels: int, expansion: float = 2.66) -> None:
        super().__init__()
        hidden = max(1, int(channels * expansion))
        self.project_in = nn.Conv2d(channels, 2 * hidden, kernel_size=1)
        self.dwconv = nn.Conv2d(2 * hidden, 2 * hidden, kernel_size=3, padding=1, groups=2 * hidden)
        self.project_out = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class SGFN(nn.Module):
    """Spatial-gate feed-forward network (HSIFormer SGFN, Fig 3b):
    1x1 expand -> GELU -> split -> a * DWConv3x3(b) -> 1x1 project."""
    def __init__(self, channels: int, expansion: float = 2.66) -> None:
        super().__init__()
        hidden = max(1, int(channels * expansion))
        self.project_in = nn.Conv2d(channels, 2 * hidden, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.project_out = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.project_in(x))
        a, b = x.chunk(2, dim=1)
        return self.project_out(a * self.dwconv(b))


class DualTransformerBlock(nn.Module):
    """HSIFormer Spectral-Spatial Transformer Block (SSTB).

    ``Xout = SST(ChannelGate(Xin)) + Xin`` where SST is two pre-norm residual
    pairs (paper Eqs. 2, 4, 5):
        spectral:  h = g + S-MSA(LN(g));  h = h + GDFN(LN(h))
        spatial:   h = h + CSwin(LN(h));  h = h + SGFN(LN(h))
    A CBAM channel gate refines the input before the dual transformer, and the
    outer residual adds the original (ungated) input. The spectral S-MSA is the
    workhorse (paper ablation: removing it collapses MRAE 0.1497 -> 0.2204).

    The class name is kept (not renamed to SSTB) for checkpoint/test/inference
    compatibility.
    """
    def __init__(
        self,
        channels: int,
        split_size: int = 7,
        num_heads: int = 4,
        config: Optional[ConfigDict] = None,
    ) -> None:
        super(DualTransformerBlock, self).__init__()

        if config is None:
            raise ValueError("config cannot be None for DualTransformerBlock")
        if num_heads is None:
            num_heads = config.get("num_heads", 4)
        ffn_expansion = float(config.get("ffn_expansion", 2.66))

        self.gate = CBAMChannelGate(channels, reduction=int(config.get("cbam_reduction", 4)))
        self.norm1 = ChannelLayerNorm(channels)
        self.norm2 = ChannelLayerNorm(channels)
        self.norm3 = ChannelLayerNorm(channels)
        self.norm4 = ChannelLayerNorm(channels)

        # Spectral sub-block (workhorse): S-MSA + GDFN. 'efficient' keeps the
        # legacy pre-pooled gate for back-compat, but 's_msa' is the default.
        spectral_type = str(config.get("spectral_attention_type", "s_msa")).lower()
        if spectral_type in ("s_msa", "smsa", "spectral_msa", "mdta"):
            spectral_attn = SpectralMSA(channels, num_heads=num_heads, config=config)
        else:
            spectral_attn = EfficientSpectralAttention(channels, num_heads=num_heads, config=config)
        self.spectral_attn = NaNSafeAttention(spectral_attn)
        self.gdfn = GDFN(channels, expansion=ffn_expansion)

        # Spatial sub-block: CSwin + SGFN.
        spatial_attn = CSWinAttentionBlock(
            channels, num_heads=num_heads, split_size=split_size, config=config
        )
        self.spatial_attn = NaNSafeAttention(spatial_attn)
        self.sgfn = SGFN(channels, expansion=ffn_expansion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.gate(x)
        h = g + self.spectral_attn(self.norm1(g))   # Eq. 4 spectral pair
        h = h + self.gdfn(self.norm2(h))
        h = h + self.spatial_attn(self.norm3(h))     # Eq. 5 spatial pair
        h = h + self.sgfn(self.norm4(h))
        return h + x                                  # Eq. 2 outer residual


class DynamicDownsampleBlock(nn.Module):
    """Downsampling block with adaptive normalization."""
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        scale_factor: int = 2, 
        config: Optional[ConfigDict] = None
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
        config: Optional[ConfigDict] = None
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


class PixelUnshuffleDownsample(nn.Module):
    """Learned, near-lossless 2x downsample: PixelUnshuffle rearranges spatial
    detail into channels (no information thrown away, unlike bilinear), then a
    3x3 conv mixes it. Requires even H,W (the generator pads to a multiple of
    the total downsample factor before the encoder)."""
    def __init__(self, in_channels: int, out_channels: int, config: Optional[ConfigDict] = None) -> None:
        super().__init__()
        base_groups = config.get("norm_groups", 8) if config else 8
        self.unshuffle = nn.PixelUnshuffle(2)
        self.conv = nn.Conv2d(in_channels * 4, out_channels, kernel_size=3, padding=1)
        self.norm = adaptive_group_norm(out_channels, base_groups)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.unshuffle(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class PixelShuffleUpsample(nn.Module):
    """Learned 2x upsample: 3x3 conv expands channels 4x, PixelShuffle folds
    them back into space, then a 3x3 conv smooths PixelShuffle's characteristic
    checkerboard pattern. Recovers high-frequency detail bilinear interpolation
    cannot."""
    def __init__(self, in_channels: int, out_channels: int, config: Optional[ConfigDict] = None) -> None:
        super().__init__()
        base_groups = config.get("norm_groups", 8) if config else 8
        self.expand = nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        self.smooth = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm = adaptive_group_norm(out_channels, base_groups)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.shuffle(self.expand(x))
        x = self.smooth(x)
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
    def __init__(self, config: ConfigDict) -> None:
        super(NoiseRobustCSWinGenerator, self).__init__()
        
        # Extract parameters
        in_channels = config.get("in_channels", 3)
        out_channels = config.get("out_channels", 31)
        base_channels = config.get("base_channels", 64)
        split_sizes = list(config.get("split_sizes", [7, 7, 7]))
        if len(split_sizes) == 0:
            raise ValueError("split_sizes must contain at least one value")
        # Keep backward compatibility with older 1-2 stage configs used in tests/scripts.
        if len(split_sizes) < 3:
            split_sizes.extend([split_sizes[-1]] * (3 - len(split_sizes)))
        elif len(split_sizes) > 3:
            split_sizes = split_sizes[:3]
        base_groups = config.get("norm_groups", 8)
        num_heads = config.get("num_heads", 4)
        
        # Output activation configuration
        self.output_activation = config.get("output_activation", "none")
        self.activation_delay_iters = config.get("activation_delay_iters", 20000)
        self.clamp_range = config.get("generator_clamp_range", 10.0)  # Configurable clamp
        self.clamp_after_iters = config.get("clamp_after_iters", 0)  # Can disable clamping after warmup
        self.register_buffer('iteration_count', torch.zeros(1, dtype=torch.long))
        self._iteration_count: int = 0
        # When the trainer calls ``set_iteration`` we switch off the
        # forward-pass auto-increment so the two paths do not double-count.
        # Auto-increment then only fires for callers that never call
        # ``set_iteration`` (legacy/test scripts).
        self._iteration_externally_managed: bool = False
        
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
        
        # Depth per stage in order [encoder1, encoder2, bottleneck, decoder1,
        # decoder2]. HSIFormer uses [2, 2, 2, 2, 4] — extra blocks at the final
        # high-resolution decoder stage (cheap, where detail is recovered).
        # ``stage_depths`` overrides; otherwise fall back to a scalar
        # ``blocks_per_stage`` replicated across all five stages.
        default_depth = max(1, int(config.get("blocks_per_stage", 2)))
        stage_depths = config.get("stage_depths", None)
        if stage_depths is None:
            stage_depths = [default_depth] * 5
        stage_depths = [max(1, int(d)) for d in stage_depths]
        if len(stage_depths) != 5:
            stage_depths = (list(stage_depths) + [default_depth] * 5)[:5]
        self._stage_depths = stage_depths

        def make_stage(ch: int, split: int, n_blocks: int) -> nn.Sequential:
            return nn.Sequential(*[
                DualTransformerBlock(ch, split_size=split, num_heads=num_heads, config=config)
                for _ in range(n_blocks)
            ])

        # Sampling: 'pixelshuffle' = learned PixelUnshuffle down / PixelShuffle
        # up (recovers high-frequency detail bilinear interpolation discards);
        # 'bilinear' = legacy F.interpolate. Default 'bilinear' for back-compat.
        sampling = str(config.get("sampling", "bilinear")).lower()
        if sampling in ("pixelshuffle", "pixel_shuffle", "ps"):
            DownBlock, UpBlock = PixelUnshuffleDownsample, PixelShuffleUpsample
            # Two stride-2 stages: pad H,W to a multiple of 4 in forward so
            # PixelUnshuffle always sees even dims.
            self._size_multiple = 4
        else:
            DownBlock, UpBlock = DynamicDownsampleBlock, DynamicUpsampleBlock
            self._size_multiple = 1

        # Encoder
        self.encoder1 = make_stage(base_channels, split_sizes[0], stage_depths[0])
        self.down1 = DownBlock(base_channels, base_channels*2, config=config)

        self.encoder2 = make_stage(base_channels*2, split_sizes[1], stage_depths[1])
        self.down2 = DownBlock(base_channels*2, base_channels*4, config=config)

        # Bottleneck
        self.bottleneck = make_stage(base_channels*4, split_sizes[2], stage_depths[2])

        # Decoder
        self.up1 = UpBlock(base_channels*4, base_channels*2, config=config)
        self.decoder1 = make_stage(base_channels*4, split_sizes[1], stage_depths[3])
        self.compressor1 = nn.Conv2d(base_channels*4, base_channels*2, kernel_size=1)

        self.up2 = UpBlock(base_channels*2, base_channels, config=config)
        self.compressor2 = nn.Conv2d(base_channels*2, base_channels, kernel_size=1)
        self.decoder2 = make_stage(base_channels, split_sizes[0], stage_depths[4])

        # Output head. Optional nonlinear lift (3x3 -> GELU -> 1x1) for a
        # richer spectral mapping than a single linear conv.
        if bool(config.get("thick_output_head", False)):
            self.to_spectral = nn.Sequential(
                nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(base_channels, out_channels, kernel_size=1),
            )
        else:
            self.to_spectral = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True):
        result = super().load_state_dict(state_dict, strict)
        try:
            self._iteration_count = int(self.iteration_count.detach().cpu().item())
        except Exception:
            self._iteration_count = 0
        return result

    def set_iteration(self, iteration: int) -> None:
        """Set the canonical training-step counter.

        The trainer should call this once per optimizer step so that
        delayed-sigmoid and clamp-after-iters thresholds key off true
        optimizer-step count, not per-forward calls.

        Calling this also flips ``_iteration_externally_managed=True`` so the
        forward-pass auto-increment does not fight the trainer's value.
        See ``forward`` for the legacy auto-increment fallback.
        """
        iteration = max(int(iteration), 0)
        self._iteration_count = iteration
        self._iteration_externally_managed = True
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            with torch.no_grad():
                self.iteration_count.fill_(iteration)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Auto-increment fallback for callers that do not invoke
        # ``set_iteration``. Once ``set_iteration`` has been called the
        # trainer is treated as the source of truth and forward stops
        # incrementing — otherwise the two paths double-counted (probe v2
        # showed counter advancing by 2 per optimizer step under
        # accumulation_steps=2 even though the trainer set the value at the
        # top of every loop iteration).
        if self.training and not self._iteration_externally_managed:
            self._iteration_count += 1
            if not torch.jit.is_scripting() and not torch.jit.is_tracing():
                with torch.no_grad():
                    self.iteration_count.add_(1)
        iter_idx = self._iteration_count

        # Pad to a multiple of the total downsample factor so learned
        # (PixelUnshuffle) downsampling sees even dims at every stage. No-op for
        # bilinear sampling (_size_multiple == 1), which handles arbitrary sizes
        # via the skip-connection interpolation fallbacks below.
        in_h, in_w = x.shape[-2], x.shape[-1]
        size_multiple = getattr(self, "_size_multiple", 1)
        pad_h = (size_multiple - in_h % size_multiple) % size_multiple
        pad_w = (size_multiple - in_w % size_multiple) % size_multiple
        if pad_h or pad_w:
            pad_mode = "reflect"
            if (pad_h and in_h <= pad_h) or (pad_w and in_w <= pad_w):
                pad_mode = "replicate"
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=pad_mode)

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
            # Validate both tensors before interpolation
            if x.shape[2] < 1 or x.shape[3] < 1:
                raise ValueError(f"Invalid spatial size before interpolation: {x.shape}")
            if e2.shape[2] < 1 or e2.shape[3] < 1:
                raise ValueError(f"Invalid spatial size for encoder stage e2: {e2.shape}")
            x = F.interpolate(x, size=e2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, e2], dim=1)
        x = self.decoder1(x)
        x = self.compressor1(x)
        
        x = self.up2(x)
        if x.shape[2:] != e1.shape[2:]:
            # Validate both tensors before interpolation
            if x.shape[2] < 1 or x.shape[3] < 1:
                raise ValueError(f"Invalid spatial size before interpolation: {x.shape}")
            if e1.shape[2] < 1 or e1.shape[3] < 1:
                raise ValueError(f"Invalid spatial size for encoder stage e1: {e1.shape}")
            x = F.interpolate(x, size=e1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, e1], dim=1)
        x = self.compressor2(x)  # Apply before decoder2
        x = self.decoder2(x)
        
        # Handle dynamic spatial dimensions for residual connection
        if x.shape[2:] != emb.shape[2:]:
            # Validate both tensors before interpolation
            if x.shape[2] < 1 or x.shape[3] < 1:
                raise ValueError(f"Invalid spatial size before interpolation: {x.shape}")
            if emb.shape[2] < 1 or emb.shape[3] < 1:
                raise ValueError(f"Invalid spatial size for embedding residual: {emb.shape}")
            x = F.interpolate(x, size=emb.shape[2:], mode='bilinear', align_corners=False)
        
        # Output with residual connection
        x = self.to_spectral(x + emb)
        
        # Apply output activation based on configuration
        delayed_sigmoid_active = (
            self.output_activation == "delayed_sigmoid"
            and iter_idx > self.activation_delay_iters
        )
        if self.output_activation == "sigmoid":
            x = torch.sigmoid(x)
        elif self.output_activation == "tanh":
            x = 0.5 * (torch.tanh(x) + 1.0)  # Map to [0, 1]
        elif delayed_sigmoid_active:
            x = torch.sigmoid(x)
        # else: no activation (linear output)

        # Soft clipping to prevent extreme values during the linear-output
        # phase. Apply for both ``output_activation == 'none'`` AND for
        # ``delayed_sigmoid`` while the sigmoid has not yet been switched on,
        # since the pre-sigmoid logits are unconstrained and can blow up
        # under early adversarial training. Sigmoid/tanh paths need no clamp.
        in_linear_phase = (
            self.output_activation == "none"
            or (self.output_activation == "delayed_sigmoid" and not delayed_sigmoid_active)
        )
        if self.training and in_linear_phase:
            if self.clamp_after_iters == 0 or iter_idx < self.clamp_after_iters:
                x = torch.clamp(x, -self.clamp_range, self.clamp_range)

        # Remove the symmetric padding added for learned downsampling.
        if pad_h or pad_w:
            x = x[..., :in_h, :in_w]

        return x
