# CAS-HSI Implementation Guide — Steps 3 to 9

> Normative specification. This file is the contract the `cas_hsi` package implements.
> (Transcribed from the user-supplied design document; the ASCII diagrams have been
> re-encoded, no requirements were changed.)

## Scope

This document specifies the implementation of the **Convolutional Attention Stack for
Hyperspectral Reconstruction (CAS-HSI)**, covering:

3. Overall architecture
4. Convolutional Attention Stack block
5. CAS-Lite block for full-resolution processing
6. Downsampling and upsampling
7. Spectral prediction head
8. Arbitrary image-size handling
9. Edge-deployment profile

The target task is RGB-to-hyperspectral reconstruction:

    X_RGB in R^(B x 3 x H x W)  -->  Y_HSI in R^(B x N_lambda x H x W)

with N_lambda = 31 by default.

The design must be:

- fully convolutional;
- independent of fixed input resolution;
- suitable for patch-based training and full-image inference;
- compatible with mixed precision;
- exportable to ONNX or TensorRT with a convolutional fallback path;
- modular enough to support controlled architectural ablations.

---

# 3. Overall Architecture

## 3.1 High-level topology

CAS-HSI is a three-resolution encoder-decoder with:

- a shallow RGB embedding stem;
- two encoder stages;
- a low-resolution bottleneck;
- two decoder stages;
- encoder-decoder skip connections;
- a learned RGB-to-HSI prior;
- a residual spectral prediction head.

```text
RGB input: B x 3 x H x W
    |
    +-- Learned RGB->HSI prior: Conv1x1(3->31) ---------------------------+
    |                                                                     |
    v                                                                     |
Reflect-pad to dimensions divisible by 4                                  |
    |                                                                     |
Conv3x3, 3->C                                                             |
    |                                                                     |
Encoder Stage 0: CAS-Lite x D0              B x C x H x W                 |
    |                       ------------- Skip S0                         |
PixelUnshuffle(2) -> Conv1x1, 4C->2C                                      |
    |                                                                     |
Encoder Stage 1: CAS x D1                   B x 2C x H/2 x W/2            |
    |                       ------------- Skip S1                         |
PixelUnshuffle(2) -> Conv1x1, 8C->4C                                      |
    |                                                                     |
Bottleneck: CAS x DB                        B x 4C x H/4 x W/4            |
    |                                                                     |
Conv1x1, 4C->8C -> PixelShuffle(2)          B x 2C x H/2 x W/2            |
Concatenate S1 -> Conv1x1, 4C->2C                                         |
    |                                                                     |
Decoder Stage 1: CAS x DD1                  B x 2C x H/2 x W/2            |
    |                                                                     |
Conv1x1, 2C->4C -> PixelShuffle(2)          B x C x H x W                 |
Concatenate S0 -> Conv1x1, 2C->C                                          |
    |                                                                     |
Decoder Stage 0: CAS-Lite x DD0             B x C x H x W                 |
    |                                                                     |
Refinement: CAS-Lite x DR                   B x C x H x W                 |
    |                                                                     |
Conv3x3, C->31 = spectral residual ------------------------------------->-+
                                                                          v
                                                          Prior + residual
                                                                          |
                                                           Crop to H x W
                                                                          |
                                                          B x 31 x H x W
```

---

## 3.2 Recommended model variants

### CAS-HSI-Tiny

Use as the first implementation and deployment baseline.

```yaml
model:
  name: cas_hsi_tiny
  input_channels: 3
  output_bands: 31
  base_width: 32

  channels:
    full: 32
    half: 64
    quarter: 128

  depths:
    encoder_full: 2
    encoder_half: 3
    bottleneck: 5
    decoder_half: 3
    decoder_full: 2
    refinement: 2

  head_dim: 32
  ffn_expansion: 2.0
  layer_scale_init: 1.0e-3
```

### CAS-HSI-Base

Use when accuracy is more important than minimum latency.

```yaml
model:
  name: cas_hsi_base
  input_channels: 3
  output_bands: 31
  base_width: 48

  channels:
    full: 48
    half: 96
    quarter: 192

  depths:
    encoder_full: 2
    encoder_half: 4
    bottleneck: 6
    decoder_half: 4
    decoder_full: 2
    refinement: 2

  head_dim: 32
  ffn_expansion: 2.0
  layer_scale_init: 1.0e-3
```

---

## 3.3 Module decomposition

Implement the network using independent modules.

```text
cas_hsi/
|-- model.py
|-- config.py
|-- layers/
|   |-- normalization.py
|   |-- layer_scale.py
|   |-- padding.py
|   |-- downsample.py
|   |-- upsample.py
|   |-- spectral_head.py
|-- blocks/
|   |-- cas_block.py
|   |-- cas_lite_block.py
|   |-- spatial_attention.py
|   |-- channel_attention.py
|   |-- gated_ffn.py
|-- deployment/
|   |-- export_onnx.py
|   |-- replace_attention.py
|   |-- quantization.py
|-- tests/
    |-- test_shapes.py
    |-- test_arbitrary_sizes.py
    |-- test_gradients.py
    |-- test_export.py
    |-- test_equivalence.py
```

---

## 3.4 Top-level model pseudocode

```python
class CASHSI(nn.Module):
    def __init__(self, config):
        super().__init__()

        c = config.base_width
        bands = config.output_bands

        self.stem = nn.Conv2d(
            config.input_channels,
            c,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.rgb_prior = nn.Conv2d(
            config.input_channels,
            bands,
            kernel_size=1,
            bias=True,
        )

        self.encoder_full = nn.Sequential(
            *[
                CASLiteBlock(
                    channels=c,
                    head_dim=config.head_dim,
                    ffn_expansion=config.ffn_expansion,
                    layer_scale_init=config.layer_scale_init,
                )
                for _ in range(config.depths.encoder_full)
            ]
        )

        self.down_1 = PixelUnshuffleDownsample(c, 2 * c)

        self.encoder_half = nn.Sequential(
            *[
                CASBlock(
                    channels=2 * c,
                    spatial_mixer="dilated_local_attention",
                    head_dim=config.head_dim,
                    dilations=(1, 2),
                    ffn_expansion=config.ffn_expansion,
                    layer_scale_init=config.layer_scale_init,
                )
                for _ in range(config.depths.encoder_half)
            ]
        )

        self.down_2 = PixelUnshuffleDownsample(2 * c, 4 * c)

        self.bottleneck = build_bottleneck(config, channels=4 * c)

        self.up_1 = PixelShuffleUpsample(4 * c, 2 * c)
        self.skip_fusion_1 = nn.Conv2d(4 * c, 2 * c, kernel_size=1)

        self.decoder_half = nn.Sequential(
            *[
                CASBlock(
                    channels=2 * c,
                    spatial_mixer="dilated_local_attention",
                    head_dim=config.head_dim,
                    dilations=(1, 2),
                    ffn_expansion=config.ffn_expansion,
                    layer_scale_init=config.layer_scale_init,
                )
                for _ in range(config.depths.decoder_half)
            ]
        )

        self.up_2 = PixelShuffleUpsample(2 * c, c)
        self.skip_fusion_0 = nn.Conv2d(2 * c, c, kernel_size=1)

        self.decoder_full = nn.Sequential(
            *[
                CASLiteBlock(
                    channels=c,
                    head_dim=config.head_dim,
                    ffn_expansion=config.ffn_expansion,
                    layer_scale_init=config.layer_scale_init,
                )
                for _ in range(config.depths.decoder_full)
            ]
        )

        self.refinement = nn.Sequential(
            *[
                CASLiteBlock(
                    channels=c,
                    head_dim=config.head_dim,
                    ffn_expansion=config.ffn_expansion,
                    layer_scale_init=config.layer_scale_init,
                )
                for _ in range(config.depths.refinement)
            ]
        )

        self.spectral_head = nn.Conv2d(
            c,
            bands,
            kernel_size=3,
            padding=1,
        )

    def forward(self, rgb):
        rgb_padded, pad_info = pad_to_multiple(rgb, multiple=4)

        prior = self.rgb_prior(rgb_padded)

        x0 = self.stem(rgb_padded)
        s0 = self.encoder_full(x0)

        x1 = self.down_1(s0)
        s1 = self.encoder_half(x1)

        x2 = self.down_2(s1)
        x2 = self.bottleneck(x2)

        y1 = self.up_1(x2)
        y1 = self.skip_fusion_1(torch.cat([y1, s1], dim=1))
        y1 = self.decoder_half(y1)

        y0 = self.up_2(y1)
        y0 = self.skip_fusion_0(torch.cat([y0, s0], dim=1))
        y0 = self.decoder_full(y0)
        y0 = self.refinement(y0)

        residual = self.spectral_head(y0)
        output = prior + residual

        return crop_to_original(output, pad_info)
```

---

## 3.5 Bottleneck construction

The bottleneck should alternate local multi-dilation attention and occasional stripe attention.

Example for five blocks:

```text
Block 0: dilated local attention
Block 1: dilated local attention
Block 2: hybrid local + axial stripe attention
Block 3: dilated local attention
Block 4: dilated local attention
```

```python
def build_bottleneck(config, channels):
    blocks = []

    for index in range(config.depths.bottleneck):
        use_stripe = (
            config.enable_stripe_attention
            and (index + 1) % config.stripe_frequency == 0
        )

        mixer = (
            "hybrid_local_stripe_attention"
            if use_stripe
            else "dilated_local_attention"
        )

        blocks.append(
            CASBlock(
                channels=channels,
                spatial_mixer=mixer,
                head_dim=config.head_dim,
                dilations=(1, 2, 3),
                ffn_expansion=config.ffn_expansion,
                layer_scale_init=config.layer_scale_init,
            )
        )

    return nn.Sequential(*blocks)
```

---

## 3.6 Construction-time validation

Fail immediately when configuration values are inconsistent.

```python
def validate_config(config):
    widths = [
        config.base_width,
        2 * config.base_width,
        4 * config.base_width,
    ]

    for width in widths:
        if width % config.head_dim != 0:
            raise ValueError(
                f"Channel width {width} must be divisible by "
                f"head_dim={config.head_dim}."
            )

    if config.ffn_expansion <= 1.0:
        raise ValueError("FFN expansion must be greater than 1.")

    if config.output_bands <= 0:
        raise ValueError("output_bands must be positive.")

    if config.size_multiple != 4:
        raise ValueError(
            "The two-stage encoder requires size_multiple=4."
        )
```

---

# 4. Convolutional Attention Stack Block

## 4.1 Residual formulation

For input X in R^(B x C x H x W):

    X1 = X  + gamma_s (*) S(Norm_s(X))
    X2 = X1 + gamma_c (*) C(Norm_c(X1))
    Y  = X2 + gamma_f (*) F(Norm_f(X2))

where:

- S is the spatial mixer;
- C is cross-channel self-attention;
- F is the gated convolutional feed-forward network;
- each gamma is a learnable vector in R^C;
- each residual branch is pre-normalized;
- the identity path is never scaled.

---

## 4.2 CAS block interface

```python
class CASBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        spatial_mixer: str,
        head_dim: int = 32,
        dilations: tuple[int, ...] = (1, 2),
        ffn_expansion: float = 2.0,
        layer_scale_init: float = 1e-3,
        drop_path: float = 0.0,
    ):
        super().__init__()

        self.norm_spatial = BiasFreeLayerNorm2d(channels)
        self.norm_channel = BiasFreeLayerNorm2d(channels)
        self.norm_ffn = BiasFreeLayerNorm2d(channels)

        self.spatial_mixer = build_spatial_mixer(
            name=spatial_mixer,
            channels=channels,
            head_dim=head_dim,
            dilations=dilations,
        )

        self.channel_attention = CrossChannelAttention(
            channels=channels,
            head_dim=head_dim,
        )

        self.ffn = GatedConvFFN(
            channels=channels,
            expansion=ffn_expansion,
        )

        self.gamma_spatial = LayerScale(channels, init_value=layer_scale_init)
        self.gamma_channel = LayerScale(channels, init_value=layer_scale_init)
        self.gamma_ffn = LayerScale(channels, init_value=layer_scale_init)

        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path(
            self.gamma_spatial(self.spatial_mixer(self.norm_spatial(x)))
        )

        x = x + self.drop_path(
            self.gamma_channel(self.channel_attention(self.norm_channel(x)))
        )

        x = x + self.drop_path(
            self.gamma_ffn(self.ffn(self.norm_ffn(x)))
        )

        return x
```

For small models, set `drop_path=0.0`. Introduce stochastic depth only after the
baseline is stable.

---

## 4.3 Bias-free LayerNorm2d

Normalize over the channel dimension independently at each spatial location.

    LN(x_p) = x_p / sqrt(Var(x_p) + eps) (*) w

The bias-free form avoids unnecessary recentering.

```python
class BiasFreeLayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.eps = eps

    def forward(self, x):
        variance = x.var(dim=1, keepdim=True, unbiased=False)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight.view(1, -1, 1, 1)
```

Alternative: RMSNorm2d.

Do not use BatchNorm in the restoration backbone.

---

## 4.4 LayerScale

```python
class LayerScale(nn.Module):
    def __init__(self, channels, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.full((channels,), float(init_value)))

    def forward(self, x):
        return x * self.scale.view(1, -1, 1, 1)
```

Initialization:

```text
Tiny/Base model: 1e-3
Very deep model:  1e-4
Shallow ablation: 1e-2
```

The identity path must remain exactly unscaled.

---

## 4.5 Spatial attention

### 4.5.1 QKV projection

Use a pointwise projection followed by depthwise convolution:

    [Q, K, V] = DWConv3x3( Conv1x1^{3C}(X) )

```python
self.qkv = nn.Sequential(
    nn.Conv2d(channels, 3 * channels, kernel_size=1, bias=False),
    nn.Conv2d(
        3 * channels,
        3 * channels,
        kernel_size=3,
        padding=1,
        groups=3 * channels,
        bias=False,
    ),
)
```

This gives Q, K, and V local positional context without learned absolute positional
embeddings.

---

### 4.5.2 Head allocation

Let h = C / d_h, where d_h is the head dimension.

For C = 128 and d_h = 32:  h = 4.

Example head grouping:

```text
Head 0: local neighborhood, dilation 1
Head 1: local neighborhood, dilation 2
Head 2: local neighborhood, dilation 3
Head 3: local neighborhood, dilation 3
```

For hybrid stripe blocks:

```text
Head 0: local neighborhood, dilation 1
Head 1: local neighborhood, dilation 2
Head 2: horizontal stripe
Head 3: vertical stripe
```

The number of heads assigned to each group must sum to the total number of heads.

---

### 4.5.3 Dilated local attention

For a query pixel p, define the neighborhood

    N_d(p) = { p + d * delta | delta in {-1,0,1} x {-1,0,1} }

Attention for head group g:

    A_g(p, q) = softmax_{q in N_{d_g}(p)} ( Q_g(p)^T K_g(q) / sqrt(d_h) + b_g(p - q) )
    O_g(p)    = sum_{q in N_{d_g}(p)} A_g(p, q) V_g(q)

Implementation options:

1. `torch.nn.functional.unfold` for the reference implementation;
2. custom gather kernels for optimization;
3. dedicated neighborhood-attention operators where deployment permits.

Start with the unfold implementation for correctness.

```python
def local_attention_reference(q, k, v, kernel_size, dilation):
    # q, k, v: [B, heads, head_dim, H, W]
    # returns:  [B, heads, head_dim, H, W]

    b, h, d, height, width = q.shape
    padding = dilation * (kernel_size // 2)

    k_patches = unfold_heads(k, kernel_size=kernel_size, dilation=dilation, padding=padding)
    v_patches = unfold_heads(v, kernel_size=kernel_size, dilation=dilation, padding=padding)

    q = q.unsqueeze(-1)

    logits = (q * k_patches).sum(dim=2)
    logits = logits / math.sqrt(d)

    weights = logits.softmax(dim=-1)

    output = (weights.unsqueeze(2) * v_patches).sum(dim=-1)

    return output
```

Avoid very large dilation values. For 3x3 neighborhoods, use:

```text
H/2 stage: [1, 2]
H/4 stage: [1, 2, 3]
```

---

### 4.5.4 Stripe attention

At low resolution, split selected heads into horizontal and vertical groups.

Horizontal attention: query tokens attend within horizontal stripes.
Vertical attention:   query tokens attend within vertical stripes.

Use stripe width:

```text
default: 8
small bottleneck feature map: min(8, H, W)
```

Do not require dimensions divisible by the stripe width. Pad internally and crop after
attention.

Recommended usage: one hybrid stripe block every 3 bottleneck blocks.

Do not use stripe attention at full resolution in the initial model.

---

### 4.5.5 Spatial output projection

Concatenate head-group outputs, then project:

    O    = Concat(O_1, ..., O_G)
    S(X) = Conv1x1(O)

```python
self.output_projection = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
```

Do not concatenate full DilateFormer and CSWin branches.

---

## 4.6 Cross-channel attention

### 4.6.1 Purpose

Cross-channel attention models interactions between latent feature channels while
aggregating evidence over all spatial locations.

It is not a pooled sigmoid gate.

---

### 4.6.2 Tensor transformation

Input X in R^(B x C x H x W). Generate convolutionally conditioned Q, K, V:

    [Q, K, V] = DWConv3x3( Conv1x1^{3C}(X) )

Reshape to Q, K, V in R^(B x h x d_h x HW).

Normalize Q and K over the spatial dimension:

    Qhat = Q / (||Q||_2 + eps),   Khat = K / (||K||_2 + eps)

Compute channel covariance attention:

    A_c = softmax( Qhat Khat^T / tau )    with A_c in R^(B x h x d_h x d_h)

Then O_c = A_c V. Finally reshape and project:

    C(X) = Conv1x1( reshape(O_c) )

---

### 4.6.3 Reference implementation

```python
class CrossChannelAttention(nn.Module):
    def __init__(self, channels, head_dim=32):
        super().__init__()

        if channels % head_dim != 0:
            raise ValueError("channels must be divisible by head_dim")

        self.channels = channels
        self.head_dim = head_dim
        self.num_heads = channels // head_dim

        self.qkv = nn.Sequential(
            nn.Conv2d(channels, 3 * channels, kernel_size=1, bias=False),
            nn.Conv2d(
                3 * channels,
                3 * channels,
                kernel_size=3,
                padding=1,
                groups=3 * channels,
                bias=False,
            ),
        )

        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))

        self.output_projection = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, self.head_dim, h * w)
        k = k.reshape(b, self.num_heads, self.head_dim, h * w)
        v = v.reshape(b, self.num_heads, self.head_dim, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attention = torch.matmul(q, k.transpose(-2, -1))

        attention = attention * self.temperature
        attention = attention.softmax(dim=-1)

        output = torch.matmul(attention, v)

        output = output.reshape(b, c, h, w)
        output = self.output_projection(output)

        return output
```

Constrain or parameterize temperature to avoid pathological values if instability
appears. Example: `temperature = F.softplus(raw_temperature) + 1e-4`.

---

## 4.7 Gated convolutional feed-forward network

### 4.7.1 Formulation

For normalized input X:

    T          = DWConv3x3( Conv1x1^{2rC}(X) )
    T          = [T1, T2]
    G          = T1 (*) T2
    F(X)       = Conv1x1^{C}(G)

Recommended expansion: r = 2.

---

### 4.7.2 Reference implementation

```python
class GatedConvFFN(nn.Module):
    def __init__(self, channels, expansion=2.0, use_activation=False):
        super().__init__()

        hidden = int(round(channels * expansion))

        self.input_projection = nn.Conv2d(channels, 2 * hidden, kernel_size=1, bias=False)

        self.depthwise = nn.Conv2d(
            2 * hidden,
            2 * hidden,
            kernel_size=3,
            padding=1,
            groups=2 * hidden,
            bias=False,
        )

        self.output_projection = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)

        self.use_activation = use_activation

    def forward(self, x):
        x = self.input_projection(x)
        x = self.depthwise(x)

        x1, x2 = x.chunk(2, dim=1)

        if self.use_activation:
            x1 = F.silu(x1)

        x = x1 * x2
        x = self.output_projection(x)

        return x
```

Default:

```yaml
ffn:
  expansion: 2.0
  use_activation: false
```

Ablate: SimpleGate `x1 * x2`; SiLU gate `SiLU(x1) * x2`; GEGLU-like `GELU(x1) * x2`.

Do not use a terminal sigmoid gate as the default.

---

## 4.8 CAS block validation tests

### Identity behavior

With all LayerScale parameters set to zero: CAS(X) = X.

```python
def test_cas_identity_at_zero_layerscale():
    block = CASBlock(
        channels=64,
        spatial_mixer="dilated_local_attention",
        head_dim=32,
        layer_scale_init=0.0,
    )

    x = torch.randn(2, 64, 32, 48)
    y = block(x)

    assert torch.allclose(x, y, atol=1e-6)
```

### Shape preservation

```python
@pytest.mark.parametrize("shape", [(1, 64, 32, 32), (2, 64, 31, 47), (1, 128, 17, 29)])
def test_cas_shape(shape):
    block = CASBlock(...)
    x = torch.randn(*shape)
    y = block(x)
    assert y.shape == x.shape
```

### Gradient validity

```python
def test_cas_gradients_are_finite():
    block = CASBlock(...)
    x = torch.randn(2, 64, 32, 32, requires_grad=True)

    loss = block(x).square().mean()
    loss.backward()

    assert torch.isfinite(x.grad).all()
```

---

# 5. CAS-Lite Block

## 5.1 Purpose

At full spatial resolution, explicit local attention can be memory-bound and difficult to
export efficiently.

CAS-Lite preserves the transformer-style block structure but replaces spatial attention
with a large-kernel depthwise convolutional token mixer.

The block remains:

```text
PreNorm -> convolutional spatial mixer -> residual
PreNorm -> cross-channel attention     -> residual
PreNorm -> gated convolutional FFN     -> residual
```

---

## 5.2 Spatial mixer

    S_lite(X) = Conv1x1( DWConv7x7(X) )

Optional two-stage version:

    S_lite(X) = Conv1x1( DWConv7x7( Conv1x1(X) ) )

Use the simpler version first.

---

## 5.3 Reference implementation

```python
class ConvSpatialMixer(nn.Module):
    def __init__(self, channels, kernel_size=7):
        super().__init__()

        padding = kernel_size // 2

        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            bias=False,
        )

        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))
```

```python
class CASLiteBlock(nn.Module):
    def __init__(
        self,
        channels,
        head_dim=32,
        ffn_expansion=2.0,
        layer_scale_init=1e-3,
        spatial_kernel=7,
    ):
        super().__init__()

        self.norm_spatial = BiasFreeLayerNorm2d(channels)
        self.norm_channel = BiasFreeLayerNorm2d(channels)
        self.norm_ffn = BiasFreeLayerNorm2d(channels)

        self.spatial_mixer = ConvSpatialMixer(channels, kernel_size=spatial_kernel)
        self.channel_attention = CrossChannelAttention(channels, head_dim=head_dim)
        self.ffn = GatedConvFFN(channels, expansion=ffn_expansion)

        self.gamma_spatial = LayerScale(channels, layer_scale_init)
        self.gamma_channel = LayerScale(channels, layer_scale_init)
        self.gamma_ffn = LayerScale(channels, layer_scale_init)

    def forward(self, x):
        x = x + self.gamma_spatial(self.spatial_mixer(self.norm_spatial(x)))
        x = x + self.gamma_channel(self.channel_attention(self.norm_channel(x)))
        x = x + self.gamma_ffn(self.ffn(self.norm_ffn(x)))
        return x
```

---

## 5.4 Optional reparameterizable edge mixer

For deployment, the full-resolution token mixer may use parallel depthwise branches
during training:

```text
DWConv 3x3 + DWConv 5x5 + DWConv 7x7 + identity
```

Then fuse them into a single equivalent depthwise kernel for inference.

    Training form:   Y = X + D3(X) + D5(X) + D7(X)
    Deployment form: Y = D_fused(X)

This is an optional optimization. Implement only after the baseline is correct and
measured.

---

## 5.5 CAS-Lite acceptance criteria

- input and output shapes are identical;
- no fixed spatial-size assumptions;
- no tensor unfolding;
- ONNX export succeeds;
- identity behavior is exact when LayerScale is zero;
- latency is lower than full local attention at H x W.

---

# 6. Downsampling and Upsampling

## 6.1 Downsampling requirements

Downsampling must:

- preserve subpixel information before projection;
- reduce spatial dimensions by exactly two;
- increase latent width;
- support odd original image sizes after outer padding;
- avoid BatchNorm.

Use: `PixelUnshuffle(2) -> 1x1 projection`.

---

## 6.2 PixelUnshuffle downsampling

    X  in R^(B x C_in x H x W)
    X' in R^(B x 4C_in x H/2 x W/2)      (after PixelUnshuffle)
    Y  in R^(B x C_out x H/2 x W/2)      (after pointwise projection)

```python
class PixelUnshuffleDownsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.unshuffle = nn.PixelUnshuffle(2)

        self.projection = nn.Conv2d(4 * in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        if x.shape[-2] % 2 != 0:
            raise ValueError("Input height must be divisible by 2.")
        if x.shape[-1] % 2 != 0:
            raise ValueError("Input width must be divisible by 2.")

        x = self.unshuffle(x)
        x = self.projection(x)

        return x
```

For the proposed hierarchy: `C -> 2C`, then `2C -> 4C`.

---

## 6.3 Upsampling requirements

Upsampling must:

- increase resolution by exactly two;
- reduce latent channel width;
- support skip concatenation;
- avoid checkerboard artifacts associated with transposed convolution;
- preserve deployment compatibility.

Use: `1x1 expansion -> PixelShuffle(2)`.

---

## 6.4 PixelShuffle upsampling

To produce C_out channels after PixelShuffle(2), project to 4*C_out channels first.

```python
class PixelShuffleUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.projection = nn.Conv2d(in_channels, 4 * out_channels, kernel_size=1, bias=False)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.projection(x)
        x = self.shuffle(x)
        return x
```

For the proposed hierarchy: `4C -> 2C`, then `2C -> C`.

---

## 6.5 Skip fusion

Concatenate decoder and encoder features, then project:

    F_cat   = Concat(F_decoder, F_skip)
    F_fused = Conv1x1(F_cat)

```python
y1 = self.up_1(bottleneck)
y1 = torch.cat([y1, skip_half], dim=1)
y1 = self.skip_fusion_1(y1)
```

Do not add encoder and decoder features directly unless they are known to have compatible
semantics. Concatenation followed by projection is the safer default.

---

## 6.6 Optional gated skip fusion

After the baseline is established, test gated skip fusion:

    G = sigmoid( Conv1x1([F_d, F_e]) )
    F = Conv1x1( [F_d, G (*) F_e] )

This can suppress irrelevant high-frequency encoder features, but it adds complexity and
should not be included in the first implementation.

---

## 6.7 Shape tests

```python
@pytest.mark.parametrize("height,width", [(32, 32), (48, 64), (128, 96)])
def test_down_up_shape(height, width):
    down = PixelUnshuffleDownsample(32, 64)
    up = PixelShuffleUpsample(64, 32)

    x = torch.randn(1, 32, height, width)
    y = up(down(x))

    assert y.shape == x.shape
```

---

# 7. Spectral Prediction Head

## 7.1 Learned RGB-to-HSI prior

Use a direct linear projection:

    Y_prior = P_RGB(X_RGB),   P_RGB = Conv1x1^{3 -> 31}

This branch provides a learned scene-independent linear color-to-spectrum baseline.

```python
self.rgb_prior = nn.Conv2d(3, output_bands, kernel_size=1, bias=True)
```

---

## 7.2 Deep residual spectral head

Let F denote the final decoder feature map.

    dY    = Conv3x3^{C -> 31}(F)
    Yhat  = Y_prior + dY

```python
prior = self.rgb_prior(rgb)
residual = self.spectral_head(features)
prediction = prior + residual
```

This formulation makes the deep model learn:

- nonlinear material-dependent corrections;
- spatial context;
- metamer disambiguation;
- spectral smoothness structure;
- high-frequency residual detail.

---

## 7.3 Initialization

Recommended initialization:

```text
RGB prior:              standard Kaiming or Xavier initialization
Residual spectral head: weights initialized near zero, bias initialized to zero
```

Near-zero residual initialization makes the initial model approximately equal to the
learned linear prior.

```python
nn.init.normal_(self.spectral_head.weight, mean=0.0, std=1e-3)

if self.spectral_head.bias is not None:
    nn.init.zeros_(self.spectral_head.bias)
```

Alternative: `nn.init.zeros_(self.spectral_head.weight)`.

Zero initialization is acceptable because earlier layers remain trainable through the head
after the first update, but a small normal initialization is generally safer.

---

## 7.4 No final nonlinear activation

Training output: `prediction = prior + residual`.

Do not apply SiLU / GELU / sigmoid / tanh / ReLU after the 31-band projection.

For evaluation only:

```python
prediction_for_metrics = prediction.clamp(min=valid_min, max=valid_max)
```

Keep the unclamped tensor for the loss unless the training protocol explicitly requires
bounded output.

---

## 7.5 Optional low-rank spectral basis head

Introduce only as an ablation.

    A      = Conv3x3^{C -> K}(F),  K in [8, 12]
    B      in R^(31 x K)  (learned spectral basis; bias-free 1x1 convolution)
    Y_basis = B A
    Yhat    = Y_prior + Y_basis + eta * dY,  with eta initialized near 0.1

```python
class LowRankSpectralHead(nn.Module):
    def __init__(self, feature_channels, output_bands=31, rank=10, residual_scale=0.1):
        super().__init__()

        self.coefficient_head = nn.Conv2d(feature_channels, rank, kernel_size=3, padding=1)
        self.basis_projection = nn.Conv2d(rank, output_bands, kernel_size=1, bias=False)
        self.residual_head = nn.Conv2d(feature_channels, output_bands, kernel_size=3, padding=1)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, features):
        coefficients = self.coefficient_head(features)
        basis_prediction = self.basis_projection(coefficients)
        residual = self.residual_head(features)

        return basis_prediction + self.residual_scale * residual
```

---

## 7.6 Optional smooth basis constraints

If the learned basis becomes spectrally irregular, add a second-order smoothness
regularizer:

    L_basis = || D_lambda^(2) B ||_1

where D_lambda^(2) is the second finite-difference operator along wavelength.

Do not hard-code smoothness into the forward pass.

---

## 7.7 Spectral head tests

```python
def test_spectral_head_shape():
    model = CASHSI(config)
    rgb = torch.randn(2, 3, 127, 193)

    output = model(rgb)

    assert output.shape == (2, 31, 127, 193)
```

```python
def test_no_output_activation():
    model = CASHSI(config)
    rgb = torch.randn(1, 3, 32, 32)

    output = model(rgb)

    # Negative values should remain representable.
    assert output.dtype == rgb.dtype
```

---

# 8. Arbitrary Image Sizes

## 8.1 Required divisibility

The model downsamples twice by a factor of two. Therefore the padded dimensions must
satisfy `H' % 4 == 0` and `W' % 4 == 0`.

The model must accept the original H and W without requiring the caller to resize the
image.

---

## 8.2 Padding calculation

For size n and multiple m:  p(n, m) = (m - n % m) % m

Use right and bottom padding by default.

```python
def required_padding(size: int, multiple: int) -> int:
    return (multiple - size % multiple) % multiple
```

---

## 8.3 Padding implementation

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class PadInfo:
    original_height: int
    original_width: int
    pad_left: int
    pad_right: int
    pad_top: int
    pad_bottom: int


def pad_to_multiple(
    x: torch.Tensor,
    multiple: int = 4,
    mode: str = "reflect",
) -> tuple[torch.Tensor, PadInfo]:
    _, _, height, width = x.shape

    pad_h = required_padding(height, multiple)
    pad_w = required_padding(width, multiple)

    pad_info = PadInfo(
        original_height=height,
        original_width=width,
        pad_left=0,
        pad_right=pad_w,
        pad_top=0,
        pad_bottom=pad_h,
    )

    if pad_h == 0 and pad_w == 0:
        return x, pad_info

    selected_mode = mode

    # Reflection padding requires the pad size to be smaller
    # than the corresponding input dimension.
    if pad_h >= height or pad_w >= width or height <= 1 or width <= 1:
        selected_mode = "replicate"

    x = F.pad(
        x,
        (
            pad_info.pad_left,
            pad_info.pad_right,
            pad_info.pad_top,
            pad_info.pad_bottom,
        ),
        mode=selected_mode,
    )

    return x, pad_info
```

---

## 8.4 Cropping implementation

```python
def crop_to_original(x: torch.Tensor, pad_info: PadInfo) -> torch.Tensor:
    return x[:, :, : pad_info.original_height, : pad_info.original_width]
```

Do not use interpolation to recover the original size.

---

## 8.5 Positional information

Do not use learned absolute positional embeddings.

Use:

- depthwise convolution in QKV generation;
- local relative offsets;
- relative bias tables indexed by neighborhood offsets;
- dynamically sized stripe attention.

For local 3x3 attention, the relative offset set is fixed: {-1,0,1} x {-1,0,1}. For dilated
attention, multiply offsets by dilation. This remains valid for arbitrary image sizes.

---

## 8.6 Arbitrary-size tests

```python
@pytest.mark.parametrize(
    "height,width",
    [(1, 1), (7, 9), (31, 47), (63, 65), (127, 193), (482, 512), (513, 769)],
)
def test_arbitrary_sizes(height, width):
    model = CASHSI(config).eval()

    x = torch.randn(1, 3, height, width)

    with torch.no_grad():
        y = model(x)

    assert y.shape == (1, config.output_bands, height, width)
```

---

## 8.7 Tiled inference for large images

Arbitrary-size support does not remove GPU-memory limits. Implement optional overlapping
tile inference.

Recommended defaults:

```yaml
inference:
  tile_size: 256
  overlap: 32
  blend: hann
```

For tile T_i and blending window W_i:

    Yhat = sum_i (W_i (*) Yhat_i) / (sum_i W_i + eps)

Use reflection padding around outer image borders.

```python
prediction = tiled_inference(
    model=model,
    rgb=image,
    tile_size=256,
    overlap=32,
    blend_mode="hann",
)
```

Validation requirement: direct full-image inference and tiled inference should agree within
a tolerance away from tile boundaries.

---

## 8.8 Shape metadata rules

All modules must infer spatial dimensions dynamically from the input tensor.

Forbidden patterns:

```python
self.height = 128
self.width = 128
x = x.view(batch, channels, 128 * 128)
```

Required pattern:

```python
batch, channels, height, width = x.shape
x = x.reshape(batch, heads, head_dim, height * width)
```

---

# 9. Edge-Deployment Profile

## 9.1 Deployment objectives

The edge profile should optimize:

- wall-clock latency;
- peak activation memory;
- operator compatibility;
- deterministic execution;
- numerical stability under FP16 or INT8;
- arbitrary-resolution support.

FLOP count alone is insufficient.

---

## 9.2 Dual spatial-mixer backend

Provide two interchangeable spatial-mixer modes.

### Research backend

```yaml
spatial_mixer:
  full: depthwise_7x7
  half: dilated_local_attention
  quarter: dilated_local_attention
  periodic_bottleneck: hybrid_local_stripe_attention
```

### Edge backend

```yaml
spatial_mixer:
  full: depthwise_7x7
  half: dilated_depthwise_conv
  quarter: dilated_depthwise_conv
  periodic_bottleneck: large_kernel_depthwise_conv
```

The block shell remains unchanged:

```text
Norm -> spatial mixer -> residual
Norm -> cross-channel attention -> residual
Norm -> gated FFN -> residual
```

This permits direct operator replacement, feature distillation, and deployment
benchmarking without redesigning the model.

---

## 9.3 Edge spatial mixer

Replace local attention with parallel depthwise dilated convolutions.

    U1 = D_{3x3, d=1}(X)
    U2 = D_{3x3, d=2}(X)
    U3 = D_{3x3, d=3}(X)
    S_edge(X) = Conv1x1( [U1, U2, U3] )

To control cost, split channels across branches rather than applying every branch to all
channels.

```python
class MultiDilationDepthwiseMixer(nn.Module):
    def __init__(self, channels, dilations=(1, 2, 3)):
        super().__init__()

        if channels % len(dilations) != 0:
            raise ValueError("channels must be divisible by number of dilations")

        group_channels = channels // len(dilations)

        self.group_channels = group_channels
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(
                    group_channels,
                    group_channels,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                    groups=group_channels,
                    bias=False,
                )
                for dilation in dilations
            ]
        )

        self.output_projection = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        groups = x.split(self.group_channels, dim=1)

        outputs = [branch(group) for branch, group in zip(self.branches, groups)]

        x = torch.cat(outputs, dim=1)
        return self.output_projection(x)
```

---

## 9.4 Attention distillation

Train the edge model from the research model.

    L_pred    = || S(X) - T(X) ||_1
    L_feat    = sum_l || P_l(F_l^S) - F_l^T ||_1     (P_l a 1x1 projection when widths differ)
    L_student = L_ground_truth + lambda_p * L_pred + lambda_f * L_feat

Recommended starting values:

```yaml
distillation:
  prediction_weight: 0.5
  feature_weight: 0.1
  feature_stages:
    - encoder_half
    - bottleneck
    - decoder_half
```

Do not distill only attention maps because the student may not contain attention
operators.

---

## 9.5 Mixed-precision policy

Recommended FP16/BF16 policy:

```text
Convolutions:             FP16 or BF16
QKV projections:          FP16 or BF16
Attention logits:         FP32 accumulation
Softmax:                  FP32
Normalization statistics: FP32
Output projection:        FP16 or BF16
Loss computation:         FP32
```

Reference pattern:

```python
with torch.autocast(device_type="cuda", dtype=torch.float16):
    prediction = model(rgb)

loss = compute_loss(prediction.float(), target.float())
```

Inside attention:

```python
logits = torch.matmul(q.float(), k.float().transpose(-2, -1))
weights = logits.softmax(dim=-1).to(v.dtype)
```

Benchmark whether this precision promotion is still required on the target runtime.

---

## 9.6 INT8 strategy

Prefer quantization-aware training for the convolution-heavy edge model.

Quantize first: pointwise convolutions; depthwise convolutions; RGB prior; spectral output
projection.

Keep initially in FP16: normalization; attention softmax; temperature parameters; residual
additions where scale mismatch is significant.

A practical mixed profile is:

```text
INT8: convolution weights; convolution activations where validated
FP16: normalization; attention; residual accumulation; final prediction
```

Do not assume INT8 improves latency until measured on the actual device.

---

## 9.7 Operator compatibility

Preferred operators:

```text
Conv2d, depthwise Conv2d, LayerNorm/RMSNorm expressed with primitive ops,
PixelShuffle, PixelUnshuffle, reshape, transpose, concatenate,
elementwise multiply, elementwise add, softmax
```

Avoid in the deployment graph:

```text
custom CUDA-only neighborhood attention, deformable attention,
dynamic Python control flow, unbounded tensor lists,
unsupported scatter/gather patterns, fixed-size learned positional embeddings
```

---

## 9.8 ONNX export

Export with dynamic spatial dimensions.

```python
torch.onnx.export(
    model,
    example_input,
    "cas_hsi.onnx",
    input_names=["rgb"],
    output_names=["hsi"],
    dynamic_axes={
        "rgb": {0: "batch", 2: "height", 3: "width"},
        "hsi": {0: "batch", 2: "height", 3: "width"},
    },
    opset_version=18,
)
```

Before export:

1. switch model to eval mode;
2. replace unsupported attention mixers;
3. disable stochastic depth;
4. verify padding implementation;
5. compare PyTorch and ONNX outputs.

---

## 9.9 Export equivalence test

```python
def test_onnx_equivalence():
    model = build_edge_model(config).eval()

    x = torch.randn(1, 3, 127, 193)

    with torch.no_grad():
        torch_output = model(x).cpu().numpy()

    onnx_output = run_onnx("cas_hsi.onnx", {"rgb": x.cpu().numpy()})[0]

    np.testing.assert_allclose(torch_output, onnx_output, rtol=1e-3, atol=1e-4)
```

Use stricter tolerances for FP32 and relaxed tolerances for FP16.

---

## 9.10 Benchmark protocol

Benchmark the actual target resolution, not only 256x256 patches.

Required measurements:

```text
parameter count, MACs, serialized model size, peak CPU/GPU/NPU memory,
mean latency, median latency, P90 latency, P99 latency, throughput,
power consumption where measurable, startup time
```

Benchmark conditions:

```yaml
benchmark:
  warmup_iterations: 50
  measured_iterations: 200
  batch_size: 1
  synchronize_device: true
  input_sizes:
    - [128, 128]
    - [256, 256]
    - [482, 512]
    - [512, 512]
```

Report: PyTorch eager FP32; PyTorch compiled FP16; ONNX Runtime FP32; ONNX Runtime FP16;
TensorRT FP16; TensorRT INT8; target edge runtime.

---

## 9.11 Deployment profiles

### Accuracy profile

```yaml
deployment:
  profile: accuracy
  model: cas_hsi_base
  spatial_attention: enabled
  stripe_attention: enabled
  precision: fp16
  quantization: none
```

### Balanced profile

```yaml
deployment:
  profile: balanced
  model: cas_hsi_tiny
  full_resolution_mixer: depthwise_7x7
  half_resolution_mixer: dilated_depthwise_conv
  quarter_resolution_attention: enabled
  stripe_attention: disabled
  precision: fp16
  quantization: convolution_only_int8
```

### Edge profile

```yaml
deployment:
  profile: edge
  model: cas_hsi_tiny
  spatial_attention: replaced
  spatial_mixer: multi_dilation_depthwise
  cross_channel_attention:
    enabled: true
    precision: fp16
  precision: mixed_int8_fp16
  stripe_attention: disabled
  tiled_inference: optional
```

---

## 9.12 Edge acceptance criteria

The deployment model is acceptable only if:

- exported inference matches PyTorch within the chosen tolerance;
- arbitrary-size inputs work;
- no unsupported operators remain;
- latency is measured on the target device;
- memory use is below the deployment budget;
- quality degradation relative to the research model is quantified;
- INT8 spectral metrics remain acceptable;
- tile seams are absent or below a defined numerical threshold.

Recommended maximum quality loss relative to the research model:

```text
MRAE increase: <= 3%
PSNR decrease: <= 0.3 dB
SAM increase:  <= 3%
```

These are initial engineering thresholds and should be adapted to the application.

---

# Implementation Order

## Phase 1 — Stable convolutional baseline

1. Implement padding and cropping.
2. Implement LayerNorm2d.
3. Implement LayerScale.
4. Implement PixelUnshuffle downsampling.
5. Implement PixelShuffle upsampling.
6. Implement CAS-Lite.
7. Build the full encoder-decoder using CAS-Lite everywhere.
8. Add skip connections.
9. Add RGB prior and residual spectral head.
10. Verify arbitrary-size inference.

## Phase 2 — Cross-channel attention

1. Implement CrossChannelAttention.
2. Insert it into CAS-Lite.
3. Validate shape and gradient behavior.
4. Compare against pooled channel attention.
5. Profile memory and latency.

## Phase 3 — Dilated local attention

1. Implement the unfold-based reference operator.
2. Add dilation groups.
3. Add relative position bias.
4. Replace only the H/4 spatial mixer.
5. Validate correctness.
6. Replace the H/2 mixer only after the bottleneck version is stable.

## Phase 4 — Stripe attention

1. Implement horizontal stripes.
2. Implement vertical stripes.
3. Support dynamic stripe padding.
4. Add one hybrid block every third bottleneck block.
5. Measure whether it improves quality enough to justify latency.

## Phase 5 — Deployment backend

1. Implement MultiDilationDepthwiseMixer.
2. Add a module-replacement utility.
3. Export the convolutional edge model.
4. Distill from the attention teacher.
5. Apply quantization-aware training.
6. Benchmark on the target runtime.

---

# Definition of Done

The implementation is complete when all of the following hold:

- the model accepts arbitrary H x W RGB tensors;
- the output exactly matches the original spatial dimensions;
- output contains the configured number of spectral bands;
- the model has no BatchNorm layers;
- every CAS residual branch uses pre-normalization;
- identity paths are unscaled;
- LayerScale initializes residual branches near zero;
- latent features remain wider than 31 channels until the final head;
- encoder-decoder skip connections exist at both resolutions;
- the RGB prior is projected explicitly from 3 to 31 bands;
- no output activation is applied after the final spectral projection;
- full-resolution blocks use CAS-Lite by default;
- true spatial attention is restricted to lower resolutions;
- the edge backend can replace unsupported attention operators;
- ONNX export supports dynamic height and width;
- automated shape, gradient, export, and numerical-equivalence tests pass.
