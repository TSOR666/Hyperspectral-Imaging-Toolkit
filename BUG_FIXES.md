# Bug Fixes for Critical and High-Priority Issues

This document contains detailed fixes for all Critical and High priority bugs identified in the model audit.

---

## 🔴 CRITICAL PRIORITY FIXES

### 1. HSIFusionNet: Replace assert with proper validation

**File:** `HSIFUSION&SHARP/hsifusion_v252_complete.py`
**Lines:** 281-285
**Issue:** Assert statements are optimized away with Python -O flag

**BEFORE:**
```python
# Ensure indices are within bounds
max_index = (2 * self.window_size - 1) * (2 * self.window_size - 1) - 1
assert relative_position_index.max() <= max_index, \
    f"Index {relative_position_index.max()} exceeds max {max_index}"
assert relative_position_index.min() >= 0, \
    f"Negative index {relative_position_index.min()}"

self.register_buffer("relative_position_index", relative_position_index)
```

**AFTER:**
```python
# Ensure indices are within bounds
max_index = (2 * self.window_size - 1) * (2 * self.window_size - 1) - 1
max_idx_value = relative_position_index.max().item()
min_idx_value = relative_position_index.min().item()

if max_idx_value > max_index:
    raise ValueError(
        f"Relative position index out of bounds: max index {max_idx_value} "
        f"exceeds maximum allowed {max_index}. This indicates a bug in "
        f"relative position computation for window_size={self.window_size}"
    )

if min_idx_value < 0:
    raise ValueError(
        f"Relative position index out of bounds: negative index {min_idx_value} "
        f"found. This indicates a bug in relative position computation "
        f"for window_size={self.window_size}"
    )

self.register_buffer("relative_position_index", relative_position_index)
```

**Why this matters:**
- Assert statements disappear when running Python with optimization (`python -O`)
- CUDA index errors are cryptic and hard to debug
- Proper ValueError provides clear error message and prevents silent failures

---

### 2. SHARP: Replace assert in sparse attention

**File:** `HSIFUSION&SHARP/sharp_v322_hardened.py`
**Line:** 467
**Issue:** Assert statement for sparsity_ratio validation

**BEFORE:**
```python
def __init__(self, dim: int, num_heads: int = 8,
             sparsity_ratio: float = 0.9, use_topk: bool = True,
             ...):
    super().__init__()
    assert 0.0 <= sparsity_ratio <= 1.0, "sparsity_ratio must be in [0,1]"
```

**AFTER:**
```python
def __init__(self, dim: int, num_heads: int = 8,
             sparsity_ratio: float = 0.9, use_topk: bool = True,
             ...):
    super().__init__()

    if not (0.0 <= sparsity_ratio <= 1.0):
        raise ValueError(
            f"sparsity_ratio must be in range [0.0, 1.0], got {sparsity_ratio}. "
            f"Use 0.0 for dense attention, 1.0 for maximum sparsity."
        )
```

**Additional validation to add:**
```python
# Add these validations as well
if dim <= 0:
    raise ValueError(f"dim must be positive, got {dim}")

if num_heads <= 0:
    raise ValueError(f"num_heads must be positive, got {num_heads}")

if dim % num_heads != 0:
    raise ValueError(
        f"dim ({dim}) must be divisible by num_heads ({num_heads}). "
        f"Consider using dim={dim - (dim % num_heads) + num_heads}"
    )
```

---

## 🟠 HIGH PRIORITY FIXES

### 3. MSWR-Net: Fix incorrect variable naming in landmark attention

**File:** `mswr_v2/model/mswr_net_v212.py`
**Line:** 684
**Issue:** Variable names in rearrange don't match actual dimensions

**BEFORE:**
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape

    # Generate queries efficiently
    q = self.q_conv(x)
    q = rearrange(q, 'b (h d) h_dim w_dim -> b h (h_dim w_dim) d', h=self.num_heads)
```

**AFTER:**
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape

    # Generate queries efficiently
    q = self.q_conv(x)
    # Reshape: (B, C, H, W) -> (B, num_heads, H*W, head_dim)
    q = rearrange(q, 'b (h d) H W -> b h (H W) d', h=self.num_heads, H=H, W=W)
```

**Why this matters:**
- Einops uses variable names to match dimensions
- Using `h_dim`/`w_dim` when you mean `H`/`W` is confusing and error-prone
- Explicit dimension binding prevents subtle shape bugs

---

### 4. MSWR-Net: Fix checkpoint wrapper fallback

**File:** `mswr_v2/model/mswr_net_v212.py`
**Lines:** 1199-1206, 1253-1260
**Issue:** Incorrect use of functools.partial for gradient checkpointing

**BEFORE:**
```python
# Intelligent gradient checkpointing
if config.use_checkpoint and i in (config.checkpoint_blocks or []):
    if hasattr(checkpoint, 'checkpoint_wrapper'):
        block = checkpoint.checkpoint_wrapper(block)
    else:
        # Fallback for older PyTorch versions
        original_forward = block.forward
        block.forward = partial(checkpoint.checkpoint, original_forward, use_reentrant=False)
```

**AFTER:**
```python
# Intelligent gradient checkpointing
if config.use_checkpoint and i in (config.checkpoint_blocks or []):
    if hasattr(checkpoint, 'checkpoint_wrapper'):
        # Modern PyTorch (1.11+)
        block = checkpoint.checkpoint_wrapper(block)
    else:
        # Fallback for older PyTorch versions
        # Create a proper wrapper that captures the original forward method
        original_forward = block.forward

        def make_checkpointed_forward(orig_fwd):
            """Factory to avoid late binding issues"""
            def checkpointed_forward(*args, **kwargs):
                # Wrapper that properly passes arguments to checkpoint
                def forward_fn(*args_inner, **kwargs_inner):
                    return orig_fwd(*args_inner, **kwargs_inner)

                return checkpoint.checkpoint(
                    forward_fn,
                    *args,
                    use_reentrant=False,
                    **kwargs
                )
            return checkpointed_forward

        block.forward = make_checkpointed_forward(original_forward)
```

**Why this matters:**
- The original code using `partial` creates incorrect function signatures
- Gradients won't flow properly through checkpointed blocks
- This completely breaks memory-efficient training on older PyTorch versions

**Alternative simpler fix** (if you don't need kwargs):
```python
original_forward = block.forward

def checkpointed_forward(x):
    return checkpoint.checkpoint(original_forward, x, use_reentrant=False)

block.forward = checkpointed_forward
```

---

### 5. MSWR-Net: Fix cache management race condition

**File:** `mswr_v2/model/mswr_net_v212.py`
**Lines:** 120-129
**Issue:** Potential KeyError when deleting from access count dict

**BEFORE:**
```python
def _manage_cache_memory(self):
    """Intelligent cache management to prevent memory bloat"""
    if len(self._filter_cache) > self._cache_size_limit:
        # Remove least recently used filters
        sorted_keys = sorted(self._cache_access_count.keys(),
                           key=lambda k: self._cache_access_count[k])
        for key in sorted_keys[:len(self._filter_cache) - self._cache_size_limit]:
            if key in self._filter_cache:
                del self._filter_cache[key]
                del self._cache_access_count[key]  # Could fail if key missing
```

**AFTER:**
```python
def _manage_cache_memory(self):
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
```

**Additional improvement** - Add lock for thread safety:
```python
import threading

class OptimizedCNNWaveletTransform(nn.Module):
    def __init__(self, J: int = 1, wave: str = 'db1', mode: str = 'periodic'):
        super().__init__()
        # ... existing code ...

        # Thread-safe cache management
        self._cache_lock = threading.Lock()

    def _manage_cache_memory(self):
        """Thread-safe cache management"""
        with self._cache_lock:
            # ... cache management code ...

    def _get_conv_filters(self, channels: int, device: torch.device, dtype: torch.dtype):
        """Thread-safe filter retrieval"""
        cache_key = f"{channels}_{self.wave}_{device}_{dtype}"

        with self._cache_lock:
            if cache_key in self._filter_cache:
                self._cache_access_count[cache_key] += 1
                return self._filter_cache[cache_key]

        # Create filters outside lock (expensive operation)
        filters = self._create_filters(channels, device, dtype)

        with self._cache_lock:
            self._manage_cache_memory()
            self._filter_cache[cache_key] = filters
            self._cache_access_count[cache_key] = 1

        return filters
```

---

### 6. CSWIN v2: Fix inefficient clone in NaNSafeAttention

**File:** `CSWIN v2/src/hsi_model/models/generator_v3.py`
**Lines:** 45-57
**Issue:** Unconditional clone on every forward pass

**BEFORE:**
```python
class NaNSafeAttention(nn.Module):
    """Wrapper for attention modules with NaN protection."""
    def __init__(self, attention_module):
        super().__init__()
        self.attention = attention_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store input for potential recovery
        x_input = x.clone()  # ← Always clones!

        # Forward through attention
        out = self.attention(x)

        # Check for NaN/Inf
        if torch.isnan(out).any() or torch.isinf(out).any():
            # Fallback to input (skip connection)
            return x_input

        return out
```

**AFTER (Option 1 - No clone, use original x):**
```python
class NaNSafeAttention(nn.Module):
    """Wrapper for attention modules with NaN protection."""
    def __init__(self, attention_module):
        super().__init__()
        self.attention = attention_module
        self.nan_count = 0  # Track NaN occurrences for debugging

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward through attention
        out = self.attention(x)

        # Check for NaN/Inf
        if torch.isnan(out).any() or torch.isinf(out).any():
            # Log this event (rate-limited)
            self.nan_count += 1
            if self.nan_count % 100 == 1:  # Log every 100 occurrences
                import warnings
                warnings.warn(
                    f"NaN/Inf detected in attention output (occurrence {self.nan_count}). "
                    f"Falling back to skip connection."
                )
            # Fallback to input (no clone needed - x hasn't been modified)
            return x

        return out
```

**AFTER (Option 2 - Conditional clone for debugging):**
```python
class NaNSafeAttention(nn.Module):
    """Wrapper for attention modules with NaN protection."""
    def __init__(self, attention_module, debug_mode=False):
        super().__init__()
        self.attention = attention_module
        self.debug_mode = debug_mode
        self.nan_count = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Only clone if in debug mode (for gradient analysis)
        if self.debug_mode:
            x_input = x.clone()

        # Forward through attention
        out = self.attention(x)

        # Check for NaN/Inf
        if torch.isnan(out).any() or torch.isinf(out).any():
            self.nan_count += 1

            if self.debug_mode and self.nan_count % 10 == 1:
                # Detailed debugging info
                print(f"NaN detected! Input stats: mean={x_input.mean():.4f}, "
                      f"std={x_input.std():.4f}, min={x_input.min():.4f}, "
                      f"max={x_input.max():.4f}")

            # Fallback to input
            return x_input if self.debug_mode else x

        return out
```

**Why this matters:**
- `clone()` doubles memory usage for attention inputs
- In a transformer with many attention layers, this adds up quickly
- NaN/Inf is rare after initial training stabilization
- Performance cost is paid on every forward pass even when never needed

---

### 7. HSIFusionNet: Index clamping masks root bug

**File:** `HSIFUSION&SHARP/hsifusion_v252_complete.py`
**Line:** 316
**Issue:** Clamping hides the real problem instead of fixing it

**BEFORE:**
```python
# Get relative position bias with bounds checking
relative_position_bias_flat = self.relative_position_bias_table[
    self.relative_position_index.view(-1).clamp(0, self.relative_position_bias_table.size(0) - 1)
]
```

**AFTER:**
```python
# Get relative position bias with proper error detection
idx_flat = self.relative_position_index.view(-1)
max_valid_idx = self.relative_position_bias_table.size(0) - 1

# Check if clamping would occur (indicates a bug)
if idx_flat.min() < 0 or idx_flat.max() > max_valid_idx:
    # This should never happen if initialization is correct
    import warnings
    warnings.warn(
        f"Relative position index out of bounds detected! "
        f"Index range: [{idx_flat.min()}, {idx_flat.max()}], "
        f"Valid range: [0, {max_valid_idx}]. "
        f"This indicates a bug in _init_relative_position_bias(). "
        f"Clamping as emergency fallback."
    )
    idx_flat = idx_flat.clamp(0, max_valid_idx)

relative_position_bias_flat = self.relative_position_bias_table[idx_flat]
```

**Better fix:** Prevent the issue at the source by fixing `_init_relative_position_bias()`:
```python
def _init_relative_position_bias(self):
    """Initialize relative position bias."""
    self.relative_position_bias_table = nn.Parameter(
        torch.zeros((2 * self.window_size - 1) * (2 * self.window_size - 1), self.num_heads)
    )
    nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    # Compute relative position index
    coords_h = torch.arange(self.window_size)
    coords_w = torch.arange(self.window_size)

    # Handle both old and new PyTorch versions
    try:
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
    except TypeError:
        coords = torch.stack(torch.meshgrid(coords_h, coords_w))

    coords_flatten = torch.flatten(coords, 1)  # (2, window_size^2)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, ws^2, ws^2)
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (ws^2, ws^2, 2)

    # Ensure offset is correct
    relative_coords[:, :, 0] += self.window_size - 1  # Shift x coords
    relative_coords[:, :, 1] += self.window_size - 1  # Shift y coords
    relative_coords[:, :, 0] *= 2 * self.window_size - 1  # Scale x coords

    # Final index computation
    relative_position_index = relative_coords.sum(-1)  # (ws^2, ws^2)

    # CRITICAL: Validate indices before registering
    max_index = (2 * self.window_size - 1) * (2 * self.window_size - 1) - 1
    min_val = relative_position_index.min().item()
    max_val = relative_position_index.max().item()

    if min_val < 0 or max_val > max_index:
        raise RuntimeError(
            f"Bug in relative position index computation! "
            f"Got range [{min_val}, {max_val}], expected [0, {max_index}]. "
            f"window_size={self.window_size}"
        )

    self.register_buffer("relative_position_index", relative_position_index)
```

---

### 8. WaveDiff: Fix incorrect channel inference

**File:** `WaveDiff/models/base_model.py`
**Lines:** 130-137
**Issue:** Accessing potentially non-existent attribute

**BEFORE:**
```python
def generate_mask(self, batch_size, height, width, device, inputs=None, num_channels=None, **kwargs):
    """Generate mask using the configured masking strategy"""
    # Determine number of bands for masking (default to inputs if available)
    if num_channels is None and inputs is not None:
        num_bands = inputs.shape[1]
    else:
        num_bands = num_channels or self.decoder.final_conv.out_channels  # ← May not exist!
```

**AFTER:**
```python
class HSILatentDiffusionModel(nn.Module):
    def __init__(self, latent_dim=64, out_channels=31, timesteps=1000, ...):
        super().__init__()

        # Store out_channels as instance variable for later reference
        self.out_channels = out_channels

        # ... rest of init ...

        self.decoder = HSIDecoder(
            out_channels=out_channels,
            latent_dim=latent_dim,
            use_batchnorm=use_batchnorm
        )

        # ... rest of init ...

def generate_mask(self, batch_size, height, width, device, inputs=None, num_channels=None, **kwargs):
    """Generate mask using the configured masking strategy"""
    # Determine number of bands for masking (default to inputs if available)
    if num_channels is None:
        if inputs is not None:
            num_bands = inputs.shape[1]
        else:
            # Use stored out_channels (safe)
            num_bands = self.out_channels
    else:
        num_bands = num_channels

    # Validate num_bands is reasonable
    if num_bands <= 0 or num_bands > 1000:  # Sanity check
        raise ValueError(f"Invalid num_bands={num_bands}")

    return self.masking_manager.generate_mask(
        inputs,
        batch_size, num_bands, height, width, device,
        **kwargs
    )
```

**Why this matters:**
- Accessing `self.decoder.final_conv.out_channels` is fragile
- Different decoder architectures may not have this attribute
- Can cause AttributeError at runtime
- Storing `out_channels` during init is cleaner and safer

---

### 9. SHARP: Add logging for silent behavior change

**File:** `HSIFUSION&SHARP/sharp_v322_hardened.py`
**Lines:** 150-153
**Issue:** Silent fallback to different algorithm

**BEFORE:**
```python
# Safety cap for very large sequences
if N > max_tokens:
    # Fallback to local window attention (ensure odd window_size)
    window_size = window_size if window_size % 2 == 1 else window_size - 1
    return local_window_attention(q, k, v, window_size, scale)
```

**AFTER:**
```python
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

    window_size = window_size if window_size % 2 == 1 else window_size - 1
    if window_size != self.window_size:
        logger.debug(f"Adjusted window_size from {self.window_size} to {window_size} (must be odd)")

    return local_window_attention(q, k, v, window_size, scale)
```

**Also add initialization warning:**
```python
def __init__(self, dim: int, num_heads: int = 8,
             sparsity_ratio: float = 0.9, use_topk: bool = True,
             ...
             max_tokens: int = 8192,
             ...):
    super().__init__()

    # ... existing validation ...

    # Document max_tokens behavior
    if max_tokens < 1024:
        logger.warning(
            f"max_tokens={max_tokens} is quite low. Sequences longer than this "
            f"will fall back to local window attention, which may not be desired. "
            f"Recommended: max_tokens >= 4096"
        )
```

---

## Summary of Fixes Applied

| Priority | Issue | File | Status |
|----------|-------|------|--------|
| 🔴 Critical | Assert in HSIFusionNet | hsifusion_v252_complete.py | ✅ Fixed |
| 🔴 Critical | Assert in SHARP | sharp_v322_hardened.py | ✅ Fixed |
| 🟠 High | Variable naming in MSWR | mswr_net_v212.py | ✅ Fixed |
| 🟠 High | Checkpoint wrapper bug | mswr_net_v212.py | ✅ Fixed |
| 🟠 High | Cache race condition | mswr_net_v212.py | ✅ Fixed |
| 🟠 High | Inefficient clone | generator_v3.py | ✅ Fixed |
| 🟠 High | Index clamping | hsifusion_v252_complete.py | ✅ Fixed |
| 🟠 High | Channel inference bug | base_model.py | ✅ Fixed |
| 🟠 High | Silent fallback | sharp_v322_hardened.py | ✅ Fixed |

---

## Testing Checklist

After applying these fixes, test:

1. **HSIFusionNet & SHARP:**
   - [ ] Run with Python -O flag (optimization enabled)
   - [ ] Test with various window sizes (even and odd)
   - [ ] Test with extreme input sizes

2. **MSWR-Net:**
   - [ ] Test gradient checkpointing on PyTorch 1.10, 1.11, 1.12, 2.0
   - [ ] Multi-threaded data loading
   - [ ] Different input sizes (small feature maps)

3. **CSWIN v2:**
   - [ ] Memory profiling before/after NaNSafeAttention fix
   - [ ] Long training runs to verify no NaN issues

4. **WaveDiff:**
   - [ ] Test masking with various channel counts
   - [ ] Test with custom decoder architectures

---

*End of fixes document*
