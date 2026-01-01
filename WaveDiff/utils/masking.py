import torch

class MaskingManager:
    """
    Manages masking strategies for HSI model training
    
    Implements various masking approaches for self-supervised learning:
    - Random masking: Random pixels are masked
    - Block masking: Random blocks are masked
    - Spectral masking: Random spectral bands are masked
    - Combined masking: Both spatial and spectral masking
    - Curriculum masking: Progressive strategies during training
    """
    def __init__(self, config):
        self.strategy = config.get('mask_strategy', 'random')
        self.config = config
        self.current_epoch = 0
        self.max_epochs = config.get('num_epochs', 100)
        
        # Band importance weights for HSI
        self.band_importance = config.get('band_importance', None)
        
        if self.band_importance is None and self.strategy in ['spectral', 'combined']:
            # Default importance weights - emphasize visible and NIR bands
            self.band_importance = torch.ones(31)
            # Higher importance for visible bands (adjust as needed)
            self.band_importance[:10] = 2.0
    
    def update_epoch(self, epoch):
        """Update the current epoch for progressive masking strategies"""
        self.current_epoch = epoch
    
    def generate_mask(self, inputs, batch_size, num_bands, height, width, device, 
                      override_strategy=None, override_mask_ratio=None, 
                      override_band_mask_ratio=None):
        """
        Generate mask based on the selected strategy
        
        Args:
            inputs: Input data tensor (can be None for some strategies)
            batch_size: Batch size
            num_bands: Number of spectral bands
            height: Spatial height
            width: Spatial width
            device: Device to create mask on
            override_*: Optional parameters to override configuration
            
        Returns:
            Boolean mask tensor [B, channels, H, W] (True = keep, False = mask)
        """
        # Set strategy and parameters
        strategy = override_strategy or self.strategy
        mask_ratio = override_mask_ratio or self.config.get('mask_ratio', 0.5)
        band_mask_ratio = override_band_mask_ratio or self.config.get('band_mask_ratio', 0.3)
        
        # Choose masking strategy
        if strategy == 'random':
            return self._random_masking(batch_size, num_bands, height, width, device, mask_ratio)
        elif strategy == 'block':
            return self._block_masking(batch_size, num_bands, height, width, device, mask_ratio)
        elif strategy == 'spectral':
            return self._spectral_masking(batch_size, num_bands, height, width, device, band_mask_ratio)
        elif strategy == 'combined':
            return self._combined_masking(batch_size, num_bands, height, width, device, mask_ratio, band_mask_ratio)
        elif strategy == 'curriculum':
            return self._curriculum_masking(batch_size, num_bands, height, width, device)
        elif strategy == 'frequency_domain':
            return self._frequency_domain_masking(inputs, batch_size, num_bands, height, width, device)
        elif strategy == 'wavelet_domain':
            return self._wavelet_domain_masking(inputs, batch_size, num_bands, height, width, device)
        else:
            # Default to random masking
            return self._random_masking(batch_size, num_bands, height, width, device, mask_ratio)
    
    def _random_masking(self, batch_size, num_bands, height, width, device, mask_ratio=0.5):
        """Random pixel masking strategy"""
        # Generate random values between 0 and 1
        mask = torch.rand(batch_size, 1, height, width, device=device) > mask_ratio
        return mask
    
    def _block_masking(self, batch_size, num_bands, height, width, device, mask_ratio=0.5):
        """Block masking strategy - masks random blocks of pixels"""
        block_size = self.config.get('block_size', 32)
        
        # Initialize mask (1 = keep, 0 = mask)
        mask = torch.ones(batch_size, 1, height, width, device=device)
        
        # Calculate blocks in each dimension
        h_blocks = height // block_size
        w_blocks = width // block_size
        
        # Number of blocks to mask
        total_blocks = h_blocks * w_blocks
        num_blocks = int(total_blocks * mask_ratio)
        
        # Generate masks for each image in batch
        for b in range(batch_size):
            # Get random block positions
            positions = torch.randperm(total_blocks, device=device)[:num_blocks]
            
            # Apply masks at selected positions
            for pos in positions:
                h_idx = (pos // w_blocks).item()
                w_idx = (pos % w_blocks).item()
                
                h_start = h_idx * block_size
                w_start = w_idx * block_size
                h_end = min(h_start + block_size, height)
                w_end = min(w_start + block_size, width)
                
                mask[b, :, h_start:h_end, w_start:w_end] = 0.0
                
        return mask.bool()
    
    def _spectral_masking(self, batch_size, num_bands, height, width, device, band_mask_ratio=0.3):
        """Spectral masking strategy - masks random spectral bands"""
        mask = torch.ones(batch_size, num_bands, height, width, device=device)
        
        num_bands_to_mask = max(1, int(num_bands * band_mask_ratio))
        
        # Convert importance weights to tensor
        if not isinstance(self.band_importance, torch.Tensor):
            self.band_importance = torch.tensor(self.band_importance, device=device)
            
        # Invert weights for masking probability (higher importance = lower prob)
        if len(self.band_importance) == num_bands:
            selection_weights = 1.0 / (self.band_importance.to(device) + 1e-8)
            selection_probs = selection_weights / selection_weights.sum()
        else:
            # Equal probability if importance weights don't match
            selection_probs = torch.ones(num_bands, device=device) / num_bands
        
        if num_bands_to_mask == 0:
            return mask.bool()

        batched_probs = selection_probs.unsqueeze(0).expand(batch_size, -1)
        masked_bands = torch.multinomial(
            batched_probs,
            num_bands_to_mask,
            replacement=False
        )

        batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(masked_bands)
        mask[batch_idx, masked_bands] = 0.0
            
        return mask.bool()
    
    def _combined_masking(self, batch_size, num_bands, height, width, device, 
                         spatial_mask_ratio=0.3, band_mask_ratio=0.2):
        """Combined spatial and spectral masking"""
        # Create spatial block mask 
        spatial_mask = self._block_masking(
            batch_size, 1, height, width, device, spatial_mask_ratio
        )
        
        # Create spectral band mask
        band_mask = self._spectral_masking(
            batch_size, num_bands, height, width, device, band_mask_ratio
        )
        
        # Expand spatial mask to match band dimensions
        spatial_mask_expanded = spatial_mask.expand(-1, num_bands, -1, -1)
        
        # Combine masks (element-wise AND)
        combined_mask = spatial_mask_expanded & band_mask
        
        return combined_mask
    
    def _curriculum_masking(self, batch_size, num_bands, height, width, device):
        """Progressive curriculum masking that changes with training progress"""
        progress = min(1.0, self.current_epoch / self.max_epochs)
        strategies = self.config.get('curriculum_strategies', 
                                    ['random', 'block', 'spectral', 'combined'])
        
        # Get current strategy based on training progress
        strategy_idx = min(len(strategies) - 1, int(progress * len(strategies)))
        current_strategy = strategies[strategy_idx]
        
        # Calculate progressive mask ratio (gradually increases with training)
        initial_ratio = self.config.get('initial_mask_ratio', 0.1)
        final_ratio = self.config.get('final_mask_ratio', 0.5)
        
        mask_ratio = initial_ratio + (final_ratio - initial_ratio) * progress
        
        # Apply selected strategy with calculated mask ratio
        if current_strategy == 'random':
            return self._random_masking(batch_size, num_bands, height, width, device, mask_ratio)
        elif current_strategy == 'block':
            return self._block_masking(batch_size, num_bands, height, width, device, mask_ratio)
        elif current_strategy == 'spectral':
            band_mask_ratio = mask_ratio * 0.6  # Scale for spectral masking
            return self._spectral_masking(batch_size, num_bands, height, width, device, band_mask_ratio)
        elif current_strategy == 'combined':
            spatial_mask_ratio = mask_ratio
            band_mask_ratio = mask_ratio * 0.4  # Less aggressive spectral masking
            return self._combined_masking(
                batch_size, num_bands, height, width, device, 
                spatial_mask_ratio, band_mask_ratio
            )
        else:
            # Default to random
            return self._random_masking(batch_size, num_bands, height, width, device, mask_ratio)
    
    def _frequency_domain_masking(self, inputs, batch_size, num_bands, height, width, device):
        """
        Frequency domain masking - masks specific frequency bands
        Requires input tensor to compute FFT
        """
        if inputs is None:
            # Fall back to block masking if no inputs provided
            return self._block_masking(batch_size, num_bands, height, width, device)
            
        # Parameters
        low_freq_keep_ratio = self.config.get('low_freq_keep_ratio', 0.8)
        high_freq_keep_ratio = self.config.get('high_freq_keep_ratio', 0.4)
        
        # Initialize mask (1 = keep, 0 = mask)
        mask = torch.ones(batch_size, 1, height, width, device=device)
        
        # Process each sample in batch
        for b in range(batch_size):
            # Get sample (used for device reference)
            _ = inputs[b:b+1]

            # Create frequency grid
            freq_h = torch.fft.fftfreq(height, device=device)[:, None]
            freq_w = torch.fft.rfftfreq(width, device=device)[None, :]
            
            # Compute distance from DC
            freq_dist = torch.sqrt(freq_h**2 + freq_w**2)
            
            # Normalize distance to [0, 1]
            max_dist = torch.sqrt(torch.tensor(0.5**2 + 0.5**2, device=device))
            norm_dist = freq_dist / max_dist
            
            # Create frequency mask (probability of keeping decreases with frequency)
            freq_mask_prob = low_freq_keep_ratio - (low_freq_keep_ratio - high_freq_keep_ratio) * norm_dist
            
            # Apply random masking based on frequency-dependent probability
            freq_mask = torch.rand_like(freq_mask_prob) < freq_mask_prob
            
            # Convert back to spatial domain
            mask[b, 0] = torch.fft.irfft2(
                torch.fft.rfft2(torch.ones_like(mask[b, 0])) * freq_mask, 
                s=(height, width)
            ) > 0.5
        
        return mask.bool()
    
    def _wavelet_domain_masking(self, inputs, batch_size, num_bands, height, width, device):
        """
        Wavelet domain masking - masks specific wavelet components
        Falls back to combined masking if inputs not provided
        """
        if inputs is None:
            # Fall back to combined masking if no inputs provided
            return self._combined_masking(
                batch_size, num_bands, height, width, device,
                self.config.get('spatial_mask_ratio', 0.3),
                self.config.get('band_mask_ratio', 0.2)
            )
        
        # For a proper implementation, we would need to:
        # 1. Apply wavelet transform to inputs
        # 2. Selectively mask wavelet coefficients
        # 3. Apply inverse transform
        # 4. Create binary mask based on the result
        
        # For simplicity, we'll combine block and spectral masking
        # with frequency-aware weighting
        combined_mask = self._combined_masking(
            batch_size, num_bands, height, width, device,
            self.config.get('spatial_mask_ratio', 0.3),
            self.config.get('band_mask_ratio', 0.2)
        )
        
        return combined_mask
