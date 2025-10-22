from losses.wavelet_loss import (
    WaveletLoss,
    MultiscaleWaveletLoss,
    CombinedWaveletLoss
)

from losses.frequency_loss import (
    FrequencyDomainLoss,
    FrequencyBandLoss,
    CombinedSpectralLoss
)

__all__ = [
    # Wavelet losses
    'WaveletLoss',
    'MultiscaleWaveletLoss',
    'CombinedWaveletLoss',
    
    # Frequency losses
    'FrequencyDomainLoss',
    'FrequencyBandLoss',
    'CombinedSpectralLoss'
]