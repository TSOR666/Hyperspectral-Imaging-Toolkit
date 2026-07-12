import math
from typing import Optional

import torch


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int, max_period: int = 10000,
                           dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Create sinusoidal timestep embeddings as in DDPM/Transformer literature."""
    if dtype is None:
        dtype = timesteps.dtype

    half_dim = embedding_dim // 2
    # Frequencies must be computed in fp32: under bf16/fp16 adjacent embedding
    # channels collapse to identical frequencies. Cast to `dtype` only at return.
    exponent = -math.log(max_period) * torch.arange(
        half_dim, device=timesteps.device, dtype=torch.float32
    ) / half_dim
    exponent = exponent.unsqueeze(0)

    timesteps = timesteps.float().unsqueeze(1)
    sinusoid_input = timesteps * torch.exp(exponent)
    emb = torch.cat([torch.sin(sinusoid_input), torch.cos(sinusoid_input)], dim=1)

    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))

    if dtype.is_floating_point and emb.dtype != dtype:
        emb = emb.to(dtype)

    return emb
