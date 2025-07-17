import torch.nn as nn
import torch
from torch import Tensor

from jaxtyping import Float

from einops import rearrange

class RoPE(nn.Module):
    def __init__(
        self, 
        theta: float,
        d_k: int,
        max_seq_len: int, 
        device: torch.device | None = None, 
    ):
        super(RoPE, self).__init__()
        # Create frequency values for each dimension pair
        freq_base = theta ** (-2 * torch.arange(d_k // 2, device=device, dtype=torch.float32) / d_k)
        # Create position indices  
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        # Compute frequencies for each position and dimension pair
        frequencies = torch.outer(positions, freq_base)
        
        self.register_buffer("freq_sin", torch.sin(frequencies), persistent=False) # (max_seq_len, dk // 2)
        self.register_buffer("freq_cos", torch.cos(frequencies), persistent=False)
    
    def forward(self, x: Float[Tensor, " ... seq_len d_k"], token_positions: Float[Tensor, " ... seq_len"]) -> Float[Tensor, " ... seq_len d_k"]:
        # '... (h d) -> ... h d' means:
        # - Keep all leading dimensions (...)
        # - Split the last dimension into two new dimensions, 'h' and 'd'
        # - The size of 'd' is explicitly provided as 2
        x = rearrange(x, '... (h d) -> ... h d', d=2) # (..., seq_len, d_k //2, 2)

        freq_sin = self.freq_sin[token_positions] # (..., seq_len, dk // 2)
        freq_cos = self.freq_cos[token_positions] # (..., seq_len, dk // 2)

        x1 = x[..., 0] * freq_cos - x[..., 1] * freq_sin
        x2 = x[..., 1] * freq_cos + x[..., 0] * freq_sin

        x = torch.stack([x1, x2], dim=-1)

        return rearrange(x, '... h d -> ... (h d)') 


