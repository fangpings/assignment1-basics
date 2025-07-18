import torch.nn as nn
import torch
from torch import Tensor

from jaxtyping import Float, Bool, Int
from einops import einsum, rearrange

from math import sqrt

from .ffn import FFN
from .attn import Attention
from .rmsnorm import RMSNorm
from .embedding import Embedding
from .linear import Linear

class Block(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        d_ff: int,
        theta: float | None = None, 
        max_seq_len: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super(Block, self).__init__()

        self.attn = Attention(d_model, n_heads, theta=theta, max_seq_len=max_seq_len, device=device, dtype=dtype)
        self.ffn = FFN(d_model, d_ff, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
    
    def forward(self, x: Float[Tensor, " ... seq_len d_model"], token_positions: Float[Tensor, " ... seq_len"] | None = None) -> Float[Tensor, " ... seq_len d_model"]:
        x = x + self.attn(self.ln1(x), token_positions=token_positions)
        return x + self.ffn(self.ln2(x))
    
class Transformer(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        d_model: int, 
        n_heads: int,
        d_ff: int,
        num_layers: int,
        theta: float | None = None, 
        max_seq_len: int | None = None,
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ):
        super(Transformer, self).__init__()
        
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            Block(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                theta=theta,
                max_seq_len=max_seq_len,
                device=device,
                dtype=dtype
            )
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: Int[Tensor, "batch_size sequence_length"]) -> Int[Tensor, "batch_size sequence_length"]:
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.lm_head(x)