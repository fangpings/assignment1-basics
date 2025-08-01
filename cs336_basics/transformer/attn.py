import torch.nn as nn
import torch
from torch import Tensor

from jaxtyping import Float, Bool
from einops import einsum, rearrange

from math import sqrt

from .linear import Linear
from .rope import RoPE

def softmax(x: Float[Tensor, " ..."], dim: int = -1) -> Float[Tensor, " ..."]:
    x = x - x.max(dim=dim, keepdim=True)[0]
    return torch.exp(x) / torch.sum(torch.exp(x), dim=dim, keepdim=True)

def scaled_dot_product_attention(
    q: Float[Tensor, " ... seq_len d"],
    k: Float[Tensor, " ... seq_len d"],
    v: Float[Tensor, " ... seq_len d_v"],
    mask: Bool[Tensor, " seq_len seq_len"] | None = None,
) -> Float[Tensor, " ... seq_len d_v"]:
    d_k = k.shape[-1]
    scaled_product = einsum(q, k, "... s1 d, ... s2 d -> ... s1 s2") / sqrt(d_k) # cannot have duplicate axis names, so a little bit weird here

    if mask is not None:
        scaled_product.masked_fill_(mask == False, float("-inf"))

    scaled_product = softmax(scaled_product)
    o = einsum(scaled_product, v, "... s1 s2, ... s2 d_v -> ... s1 d_v")
    
    return o

class Attention(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        theta: float | None = None, 
        max_seq_len: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super(Attention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_heads = d_model // num_heads

        self.attn_proj = Linear(d_model, 3 * d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        if theta is not None and max_seq_len is not None:
            self.rope = RoPE(theta=theta, d_k=self.d_heads, max_seq_len=max_seq_len)
        else:
            self.rope = None
    
    def forward(self, x: Float[Tensor, " ... seq_len d_model"], token_positions: Float[Tensor, " ... seq_len"] | None = None) -> Float[Tensor, " ... seq_len d_model"]:
        q, k, v = self.attn_proj(x).split(self.d_model, dim=-1)
        q = rearrange(q, '... seq_len (h d_v) -> ... h seq_len d_v', h=self.num_heads)
        k = rearrange(k, '... seq_len (h d_v) -> ... h seq_len d_v', h=self.num_heads)
        v = rearrange(v, '... seq_len (h d_v) -> ... h seq_len d_v', h=self.num_heads)

        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len)) == 1
        mask = mask.to(x.device)

        if token_positions is None:
            token_positions = torch.arange(seq_len)
        if self.rope:
            q, k = self.rope(q, token_positions), self.rope(k, token_positions)
        
        o = scaled_dot_product_attention(q, k, v, mask)
        o = rearrange(o, '... h seq_len d_v -> ... seq_len (h d_v)')
        o = self.output_proj(o)

        return o