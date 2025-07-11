import torch.nn as nn
import torch
from torch import Tensor

from jaxtyping import Float, Int
from einops import einsum

class Embedding(nn.Module):
    def __init__(self, 
        vocab_size: int, 
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super(Embedding, self).__init__()
        w = torch.empty(vocab_size, embedding_dim, device=device, dtype=dtype)
        nn.init.trunc_normal_(w)
        self.w = nn.Parameter(w)
    
    def forward(self, token_ids: Int[Tensor, '...']) -> Float[Tensor, '... d_model']:
        # see https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing
        # magic
        return self.w[token_ids]