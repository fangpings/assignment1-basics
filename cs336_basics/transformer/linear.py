import torch.nn as nn
import torch
from torch import Tensor

from jaxtyping import Float
from einops import einsum

from math import sqrt

class Linear(nn.Module):
    def __init__(
        self, 
        in_features: int, 
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super(Linear, self).__init__()
        w = torch.empty(out_features, in_features, device=device, dtype=dtype)

        std = sqrt(2/(out_features+in_features))
        nn.init.trunc_normal_(w, std=std, a=-3*std, b=3*std)
        self.weight = nn.Parameter(w)
    
    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")