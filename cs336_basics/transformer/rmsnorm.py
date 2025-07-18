import torch.nn as nn
import torch
from torch import Tensor

from jaxtyping import Float

class RMSNorm(nn.Module):
    def __init__(
        self, 
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        in_dtype = x.dtype 
        x = x.to(torch.float32)
        rms = torch.sqrt((x ** 2).sum(dim=-1, keepdim=True) / self.d_model + self.eps)
        x = x * self.weight / rms
        return x.to(in_dtype)