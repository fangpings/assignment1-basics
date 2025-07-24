import torch.nn as nn
import torch
from torch import Tensor

from jaxtyping import Float, Int

def cross_entropy_loss(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"])  -> Float[Tensor, ""]:
    # subtract max value to have numerical stability
    inputs = inputs - inputs.max(dim=-1, keepdim=True)[0]
    # To do indexing on second dim, we need to select all first dim
    # honestly I still don't understand how this indexing thing works...
    #
    # log(exp(x[n, y_n]) / sum_i^vocab(exp(x[n, i]))) = x[n, y_n] - log(sum_i^vocab(exp(x[n, i]))))
    output = inputs[torch.arange(inputs.shape[0]), targets] - torch.log(torch.sum(torch.exp(inputs), dim=1))
    return -output.mean()