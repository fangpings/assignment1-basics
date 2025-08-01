import numpy.typing as npt
import torch
import os
from typing import IO, Any, BinaryIO

from torch import Tensor

from jaxtyping import Float, Int

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[Int[Tensor, "batch_size context_length"], Int[Tensor, "batch_size context_length"]]:
    """
    dataset is a big 1D numpy array of tokens. To sample B sequences, we just pick B random starting point from
    dataset array, and for each sequence, we pick the next `context_length` tokens.

    Note that this way of sampling will cross <|endoftext|> boundary. This means in one batch, we will see sequence like this
    [..., dx_ty-2, dx_ty-1, <|endoftext|>, dx+1_t0, dx+1_t1] (dx means x-th document, ty means y-th tokens)

    It's actually a trade-off because in this way no padding is required and we don't need to track document boundaries.
    Also  The number of training examples that happen to fall across a document boundary is extremely small compared to the total number of examples,
    so it's statistically insignificant noise. 
    """

    max_start_idx = len(dataset) - context_length - 1 # -1 since we need to consider target offset by 1
    start_indices = torch.randint(low=0, high=max_start_idx + 1, size=(batch_size,))

    # calling from_numpy multiple times is ok, it shares memory with ndarray
    inputs = torch.stack([torch.from_numpy(dataset[i:i+context_length]) for i in start_indices])
    targets = torch.stack([torch.from_numpy(dataset[i+1:i+context_length+1]) for i in start_indices])

    return inputs.to(device), targets.to(device)

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    with open(out, "wb") as f:
        torch.save(state, f)

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    with open(src, "rb") as f:
        state = torch.load(f)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    iteration = state["iteration"]
    return iteration