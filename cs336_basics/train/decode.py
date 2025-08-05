import os
import logging
import time
import argparse

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from ..transformer import Transformer, softmax
from ..bpe.tokenizer import Tokenizer
from . import *
from .config import Config

from tqdm import tqdm

def top_p_sample(
    probs: torch.Tensor,
    top_p: float,
) -> torch.Tensor:
    sorted_probs, index = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, 0)

    # right shift 1 since we need to surpass the topp prob
    mask = cumsum > top_p
    mask[1:] = mask[:-1].clone()
    mask[0] = False

    unselected_index = index[mask]
    probs[unselected_index] = 0
    return torch.multinomial(probs, 1)

@torch.no_grad()
def decode(
    model: Transformer,
    tokens: torch.Tensor, # (seq_length, )
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    batched_tokens = tokens.unsqueeze(0)
    logits = model(batched_tokens) # (1, seq_length, vocab_size)
    last = logits[:, -1, :].squeeze() # (vocab_size, )
    probs = softmax(last, temperature=temperature) # (vocab_size, )

    next_token = top_p_sample(probs, top_p)
    return next_token

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decoding configuration")
    parser.add_argument("--checkpoint_dir", type=str,
                          help="Path to checkpoint dir")
    parser.add_argument("--tokenizer_path", type=str,
                          help="Path to tokenizer pickle")
    parser.add_argument("--prompt", type=str,
                          help="Prompt for decoding")
    parser.add_argument("--max_length", type=int,
                          help="Maximum generated length")
    parser.add_argument("--temperature", type=float,
                          help="Temperature for sampling")
    parser.add_argument("--topp", type=float,
                          help="topp for sampling")
    parser.add_argument("--eos_token", type=str,
                          help="eos token")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_pikcle(args.tokenizer_path)
    eos_token_id = tokenizer.encode(args.eos_token)[0]

    config = Config.from_yaml(os.path.join(args.checkpoint_dir, "config.yaml"))

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = Transformer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_heads=config.num_heads,
        d_ff=config.d_ff,
        num_layers=config.num_layers,
        max_seq_len=config.context_length,
        theta=config.rope_theta
    ).to(device)

    if device == torch.device("mps"):
        model = torch.compile(model, backend="aot_eager")

    checkpoints = os.listdir(args.checkpoint_dir)
    checkpoints.remove("config.yaml")
    checkpoints.sort()
    latest_checkpoint = os.path.join(args.checkpoint_dir, checkpoints[-1])
    load_checkpoint(latest_checkpoint, model)

    tokenized_prompt = tokenizer.encode(args.prompt)
    tokenized_prompt = torch.tensor(tokenized_prompt, device=device)

    generated_response = []
    print(args.prompt, end='', flush=True)
    while True:
        generated_token = decode(model, tokenized_prompt, args.temperature, args.topp)

        generated_token_detached = generated_token.detach().item()
        if generated_token_detached == eos_token_id:
            break
        generated_response.append(generated_token_detached)

        print(tokenizer.decode([generated_token_detached]), end='', flush=True)

        if len(generated_response) == args.max_length:
            break
        tokenized_prompt = torch.concat([tokenized_prompt, generated_token])
