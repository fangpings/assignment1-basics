import os
import logging
import time
import argparse

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from ..transformer import Transformer
from . import *
from .config import Config

from tqdm import tqdm

logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def train(config: Config):
    train_data = np.load(config.train_path, mmap_mode='r')
    validation_data = np.load(config.validation_path, mmap_mode='r')
    
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

    optimizer = AdamW(
        params=model.parameters(),
        lr=config.max_learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay
    )

    current_iter = -1
    if not os.path.isdir(config.checkpoint_dir) or not os.listdir(config.checkpoint_dir):
        # if checkpoint dir does not exist or have no content
        os.makedirs(config.checkpoint_dir, exist_ok=True)
    else:
        # else read from latest checkpoint
        checkpoints = os.listdir(config.checkpoint_dir)
        checkpoints.sort()
        latest_checkpoint = os.path.join(config.checkpoint_dir, checkpoints[-1])
        current_iter = load_checkpoint(latest_checkpoint, model, optimizer)
        logging.info(f"Resuming from checkpoint at step {current_iter}")

    config.log()
    writer = SummaryWriter(os.path.join(config.log_dir, f"{int(time.time())}"))

    # sample_inputs, _ = get_batch(train_data, config.batch_size, config.context_length, device=device)
    # writer.add_graph(model, sample_inputs)

    train_iter = tqdm(
        range(current_iter+1, config.max_iteration), 
        desc="Training", 
        initial=current_iter+1, 
        total=config.max_iteration
    )
    for it in train_iter:
        optimizer.zero_grad()
        batch, target = get_batch(train_data, config.batch_size, config.context_length, device=device)
        logits = model(batch) # (batch_size, sequence_length, vocab_size)

        logits = logits.view(-1, logits.shape[-1])
        target = target.view(-1)
        loss = cross_entropy_loss(logits, target)

        loss.backward()
        gradient_clipping(model.parameters(), config.max_l2_norm)
        
        lr = lr_cosine_schedule(
            it,
            config.max_learning_rate,
            config.min_learning_rate,
            config.warmup_iters,
            config.cosine_cycle_iters
        )
        optimizer.param_groups[0]['lr'] = lr # this is how lr scheduler works
        optimizer.step()

        writer.add_scalar('Loss/train', loss.item(), it)
        train_iter.set_description(f"Training - Loss {loss.item():04f}")

        if it > 0 and it % config.checkpoint_steps == 0:
            save_dir = os.path.join(config.checkpoint_dir, f"step_{it:04d}.bin")
            save_checkpoint(model, optimizer, it, save_dir)
    
    writer.close()
    save_dir = os.path.join(config.checkpoint_dir, f"step_{it:04d}.bin")
    save_checkpoint(model, optimizer, it, save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training configuration")
    parser.add_argument("--config", type=str,
                          help="Path to YAML config file")
    
    args = parser.parse_args()
    config = Config.from_yaml(args.config)

    train(config)
