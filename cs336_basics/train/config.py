import os
import yaml
import argparse
import logging
from dataclasses import dataclass, fields
from typing import IO, BinaryIO, Dict, Any, Optional, Union


@dataclass
class Config:
    """
    Combined configuration class that includes model, optimizer, and trainer configs.
    Supports loading from YAML files and command-line arguments.
    """
    # Model config fields
    vocab_size: int  # Number of unique items in the output vocabulary
    context_length: int  # Maximum number of tokens to process at once
    d_model: int  # Dimensionality of model embeddings
    num_layers: int  # Number of Transformer layers
    num_heads: int  # Number of heads in multi-headed attention
    d_ff: int  # Dimensionality of feed-forward inner layer
    rope_theta: float  # RoPE theta parameter
    
    # Optimizer config fields
    max_learning_rate: float  # Maximum learning rate for cosine schedule
    min_learning_rate: float  # Minimum learning rate for cosine schedule
    warmup_iters: int  # Number of warmup iterations
    cosine_cycle_iters: int  # Number of cosine annealing iterations
    beta1: float  # Beta1 parameter for AdamW
    beta2: float  # Beta2 parameter for AdamW
    weight_decay: float  # Weight decay rate for AdamW
    max_l2_norm: float  # Maximum gradient norm for clipping
    
    # Trainer config fields
    train_path: str | os.PathLike | BinaryIO | IO[bytes]  # Path to tokenized training file
    validation_path: str | os.PathLike | BinaryIO | IO[bytes]  # Path to tokenized validation file
    checkpoint_dir: str | os.PathLike | BinaryIO | IO[bytes]  # Directory for saving checkpoints
    max_iteration: int  # Maximum number of training iterations
    batch_size: int  # Number of samples per training batch
    checkpoint_steps: int  # Number of steps between checkpoints
    eval_steps: int  # Number of steps between evaluations
    log_dir: str | os.PathLike | BinaryIO | IO[bytes]  # Directory for saving logs
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration from a YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Initialize Config from a dictionary."""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Config to dictionary."""
        result = {}
        for field in fields(self):
            result[field.name] = getattr(self, field.name)
        return result
    
    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to a YAML file."""
        config_dict = self.to_dict()
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def log(self, logger: Optional[logging.Logger] = None) -> None:
        """Log all configuration fields."""
        if logger is None:
            logger = logging.getLogger(__name__)
        
        logger.info("Configuration:")
        logger.info("-" * 50)
        
        # Group fields by category
        model_fields = []
        optimizer_fields = []
        trainer_fields = []
        
        for field in fields(self):
            value = getattr(self, field.name)
            field_info = f"{field.name}: {value}"
            
            # Categorize fields based on their position in the class
            if field.name in ['vocab_size', 'context_length', 'd_model', 'num_layers', 'num_heads', 'd_ff', 'rope_theta']:
                model_fields.append(field_info)
            elif field.name in ['max_learning_rate', 'min_learning_rate', 'warmup_iters', 'cosine_cycle_iters', 'beta1', 'beta2', 'weight_decay', 'max_l2_norm']:
                optimizer_fields.append(field_info)
            else:
                trainer_fields.append(field_info)
        
        # Log each category
        if model_fields:
            logger.info("Model Configuration:")
            for field_info in model_fields:
                logger.info(f"  {field_info}")
        
        if optimizer_fields:
            logger.info("Optimizer Configuration:")
            for field_info in optimizer_fields:
                logger.info(f"  {field_info}")
        
        if trainer_fields:
            logger.info("Trainer Configuration:")
            for field_info in trainer_fields:
                logger.info(f"  {field_info}")
        
        logger.info("-" * 50)