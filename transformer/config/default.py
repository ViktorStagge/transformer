import os

from typing import Dict, \
                   Any, \
                   Optional, \
                   List
from dataclasses import dataclass, \
                        field
from omegaconf import OmegaConf


@dataclass
class _Config:
    # ## Meta ### #
    dataset: str = 'pg-19'
    version: int = 0
    verbose: bool = True
    use_positional_encoding: bool = True
    tqdm: Optional[str] = 'tqdm'  # [tqdm, tqdm-notebook]

    # ## Run ### #
    tokenize: bool = True
    train: bool = True

    # ## Tokenize ### #
    save_tokens: bool = True
    load_tokens: bool = True
    lowercase: bool = False
    vocab_size: int = 16384
    max_tokens_files: Optional[int] = None

    # ### Training ### #
    continue_training: bool = True
    train_steps: int = 12000000
    validation_steps: int = 100000

    epochs: int = 5
    batch_size: int = 128
    d_layers: int = 2
    d_heads: int = 2
    sequence_length: int = 128
    memory_size: int = 256
    d_k: int = 16
    output_size: int = vocab_size
    steps_per_epoch: int = train_steps//sequence_length - 1
    save_interval: int = 5000

    # ### Paths ### #
    input_dir: str = 'data/wma-en-de/input/v0/'
    input_paths: List[str] = field(default_factory=list)
    tokenizer_output_path: str = f'data/wma-en-de/tokenizer/wma-en-de-' \
                                 f't{vocab_size}-' \
                                 f'v{version}.tok'
    tokens_output_dir: str = f'data/wma-en-de/tokenized/v{version}'
    processed_path: str = f'data/wma-en-de/processed/v{version}/train.pkl'
    train_logs_output_path: str = f'data/wma-en-de/training-logs/transformer-wma-en-de-v{version}.txt'
    model_output_path: str = f'data/wma-en-de/model/' \
                             f'transformer-wma-en-de-' \
                             f'v{version}-' \
                             f'e{epochs}-' \
                             f'vs{vocab_size}-' \
                             f'bs{batch_size}-' \
                             f'l{d_layers}-' \
                             f's{sequence_length}-' \
                             f'dk{d_k}.h5'


config = OmegaConf.structured(_Config)

if os.path.exists(config.input_dir):
    config.input_paths = [os.path.join(config.input_dir, filename) for filename in os.listdir(config.input_dir)]