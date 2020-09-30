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
    # ### Meta ### #
    dataset: str = 'wma-en-de'
    version: int = 2
    verbose: bool = True
    use_positional_encoding: bool = False
    use_mask: bool = True
    tqdm: Optional[str] = 'tqdm'  # [tqdm, tqdm-notebook, None]
    logging_level: str = 'info'  # [info, debug, critical]

    # ### Run ### #
    tokenize: bool = False
    create_dataset: bool = True
    train: bool = True

    # ### Preprocess: Tokenize ### #
    load_tokens: bool = True
    lowercase: bool = False
    vocab_size: int = 16384
    sample_length: int = 100

    # ### Preprocess: Create [training] Dataset ### #
    max_samples: Optional[int] = 1000000
    save_training_dataset: bool = True
    save_interval: int = 200  # maximum samples per file
    compression: str = 'zipfile'  # ['pickle', 'gzip', 'bz2', 'lzma', 'zipfile', 'lz4']

    # ### Training ### #
    retrain: bool = True
    train_steps: int = 12000000
    validation_steps: int = 100000
    learning_rates: Dict[str, int] = field(default_factory=dict)

    epochs: int = 500
    batch_size: int = 2048  # tokens
    d_layers: int = 1
    d_heads: int = 2
    sequence_length: int = sample_length
    d_model: int = 128
    d_k: int = d_model // d_heads
    d_v: int = d_model // d_heads
    d_mlp_hidden: int = 128
    output_size: int = vocab_size
    steps_per_epoch: int = train_steps//sequence_length - 1
    save_interval_training: int = 5000

    # ### Paths ### #
    input_dir: str = 'data/wma-en-de/input/v1/'
    input_paths: List[str] = field(default_factory=list)
    tokenizer_output_path: str = f'data/wma-en-de/tokenizer/wma-en-de-' \
                                 f't{vocab_size}-' \
                                 f'v{version}.tok'
    tokens_output_dir: str = f'data/wma-en-de/tokenized/v{version}'
    processed_dir: str = f'data/wma-en-de/processed/v{version}/'
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

# Additional Nested Fields
config.learning_rates = {'0': 0.001,
                         '10': 0.001}

# Some interpolated fields
config.logging_level = config.logging_level.upper()
if os.path.exists(config.input_dir):
    config.input_paths = [os.path.join(config.input_dir, filename) for filename in os.listdir(config.input_dir)]
