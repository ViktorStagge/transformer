import os
import gzip
import pickle
import compress_pickle
import logging
import numpy as np

from sometimer import timer, time_this_method
from typing import Optional, Iterable

from transformer.config import get_config
from transformer.preprocess import Tokenizer
from transformer.preprocess import create_training_dataset
from transformer.model import Transformer
from transformer.utils.generators import NextTokenBatchGenerator
from transformer.train.callbacks import WriteLogsToFile, \
                                        SaveModel
from transformer.utils import get_tqdm
from transformer.utils import get_logger

logger = get_logger('TRAINING')


def train(config_path: str = 'default',
          tqdm: Optional[Iterable] = None,
          **kwargs):
    """Trains a Transformer model on the config-specified dataset.

    Optionally Tokenizes and restructures the input data to be used with a batch generator.
    Any additional preprocessing (removal of certain words, replacement of words, etc.)
    ought to be done beforehand if required using a custom preprocess.initial_cleanup for each
    respective dataset used.
    """
    logger('Start Training.')
    config = get_config(config_path)
    tqdm = get_tqdm(tqdm or config.get('tqdm'))

    # ### Setup ### #
    if config.tokenize:
        logger('> creating tokenizer...')
        tokenizer = Tokenizer(input_paths=config.input_paths,
                              tokenizer_output_path=config.tokenizer_output_path,
                              vocab_size=config.vocab_size,
                              lowercase=config.lowercase)
        logger('> creating tokens with tokenizer')
        tokenizer.encode_files(input_paths=config.input_paths,
                               tokens_output_dir=config.tokens_output_dir,
                               return_encodings=False,
                               tqdm=tqdm)

    if config.create_dataset and config.load_tokens:
        logger('> loading tokens (1/2)')
        english_tokens_path = os.path.join(config.tokens_output_dir, 'train-en-ascii.pkl')
        german_tokens_path = os.path.join(config.tokens_output_dir, 'train-de-ascii.pkl')

        with open(english_tokens_path, 'rb') as file:
            english_tokens = pickle.load(file)
        logger(f'>>> length of tokens: {len(english_tokens)}')
        logger('> loading tokens (2/2)')
        with open(german_tokens_path, 'rb') as file:
            german_tokens = pickle.load(file)
        logger(f'>>> length of tokens: {len(german_tokens)}')

    if config.create_dataset:
        logger('> creating dataset for training')
        create_training_dataset(english_tokens=english_tokens,
                                german_tokens=german_tokens,
                                max_samples=config.max_samples,
                                sample_length=config.sample_length,
                                save_dataset=config.save_training_dataset,
                                save_interval=config.save_interval,
                                save_dir=config.processed_dir,
                                save_compression=config.compression,
                                tqdm=tqdm)

    if config.retrain or not os.path.exists(config.model_output_path):
        logger('> creating Transformer model')
        model = Transformer(sequence_length=config.sequence_length,
                            d_layers=config.d_layers,
                            d_heads=config.d_heads,
                            d_model=config.d_model,
                            d_k=config.d_k,
                            d_v=config.d_v,
                            d_mlp_hidden=config.d_mlp_hidden,
                            batch_size=config.batch_size,
                            vocab_size=config.vocab_size,
                            use_mask=config.use_mask,
                            use_positional_encoding=config.use_positional_encoding)
    else:
        logger('> loading Transformer model')
        model = Transformer.load(config.model_output_path, compile=False)

    logger('>>> compiling Transformer model')
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    if config.verbose:
        model.summary(print_fn=logger)

    logger('> creating batch generator')
    generator = NextTokenBatchGenerator(data_dir=config.processed_dir,
                                        epoch_steps=config.train_steps,
                                        batch_size=config.batch_size,
                                        vocab_size=config.vocab_size)

    logger('> creating callbacks')
    callbacks = [WriteLogsToFile(filepath=config.train_logs_output_path, overwrite_old_file=False),
                 SaveModel(filepath=config.model_output_path,
                           save_every_n_batches=config.save_interval_training)]

    logger('> starting training of model')
    model.fit_generator(generator(),
                        steps_per_epoch=generator.steps_per_epoch,
                        epochs=config.epochs,
                        callbacks=callbacks,
                        shuffle=False)
    logger('Completed Training.')
