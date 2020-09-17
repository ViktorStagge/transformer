import os
import gzip
import pickle
import logging
import numpy as np

from sometimer import timer, time_this_method
from typing import Optional, Iterable

from transformer.config import get_config
from transformer.preprocess.tokenize import Tokenizer
from transformer.model import Transformer
from transformer.utils.generators import NextTokenBatchGenerator
from transformer.train.callbacks import WriteLogsToFile, \
                                        SaveModel
from transformer.utils.utils import get_tqdm
from transformer.utils.utils import get_logger

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
        training_data = preprocess_create_training_data(english_tokens=english_tokens,
                                                        german_tokens=german_tokens,
                                                        max_samples=config.max_samples,
                                                        sample_length=config.sample_length,
                                                        tqdm=tqdm)

        if config.save_training_dataset:
            logger('>>> saving dataset for training')
            os.makedirs(os.path.split(config.processed_path)[0], exist_ok=True)

            with gzip.open(config.processed_path, 'wb') as file:
                file.write(pickle.dumps(training_data))
    else:
        logger('> loading dataset for training')
        with gzip.open(config.processed_path, 'rb') as file:
            training_data = pickle.loads(file.read())

    config.train_steps = config.train_steps or len(training_data)
    config.steps_per_epoch = config.steps_per_epoch or config.train_steps//config.sequence_length - 1

    if not config.continue_training or not os.path.exists(config.model_output_path):
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
    generator = NextTokenBatchGenerator(data=training_data,
                                        data_path=None,
                                        epoch_steps=config.train_steps,
                                        batch_size=config.batch_size,
                                        vocab_size=config.vocab_size)

    logger('> creating callbacks')
    callbacks = [WriteLogsToFile(filepath=config.train_logs_output_path, overwrite_old_file=False),
                 SaveModel(filepath=config.model_output_path,
                           save_every_n_batches=config.save_interval)]

    logger('> starting training of model')
    model.fit_generator(generator(),
                        steps_per_epoch=generator.steps_per_epoch,
                        epochs=config.epochs,
                        callbacks=callbacks,
                        shuffle=False)
    logger('Completed Training.')


def preprocess_create_training_data(english_tokens,
                                    german_tokens,
                                    max_samples=None,
                                    sample_length=100,
                                    tqdm=None):
    max_samples = min(max_samples or len(english_tokens), len(english_tokens), len(german_tokens))
    if tqdm is None:
        tqdm = get_tqdm()
    x = []
    y = []
    y_target_position_info = []

    for sample_index, e_sample, d_sample in zip(tqdm(range(max_samples)), english_tokens, german_tokens):
        e_sample_entries = _preprocess_create_training_data_samples(e_sample, sample_length=sample_length)
        d_sample_entries = _preprocess_create_training_data_samples(d_sample, sample_length=sample_length)
        y_sample_target_position = np.arange(0, sample_length)

        x.append(e_sample_entries)
        y.append(d_sample_entries)
        y_target_position_info.append(y_sample_target_position)

    x = np.concatenate(x)
    y = np.concatenate(y)
    y_target_position_info = np.concatenate(y_target_position_info)

    return x, y, y_target_position_info


def _preprocess_create_training_data_samples(sample,
                                             sample_length):
    if sample_length < len(sample):
        sample = sample[:sample_length]

    sample_entries = np.tile(sample, reps=(len(sample), 1))
    sample_entries = np.tril(sample_entries)

    padding_length = sample_length - len(sample)
    if padding_length:
        sample_entries = np.pad(sample_entries, ((0, 0), (0, padding_length)), mode='constant', constant_values=0)
    return sample_entries
