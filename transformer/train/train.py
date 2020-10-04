import os
import pickle

from typing import Optional, Iterable

from transformer.train import callbacks
from transformer.config import get_config
from transformer.preprocess import Tokenizer
from transformer.preprocess import load_tokens
from transformer.preprocess import create_training_dataset
from transformer.model import Transformer
from transformer.train.generators import NextTokenBatchGenerator
from transformer.utils import get_tqdm
from transformer.utils import get_logger

logger = get_logger('transformer.train')


def train(config_path: str = 'default',
          tqdm: Optional[Iterable] = None,
          **kwargs):
    """Trains a Transformer model on the config-specified dataset.

    Optionally Tokenizes and restructures the input data to be used with a batch generator.
    Any additional preprocessing (removal of certain words, replacement of words, etc.)
    ought to be done beforehand if required using a custom preprocess.initial_cleanup for each
    respective dataset used.
    """
    config = get_config(config_path)
    tqdm = get_tqdm(tqdm or config.get('tqdm'))
    logger.setLevel(config.logging_level)

    logger('Start Training.')
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
    else:
        tokenizer = Tokenizer.load(path=config.tokenizer_output_path)

    if config.create_dataset and config.load_tokens:
        logger('> loading tokens')
        english_tokens_path = os.path.join(config.tokens_output_dir, 'train-en.pkl')
        german_tokens_path = os.path.join(config.tokens_output_dir, 'train-de.pkl')

        logger('> loading tokens (1/2)')
        english_tokens = load_tokens(path=english_tokens_path)
        logger(f'>>> length of tokens: {len(english_tokens)}')

        logger('> loading tokens (2/2)')
        german_tokens = load_tokens(path=german_tokens_path)
        logger(f'>>> length of tokens: {len(german_tokens)}')

        logger.debug(f'GERMAN TOKENS:  {german_tokens[:3]}')
        logger.debug(f'ENGLISH TOKENS: {english_tokens[:3]}')

    if config.create_dataset:
        logger('> creating dataset for training')
        create_training_dataset(english_tokens=english_tokens,
                                german_tokens=german_tokens,
                                max_samples=config.max_samples,
                                validation_split=config.validation_split,
                                sample_length=config.sample_length,
                                save_dataset=config.save_training_dataset,
                                save_interval=config.save_interval,
                                save_dir=config.processed_dir,
                                save_dir_validation=config.processed_dir_validation,
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

        logger('>>> compiling Transformer model')
        model.compile(optimizer='Adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    else:
        logger('> loading Transformer model')
        model = Transformer.load(config.model_output_path,
                                 compile=True)

    if config.verbose:
        model.summary(print_fn=logger)

    logger('>>> creating batch generator')
    generator = NextTokenBatchGenerator(data_dir=config.processed_dir,
                                        epoch_steps=config.train_steps,
                                        batch_size=config.batch_size,
                                        vocab_size=config.vocab_size,
                                        sample_length=config.sample_length)
    validation_generator = NextTokenBatchGenerator(data_dir=config.processed_dir_validation,
                                                   epoch_steps=config.validation_steps,
                                                   batch_size=config.batch_size,
                                                   vocab_size=config.vocab_size,
                                                   sample_length=config.sample_length)

    logger('>>> creating callbacks')
    use_callbacks = [callbacks.VaswaniLearningRate(steps_per_epoch=generator.steps_per_epoch,
                                                   warmup_steps=config.warmup_steps,
                                                   print_fn=logger.debug),
                     callbacks.WriteLogsToFile(filepath=config.train_logs_output_path, overwrite_old_file=False),
                     callbacks.SaveModel(filepath=config.model_output_path,
                                         on_epoch_end=True),
                     callbacks.PrintExamples(tokenizer=tokenizer, generator=generator, print_fn=logger)]

    logger('> starting training of model')
    model.fit_generator(generator=generator(),
                        validation_data=validation_generator(),
                        steps_per_epoch=generator.steps_per_epoch,
                        validation_steps=validation_generator.steps_per_epoch,
                        epochs=config.epochs,
                        callbacks=use_callbacks,
                        shuffle=False)
    logger('Completed Training.')
