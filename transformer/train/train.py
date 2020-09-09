import os
import pickle

from tqdm import tqdm
from sometimer import timer, time_this_method

from transformer.config import get_config
from transformer.preprocess.tokenize import Tokenizer
from transformer.model import Transformer
from transformer.utils.generators import next_token_batch_generator
from transformer.train.callbacks import WriteLogsToFile, \
                                        SaveModel


def train(config_path: str = 'default',
          **kwargs):
    """Trains a Transformer model on the config-specified dataset.

    Optionally Tokenizes and restructures the input data to be used with a batch generator.
    Any additional preprocessing (removal of certain words, replacement of words, etc.)
    ought to be done beforehand if required using a custom preprocess.initial_cleanup for each
    respective dataset used.
    """
    config = get_config(config_path)

    # ### Setup ### #
    if config.tokenize:
        tokenizer = Tokenizer(input_paths=config.input_paths,
                              tokenizer_output_path=config.tokenizer_output_path if config.save_tokens else None,
                              vocab_size=config.vocab_size,
                              lowercase=config.lowercase)
        encodings = tokenizer.encode_files(input_paths=config.input_paths,
                                           tokens_output_dir=config.tokens_output_dir if config.save_tokens else None,
                                           return_encodings=not config.load_tokens,
                                           tqdm=tqdm)
        if config.load_tokens:
            tokens_paths = [os.path.join(config.tokens_output_dir, filename) for
                            filename in os.listdir(config.tokens_output_dir)]

            tokens = []
            for path in tqdm(tokens_paths):
                with open(path, 'rb') as file:
                    t = pickle.load(file)
                tokens.append(t)
        else:
            tokens = [encoding.ids for encoding in encodings]
            del encodings

        training_data = [t for toks in tqdm(tokens) for t in toks]
        del tokens

        if config.save_tokens:
            os.makedirs(os.path.split(config.processed_path)[0], exist_ok=True)

            with open(config.processed_path, 'wb') as file:
                pickle.dump(training_data, file)

    if not config.tokenize or config.load_tokens:
        with open(config.processed_path, 'rb') as file:
            training_data = pickle.load(file)

    config.train_steps = config.train_steps or len(training_data)
    config.steps_per_epoch = config.steps_per_epoch or config.train_steps//config.sequence_length - 1

    # TODO:
    if config.continue_training and os.path.exists(config.model_output_path):
        model = Transformer.load(config.model_output_path, compile=False)
    else:
        model = Transformer(sequence_length=config.sequence_length,
                            d_layers=config.d_layers,
                            d_heads=config.d_heads,
                            d_model=config.d_model,
                            d_k=config.d_k,
                            d_v=config.d_v,
                            d_mlp_hidden=config.d_mlp_hidden,
                            batch_size=config.batch_size,
                            vocab_size=config.vocab_size,
                            use_positional_encoding=True)

    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    if config.verbose:
        print(model.summary())

    generator = next_token_batch_generator(ct=ct,  # TODO: change input structure
                                           data=training_data,
                                           data_path=None,
                                           epoch_steps=config.train_steps,
                                           sequence_length=config.sequence_length,
                                           batch_size=config.batch_size,
                                           vocab_size=config.vocab_size)

    callbacks = [WriteLogsToFile(filepath=config.train_logs_output_path, overwrite_old_file=False),
                 SaveModel(filepath=config.model_output_path,
                           save_every_n_batches=config.save_interval)]

    model.fit_generator(generator(),
                        steps_per_epoch=config.steps_per_epoch,
                        epochs=config.epochs,
                        callbacks=callbacks,
                        shuffle=False)
