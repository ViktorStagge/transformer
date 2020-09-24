import os
import warnings
import numpy as np

from typing import Dict, \
                   Optional, \
                   Callable
from keras import callbacks
from keras.callbacks import Callback
from keras import backend as K

from transformer.preprocess.tokenize import Tokenizer
from transformer.train.generators import NextTokenBatchGenerator


class WriteLogsToFile(Callback):
    """Writes the logs created during training containing losses and metrics
    to an additional file.
    """
    def __init__(self, filepath, overwrite_old_file=False):
        super().__init__()
        self.filepath = filepath
        self.overwrite_old_file = overwrite_old_file
        self.line_ending = '\n'

        if overwrite_old_file:
            if os.path.exists(filepath):
                os.remove(filepath)

        directory = os.path.split(filepath)[0]
        os.makedirs(directory, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        log_msg = '\t'.join(f'{k}={v}' for k, v in logs.items())
        msg = f'{epoch:4d}:   {log_msg}{self.line_ending}'
        with open(file=self.filepath, mode='a+') as file:
            file.write(msg)


class AlterLearningRate(Callback):
    """Alters the learning rate during training.
    """
    def __init__(self,
                 learning_rates: Dict):
        # Alternatively use keras.callback.LearningRateScheduler
        super().__init__()
        self.learning_rates = dict((int(k), v) for k, v in learning_rates.items())

    def on_epoch_begin(self, epoch, logs=None):
        if epoch in self.learning_rates:
            # Duplicated due to some confusion in keras community on using `lr` vs. `leaning_rate`
            K.set_value(self.model.optimizer.lr, self.learning_rates[epoch])
            K.set_value(self.model.optimizer.learning_rate, self.learning_rates[epoch])

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        logs['learning_rate'] = K.get_value(self.model.optimizer.learning_rate)


class SaveModel(Callback):
    """Saves the model with the specified interval(s).
    """
    def __init__(self,
                 filepath=None,
                 on_epoch_end=True,
                 save_every_n_batches=None,
                 overwrite_old_file=True):
        super().__init__()
        self.filepath = filepath
        self.overwrite_old_file = overwrite_old_file

        if on_epoch_end:
            self.on_epoch_end = self._on_epoch_end

        if save_every_n_batches:
            warnings.filterwarnings('ignore',
                                    category=RuntimeWarning,
                                    module=callbacks.__name__)
            self.save_every_n_batches = save_every_n_batches
            self.on_batch_end = self._on_batch_end

        directory = os.path.split(filepath)[0]
        os.makedirs(directory, exist_ok=True)

    def _on_epoch_end(self, epoch, logs=None):
        self.model.save(self.filepath, overwrite=self.overwrite_old_file)

    def _on_batch_end(self, batch, logs=None):
        if batch % self.save_every_n_batches == 0 and batch > 0:
            self.model.save(self.filepath, overwrite=self.overwrite_old_file)


class PrintExamples(Callback):
    """Prints output from the model after each epoch.
    """

    def __init__(self,
                 tokenizer: Tokenizer,
                 generator: NextTokenBatchGenerator,
                 print_fn: Optional[Callable] = None):
        super().__init__()
        self.print_fn = print_fn or print
        self.generator = generator
        self.tokenizer = tokenizer

        self.x_test, self.y_test = next(generator())

    def on_epoch_end(self, epoch, logs=None):
        y_pred_tokens = self.model.predict(self.x_test)
        y_pred_words = self.tokenizer.decode_batch(y_pred_tokens)

        for en_sample, de_pred, de_sample in zip(self.x_test, y_pred_words, self.y_test):
            self.print_fn(f'english-actual-input: {en_sample}')
            self.print_fn(f'german - pred -output: {de_pred}')
            self.print_fn(f'german -actual-output: {de_sample}')
