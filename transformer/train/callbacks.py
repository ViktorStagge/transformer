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


class VaswaniLearningRate(Callback):
    """Alters the learning rate during training according to Vaswani et. al
    in `Attention Is All You Need`.
    """
    def __init__(self,
                 steps_per_epoch: int,
                 warmup_steps: int = 4000,
                 previous_steps: Optional[int] = None,
                 print_fn: Optional[Callable] = None):
        # Alternatively use keras.callback.LearningRateScheduler
        super().__init__()
        self.steps_per_epoch = steps_per_epoch
        self.warmup_steps = warmup_steps
        self.print_fn = print_fn
        if previous_steps is not None:
            self.model.previous_epoch_steps = previous_steps

    def learning_rate(self, batch):
        current_step = self.model.previous_epoch_steps + batch + 1  # referred to as `step_num` in paper
        learning_rate = self.model.d_model**(-0.5) * min(current_step**(-0.5),
                                                         current_step * self.warmup_steps**(-1.5))
        return learning_rate

    def on_batch_begin(self, batch, logs=None):
        learning_rate = self.learning_rate(batch)

        # Duplicated due to some confusion in keras community on using `lr` vs. `leaning_rate`
        K.set_value(self.model.optimizer.lr, learning_rate)
        K.set_value(self.model.optimizer.learning_rate, learning_rate)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        logs['learning_rate'] = K.get_value(self.model.optimizer.learning_rate)

        if self.print_fn is not None:
            self.print_fn(f'Current learning rate: {K.get_value(self.model.optimizer.learning_rate)}')

        self.model.previous_epoch_steps += self.steps_per_epoch


class SaveModel(Callback):
    """Saves the model with the specified interval(s).
    """
    def __init__(self,
                 filepath=None,
                 on_epoch_end=None,
                 save_every_n_batches=None,
                 overwrite_old_file=True):
        super().__init__()
        self.filepath = filepath
        self.overwrite_old_file = overwrite_old_file
        if save_every_n_batches is None:
            on_epoch_end = on_epoch_end or True

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
                 max_output_tokens: int = 1500,
                 print_fn: Optional[Callable] = None):
        super().__init__()
        self.print_fn = print_fn or print
        self.generator = generator
        self.tokenizer = tokenizer
        self.max_output_tokens = max_output_tokens

        self.print_fn(f'generator ...')
        (self.x, self.x_output, self.x_target), self.y = next(generator())
        importance_mask = np.arange(np.prod(self.x.shape)).reshape(self.x.shape) * (self.x > 0)
        last_index = np.argmax(importance_mask, axis=1)
        self.x_text = tokenizer.decode([int(self.x[i, j]) for i, j in enumerate(last_index)])

        self.y_text = tokenizer.decode(list(np.argmax(self.y, axis=1).astype('int')))
        self.print_fn('Sample for evaluation:')
        self.print_fn(f'{self.x_text[:100]} [...]')
        self.print_fn(f'{self.y_text[:100]} [...]\n')

    def on_epoch_end(self, epoch, logs=None):
        y_pred_tokens = self.model.predict([self.x, self.x_output, self.x_target],
                                           batch_size=self.generator.batch_size)
        y_pred_tokens = np.argmax(y_pred_tokens, axis=1)
        y_pred_tokens = y_pred_tokens.astype('int')
        y_pred_tokens = list(y_pred_tokens)

        try:

            # thread '<unnamed>' panicked at 'assertion failed: !b.shape.is_null()',
            # /github/home/.cargo/registry/src/github.com-[...]
            y_pred_words = self.tokenizer.decode(y_pred_tokens)

            self.print_fn(f'english-actual-input: {self.x_text[:self.max_output_tokens]}\n')
            self.print_fn(f'german - pred -output: {y_pred_words[:self.max_output_tokens]}\n')
            self.print_fn(f'german -actual-output: {self.y_text[:self.max_output_tokens]}\n')
        except Exception as e:
            self.print_fn(f'failed to create samples for epoch {epoch}')
