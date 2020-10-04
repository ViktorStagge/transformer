import os
import numpy as np
import compress_pickle

from keras.utils import to_categorical

from transformer.utils.utils import get_tqdm
from transformer.utils.utils import get_logger

logger = get_logger('transformer')


class NextTokenBatchGenerator:

    def __init__(self,
                 *,
                 data_dir,
                 epochs=None,
                 epoch_steps=None,
                 batch_size,
                 vocab_size,
                 sample_length=100,
                 shuffle=False,
                 tqdm=None):
        """Creates a generator which returns a batch of data, suitable for using with keras.Model.fit_generator()
        """
        assert os.path.exists(data_dir), \
            f'specified directory for files does not exist: {data_dir}'
        if epochs is None:
            epochs = int(1e16)
        self.tqdm = get_tqdm(tqdm)
        self.epoch_steps_specified = epoch_steps
        self.sample_length = sample_length

        self.data_paths = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir)]
        self.total_sample_length = self._count_total_sample_length()

        epoch_steps = min(self.total_sample_length, self.epoch_steps_specified or self.total_sample_length)
        epoch_steps = epoch_steps - batch_size - epoch_steps % batch_size

        self.data = None
        self.data_dir = data_dir
        self.vocab_size = vocab_size
        self.epochs = epochs
        self.epoch_steps = epoch_steps
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.steps_per_epoch = epoch_steps // batch_size

    def __call__(self):
        def _next_token_batch_generator():
            for e in range(self.epochs):
                if self.shuffle:
                    raise NotImplementedError  # maintain same samples after shuffle

                epoch_step = 0
                for path in self.data_paths:
                    logger.debug(f'{path}')
                    english_samples, german_samples = compress_pickle.load(path)
                    logger.debug(f'en={len(english_samples)}, de={len(german_samples)}')

                    x_input, y, y_position = self._expand_batch(english_samples,
                                                                german_samples,
                                                                sample_length=self.sample_length)
                    logger.debug(f'x_input={len(x_input)}, y={len(y)}, y_position={len(y_position)}')

                    y_target = np.array(list(row[col] for row, col in zip(y, y_position)))

                    x_output = y.copy()
                    for i, pos in enumerate(y_position):
                        x_output[i, pos] = 0.
                    x_output = np.roll(x_output, shift=1, axis=1)

                    logger.debug(f'x_output={len(x_output)}, y_target={len(y_target)}')

                    end_step = sum(max(len(e), len(d)) for e, d in zip(english_samples, german_samples))
                    end_step -= end_step % self.batch_size
                    end_step = min(end_step, self.epoch_steps - epoch_step)
                    logger.debug(f'end_step={end_step}')

                    for i in range(0, end_step, self.batch_size):
                        logger.debug(f' about to yield batch')
                        x_input_batch = x_input[i: i+self.batch_size]
                        x_output_batch = x_output[i: i+self.batch_size]
                        y_target_batch = y_target[i: i+self.batch_size]
                        y_position_batch = y_position[i: i+self.batch_size]

                        y_position_matrix_batch = np.full(shape=(len(y_position_batch), self.sample_length),
                                                          fill_value=1e-7)
                        for index, target_position in enumerate(y_position_batch):
                            y_position_matrix_batch[index, target_position] = 1

                        x_batch = [x_input_batch, x_output_batch, y_position_matrix_batch]
                        y_batch = to_categorical(y_target_batch, num_classes=self.vocab_size)

                        epoch_step += self.batch_size
                        # yield x_batch, y_batch, y_position_batch  # TODO: sort position-selection for unknown batch size
                        yield x_batch, y_batch

        # must have a callable `__next__` for keras `fit_generator` to function properly .
        # So to avoid having to use the generator by `iter(generator)`
        return iter(_next_token_batch_generator())

    def new(self):
        return self()

    @staticmethod
    def _expand_batch(x_batch, y_batch, sample_length):
        x = []
        y = []
        y_position = []

        for sample_index, (e_sample, d_sample) in enumerate(zip(x_batch, y_batch)):
            tokens_in_sample = min(max(len(e_sample), len(d_sample)), sample_length)

            e_i = NextTokenBatchGenerator._expand_sample(e_sample, sample_length, tokens_in_sample)
            d_i = NextTokenBatchGenerator._expand_sample(d_sample, sample_length, tokens_in_sample)
            y_position_i = np.arange(0, tokens_in_sample)

            x.append(e_i)
            y.append(d_i)
            y_position.append(y_position_i)

        x = np.concatenate(x)
        y = np.concatenate(y)
        y_position = np.concatenate(y_position)

        return x, y, y_position

    @staticmethod
    def _expand_sample(sample,
                       sample_length,
                       tokens_in_sample):
        if sample_length < len(sample):
            sample = sample[:sample_length]

        sample_entries = np.tile(sample, reps=(tokens_in_sample, 1))
        sample_entries = np.tril(sample_entries)

        padding_length = sample_length - len(sample)
        if padding_length:
            sample_entries = np.pad(sample_entries, ((0, 0), (0, padding_length)), mode='constant', constant_values=0)
        return sample_entries

    def _count_total_sample_length(self):
        total_sample_length = 0

        for path in self.tqdm(self.data_paths, desc='counting samples'):
            en, de = compress_pickle.load(path)
            for en_sample, de_sample in zip(en, de):
                tokens_in_sample = min(max(len(en_sample), len(de_sample)), self.sample_length)
                total_sample_length += tokens_in_sample

        return total_sample_length
