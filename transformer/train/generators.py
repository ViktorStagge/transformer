import os
import numpy as np
import compress_pickle

from keras.utils import to_categorical

from transformer.utils.utils import get_tqdm


class NextTokenBatchGenerator:

    def __init__(self,
                 *,
                 data_dir,
                 epochs=None,
                 epoch_steps=None,
                 shuffle=False,
                 batch_size,
                 vocab_size,
                 tqdm=None):
        """Creates a generator which returns a batch of data, suitable for using with keras.Model.fit_generator()
        """
        assert os.path.exists(data_dir), \
            f'specified directory for files does not exist: {data_dir}'
        if epochs is None:
            epochs = int(1e16)
        tqdm = get_tqdm(tqdm)
        _epoch_steps = epoch_steps

        self.data_paths = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir)]
        self.total_sample_length = sum(len(compress_pickle.load(path)[0]) for path in tqdm(self.data_paths,
                                                                                           desc='counting samples'))

        epoch_steps = min(self.total_sample_length, _epoch_steps or self.total_sample_length)
        epoch_steps = epoch_steps - batch_size - epoch_steps % batch_size

        self.data = None
        self.data_dir = data_dir
        self.vocab_size = vocab_size
        self.epochs = epochs
        self.epoch_steps = epoch_steps
        self.epoch_steps_specified = _epoch_steps
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.steps_per_epoch = epoch_steps // batch_size
        self.tqdm = tqdm

    def __call__(self):
        def _next_token_batch_generator():
            for e in range(self.epochs):
                if self.shuffle:
                    raise NotImplementedError  # maintain same samples after shuffle

                epoch_step = 0
                for path in self.data_paths:
                    x_data, y_data, y_target_position_info = compress_pickle.load(path)
                    end_step = min(self.epoch_steps - epoch_step,
                                   len(x_data) - len(x_data) % self.batch_size)

                    for i in range(0, end_step, self.batch_size):
                        x_batch_input = x_data[i: i+self.batch_size]
                        x_batch_output = y_data[i: i+self.batch_size]
                        target_position = y_target_position_info[i: i+self.batch_size]

                        y_batch = np.array(list(row[col] for row, col in zip(x_batch_output, target_position)))
                        x_batch_output[:, target_position] = 0.

                        x_batch = [x_batch_input, x_batch_output]
                        y_batch = to_categorical(y_batch, num_classes=self.vocab_size)

                        epoch_step += self.batch_size
                        yield x_batch, y_batch

        # must have a callable `__next__` for keras `fit_generator` to function properly ...
        return iter(_next_token_batch_generator())

    def new(self):
        return self()
