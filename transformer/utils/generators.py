import gzip
import pickle
import numpy as np

from keras.utils import to_categorical


class NextTokenBatchGenerator:

    def __init__(self,
                 *,
                 data=None,
                 data_path=None,
                 epochs=None,
                 epoch_steps=None,
                 shuffle=False,
                 batch_size,
                 vocab_size):
        """Creates a generator which returns a batch of data, suitable for using with keras.Model.fit_generator()
        """
        assert data is not None or data_path is not None, \
            'provide either a dataset or a path to the dataset'
        if data is None:
            with gzip.open(data_path, 'rb') as file:
                data = pickle.loads(file.read())
        if epochs is None:
            epochs = int(1e16)
        _epoch_steps = epoch_steps

        x_data, y_data, y_target_position_info = data

        epoch_steps = min(len(x_data), len(y_data), _epoch_steps or len(x_data))
        epoch_steps = epoch_steps - batch_size - epoch_steps % batch_size

        x_data = x_data[:epoch_steps]
        y_data = y_data[:epoch_steps]
        y_target_position_info = y_target_position_info[:epoch_steps]

        self.x_data = x_data
        self.y_data = y_data
        self.y_target_position_info = y_target_position_info
        self.vocab_size = vocab_size
        self.epochs = epochs
        self.epoch_steps = epoch_steps
        self.epoch_steps_specified = _epoch_steps
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.steps_per_epoch = epoch_steps // batch_size

    def __call__(self):

        def _next_token_batch_generator():
            for e in range(self.epochs):
                if self.shuffle:
                    raise NotImplementedError  # maintain same samples after shuffle

                for i in range(0, self.epoch_steps, self.batch_size):
                    x_batch_input = self.x_data[i: i+self.batch_size]
                    x_batch_output = self.y_data[i: i+self.batch_size]
                    target_position = self.y_target_position_info[i: i+self.batch_size]

                    y_batch = np.array(list(row[col] for row, col in zip(x_batch_output, target_position)))
                    x_batch_output[:, target_position] = 0.

                    x_batch = [x_batch_input, x_batch_output]
                    y_batch = to_categorical(y_batch, num_classes=self.vocab_size)

                    yield x_batch, y_batch

        return iter(_next_token_batch_generator())
