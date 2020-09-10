import numpy as np
from keras.utils import to_categorical


def next_token_batch_generator(*,
                               data=None,
                               data_path=None,
                               epochs=None,
                               epoch_steps=None,
                               shuffle=False,
                               batch_size,
                               sequence_length,
                               vocab_size):
    """Creates a generator which returns a batch of data, suitable for using with keras.Model.fit_generator()
    """
    assert data is not None or data_path is not None, \
        'provide either a dataset or a path to the dataset'
    if data is None:
        data = np.load(data_path)
    if epochs is None:
        epochs = int(1e16)
    if batch_size != sequence_length:
        raise NotImplementedError('each sample is a position in the sequence')
    _epoch_steps = epoch_steps

    def _next_token_batch_generator():
        x_data, y_data, y_target_position_info = data
        epoch_steps = _epoch_steps or min(x_data.size[0], y_data.size[0])
        epoch_steps = epoch_steps - batch_size - epoch_steps % batch_size

        x_data = x_data[:epoch_steps]
        y_data = y_data[:epoch_steps]

        for e in range(epochs):
            if shuffle:
                raise NotImplementedError  # maintain same samples after shuffle

            for i in range(0, epoch_steps, batch_size):
                x_batch_input = x_data[i: i+batch_size]
                target_position = y_target_position_info[i: i+batch_size]
                y_batch = y_data[i: i+batch_size, target_position]

                x_batch_output = y_batch.copy()
                x_batch_output[:, target_position] = 0.

                x_batch = [x_batch_input, x_batch_output]
                y_batch = to_categorical(y_batch, num_classes=vocab_size)

                yield x_batch, y_batch
        return _next_token_batch_generator
