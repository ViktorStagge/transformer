import numpy as np
import itertools

from typing import Tuple, \
                   Union, \
                   List
from keras import layers
from keras import activations
from keras import backend as K
from keras.layers import Layer


class ReverseEmbedding(Layer):
    def __init__(self,
                 batch_size,
                 embedding_layer: layers.Embedding = None,
                 activation=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.embedding_layer = embedding_layer
        self.vocab_size = embedding_layer.get_config()['input_dim']
        self.activation = activations.get(activation)
        self.trainable = False
        self.batch_size = batch_size

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        assert len(inputs.shape) == 3, \
            'expected 3 dimensions'
        if self.embedding_layer is None:
            return inputs

        input_emb = inputs[:, -1, :]
        w_transpose = K.transpose(self.embedding_layer.embeddings)

        y = K.dot(input_emb, w_transpose)

        if self.activation is not None:
            y = self.activation(y)
        return y

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return mask
        import tensorflow as tf

        new_mask = tf.constant(np.ones(shape=(self.batch_size,)))
        # Unclear behaviour from `keras`.
        # keras output mask during categorical_crossentropy
        # does not use a second dimension for its mask.
        # I.e. the mask shape and output shape *does not match*.
        # In this case, second dimension ought to be `self.embedding_layer.input_dim`
        return new_mask

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.embedding_layer.input_dim

    def get_config(self):
        config = super().get_config()
        config.update(activation=self.activation)
        return config


class PositionalEncoding(Layer):
    def __init__(self,
                 batch_size: int,
                 verbose: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.verbose = verbose
        self.sequence_length = None
        self.d_model = None
        self.encodings = None
        self.W_kr = None

    def build(self,
              input_shape: Tuple):
        assert isinstance(input_shape, tuple), \
            f'received input_shape={input_shape}. Expected a tuple (i.e. single input).'
        assert len(input_shape) == 3, \
            f'expected shape with 3 dimensions: (batch_size, sequence_length, dimensions), ' \
            f'received shape with {len(input_shape)} dimensions: {input_shape}'
        self.sequence_length = input_shape[1]
        self.d_model = input_shape[2]

        self.W_kr = self.add_weight(name='W_k,r',
                                    shape=input_shape[1:],
                                    initializer='uniform',
                                    trainable=True)
        self.encodings = self.create_positional_encodings()

        super().build(input_shape)

    def call(self,
             inputs,
             mask=None,
             **kwargs):
        y = self.encodings * self.W_kr

        if mask is not None:
            next_mask = K.cast(self.compute_mask(inputs, mask), 'float32')

            y = y * next_mask

        if self.verbose:
            print(f'{self.__class__.__name__} call:')
            print(f'  encodings: {self.encodings.shape}')
            print(f'  W_kr:      {self.W_kr.shape}')
            print(f'  inputs:    {inputs.shape}')
            print(f'  mask:      {mask.shape}')
            print(f'  next_mask:      {next_mask.shape}')
            print(f'  y:         {y.shape}')

        assert len(inputs.shape) == len(y.shape), \
            f'unexpected length for produced output: ' \
            f'expected {inputs.shape}, ' \
            f'produced {y.shape}'
        assert inputs.shape[1:] == y.shape[1:], \
            f'unexpected shape for produced output: ' \
            f'expected {inputs.shape[1:]}, ' \
            f'produced {y.shape[1:]}'
        return y

    def compute_output_shape(self,
                             input_shape: Tuple):
        return input_shape

    def create_positional_encodings(self):
        encoding = [PE(pos, i, self.d_model) for pos, i in itertools.product(range(self.sequence_length),
                                                                             range(self.d_model))]
        encoding = np.array(encoding)
        encoding = encoding.reshape((self.sequence_length, self.d_model))
        encoding = np.tile(encoding, (self.batch_size, 1, 1))
        return encoding

    def compute_mask(self, inputs, mask=None):
        next_mask = K.tile(K.expand_dims(mask, -1), [1, 1, self.d_model])
        return next_mask

    def get_config(self):
        config = super().get_config()
        config.update(batch_size=self.batch_size,
                      verbose=self.verbose)
        return config


def PE(pos, i, max_dimension):
    """Positional Encoding

    Arguments:
        pos: position in the sequence
        i: dimension, referred to in the paper as "i"
        max_dimension: maximum amount of dimensions used
    """
    alpha = pos/10000**(2 * i / max_dimension)

    if i % 2 == 0:
        return np.sin(alpha)
    return np.cos(alpha)
