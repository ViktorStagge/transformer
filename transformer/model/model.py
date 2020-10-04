import warnings
from typing import Dict, \
    Optional, \
    Any

from keras import backend as K
from keras.layers import Dense, \
    Input, \
    Lambda
from keras.models import Model

from transformer.model.decoder import Decoder
from transformer.model.embedder import Embedder
from transformer.model.encoder import Encoder
from transformer.model.layers import MultiHeadAttention, \
    ScaledDotProductAttention, \
    LayerNormalization, \
    ReverseEmbedding, \
    PositionalEncoding
from transformer.model.optimizers import get_optimizer


class Transformer(Model):
    """Transformer as described in "Attention is All You Need" by Vaswani et. al.
    """
    def __init__(self,
                 batch_size: int,
                 sequence_length: int,
                 vocab_size: int = 16384,
                 d_layers: int = 1,
                 d_heads: int = 2,
                 d_model: int = 128,
                 d_k: int = 16,
                 d_v: int = 128,
                 d_mlp_hidden: int = 1024,
                 dropout_embedding: float = 0.1,
                 dropout_mlp: float = 0.1,
                 name: str = 'Transformer',
                 use_mask: bool = True,
                 use_positional_encoding: bool = True,
                 previous_epoch_steps: int = 0,
                 **kwargs):
        if batch_size != sequence_length:
            warnings.warn('batch_size and sequence_length have to be of the same size to '
                          'correctly train on all data.')
        if d_layers <= 0:
            warnings.warn('d_layers is 0, not using any layers of the Compressive Transformer.')
        if d_k is None:
            d_k = d_model  # // d_heads
        if d_mlp_hidden is None:
            d_mlp_hidden = d_model

        d_q = d_k

        # Build the internal model structure
        x = Input(batch_shape=(batch_size, sequence_length),
                  name='x')
        x_output = Input(batch_shape=(batch_size, sequence_length),
                         name='x_output')
        x_position = Input(batch_shape=(batch_size, sequence_length),
                           name='x_position')

        # Embed & Encode the input/output sequence
        embedder = Embedder(sequence_length=sequence_length,
                            vocab_size=vocab_size,
                            d_model=d_model,
                            dropout=dropout_embedding,
                            batch_size=batch_size,
                            use_mask=use_mask,
                            use_positional_encoding=use_positional_encoding)
        h_L0, h_output = embedder([x, x_output])

        # Encoder stack
        encoder = Encoder(sequence_length=sequence_length,
                          d_layers=d_layers,
                          d_heads=d_heads,
                          d_model=d_model,
                          d_k=d_k,
                          d_v=d_v,
                          d_mlp_hidden=d_mlp_hidden,
                          dropout_embedding=dropout_embedding,
                          dropout_mlp=dropout_mlp,
                          batch_size=batch_size)
        z_encoder = encoder(h_L0)

        # Decoder stack
        decoder = Decoder(sequence_length=sequence_length,
                          d_layers=d_layers,
                          d_heads=d_heads,
                          d_model=d_model,
                          d_k=d_k,
                          d_v=d_v,
                          d_mlp_hidden=d_mlp_hidden,
                          dropout_embedding=dropout_embedding,
                          dropout_mlp=dropout_mlp,
                          batch_size=batch_size)
        z_decoder = decoder([h_output, z_encoder])

        z_decoder_target = Lambda(lambda inputs: K.batch_dot(inputs[0], inputs[1], axes=(1, 1)))([z_decoder, x_position])
        y = Dense(units=vocab_size,
                  activation='softmax',
                  name='output_dense')(z_decoder_target)
        # dense = Dense(units=d_model,
        #               activation=None,
        #               name='dense')(z_decoder)
        # y = ReverseEmbedding(embedding_layer=embedder.embedding_layer[0],
        #                      batch_size=batch_size,
        #                      activation='softmax',
        #                      name='output')(dense)

        _inputs = [x, x_output, x_position]
        _outputs = y

        super().__init__(inputs=_inputs,
                         outputs=_outputs,
                         name=name,
                         **kwargs)

        # settings
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.d_layers = d_layers
        self.d_model = d_model
        self.d_heads = d_heads
        self.d_q = d_q
        self.d_k = d_k
        self.d_v = d_v
        self.d_mlp_hidden = d_mlp_hidden
        self.dropout_embedding = dropout_embedding
        self.dropout_mlp = dropout_mlp
        self.batch_size = batch_size
        self.use_mask = use_mask
        self.use_positional_encoding = use_positional_encoding
        self.previous_epoch_steps = previous_epoch_steps

    def compile(self,
                optimizer,
                loss=None,
                metrics=None,
                loss_weights=None,
                sample_weight_mode=None,
                weighted_metrics=None,
                target_tensors=None,
                **kwargs):
        optimizer = get_optimizer(optimizer)
        return super().compile(optimizer=optimizer,
                               loss=loss,
                               metrics=metrics,
                               loss_weights=loss_weights,
                               sample_weight_mode=sample_weight_mode,
                               weighted_metrics=weighted_metrics,
                               target_tensors=target_tensors,
                               **kwargs)

    def get_config(self):
        """Returns the config of the Transformer model.
        """
        config = super().get_config()

        config.update(attributes=dict(sequence_length=self.sequence_length,
                                      batch_size=self.batch_size,
                                      vocab_size=self.vocab_size,
                                      d_layers=self.d_layers,
                                      d_heads=self.d_heads,
                                      d_model=self.d_model,
                                      d_k=self.d_k,
                                      d_v=self.d_v,
                                      d_mlp_hidden=self.d_mlp_hidden,
                                      dropout_embedding=self.dropout_embedding,
                                      dropout_mlp=self.dropout_mlp,
                                      use_mask=self.use_mask,
                                      use_positional_encoding=self.use_positional_encoding,
                                      previous_epoch_steps=self.previous_epoch_steps))
        return config

    @classmethod
    def from_config(cls,
                    config: Dict[str, Any],
                    custom_objects: Dict[str, Any] = None):
        """Builds a Transformer model from a config.
        """
        assert 'attributes' in config, \
            f'expected `attributes` to be in config. Received: {config.keys()}'

        model = Transformer(**config['attributes'], name=config['name'])

        return model

    @staticmethod
    def load(filepath: str,
             custom_objects: Dict[str, Any] = None,
             compile: bool = True):
        """Load the Transformer from file.

        Arguments:
            filepath: path to load the Transformer model from
            compile: if specified, compiles the model immediately after loading,
                    with the state of the saved optimizer
            custom_objects: (optional) specify additional custom_objects which are required in order to be able
                    to load the model using `keras.models.load_model`.
        """
        from keras.models import load_model

        if custom_objects is None:
            custom_objects = {'Transformer': Transformer,
                              'Embedder': Embedder,
                              'Encoder': Encoder,
                              'Decoder': Decoder,
                              'PositionalEncoding': PositionalEncoding,
                              'ScaledDotProductAttention': ScaledDotProductAttention,
                              'MultiHeadAttention': MultiHeadAttention,
                              'LayerNormalization': LayerNormalization,
                              'ReverseEmbedding': ReverseEmbedding}

        model = load_model(filepath,
                           custom_objects=custom_objects,
                           compile=compile)

        return model
