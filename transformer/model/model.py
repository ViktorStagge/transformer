import warnings
import numpy as np

from typing import List, \
                   Dict
from keras import backend as K
from keras.layers import Embedding, \
                         Dense, \
                         Dropout, \
                         Flatten, \
                         Input, \
                         Add, \
                         Lambda, \
                         concatenate as Concatenate
from keras.models import Model

from transformer.config import get_config
from transformer.model.layers import MultiHeadAttention, \
                                     ScaledDotProductAttention, \
                                     LayerNormalization, \
                                     ReverseEmbedding, \
                                     PositionalEncoding
from transformer.model.optimizers import get_optimizer

config = get_config(path='default')


class Transformer(Model):
    """Transformer as described in "Attention is All You Need" by [...].
    """
    def __init__(self,
                 sequence_length,
                 vocab_size=16384,
                 d_layers=1,
                 d_heads=2,
                 d_model=128,
                 d_k=16,
                 d_v=128,
                 d_mlp_hidden=1024,
                 dropout_embedding=0.1,
                 dropout_mlp=0.1,
                 batch_size=None,
                 name='Transformer',
                 use_positional_encoding=True,
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
        if use_positional_encoding is None:
            use_positional_encoding = config.use_positional_encoding

        d_q = d_k

        # Build the internal model structure
        x = Input(shape=(sequence_length,),
                  name='x')
        x_output = Input(shape=(sequence_length,),
                         name='x_output')

        embedder = Embedder(sequence_length=sequence_length,
                            vocab_size=vocab_size,
                            d_model=d_model,
                            dropout=dropout_embedding,
                            batch_size=batch_size,
                            use_positional_encoding=use_positional_encoding)
        h_L0, h_output = embedder([x, x_output])

        # Encoder stack
        encoder = Encoder(sequence_length=sequence_length,
                          vocab_size=vocab_size,
                          d_layers=d_layers,
                          d_heads=d_heads,
                          d_q=d_q,
                          d_k=d_k,
                          d_v=d_v,
                          d_mlp_hidden=d_mlp_hidden,
                          dropout_embedding=dropout_embedding,
                          dropout_mlp=dropout_mlp,
                          batch_size=batch_size,
                          use_positional_encoding=use_positional_encoding)
        z_encoder = encoder(h_L0)

        # Decoder stack
        decoder = Decoder(sequence_length=sequence_length,
                          vocab_size=vocab_size,
                          d_layers=d_layers,
                          d_heads=d_heads,
                          d_q=d_q,
                          d_k=d_k,
                          d_v=d_v,
                          d_mlp_hidden=d_mlp_hidden,
                          dropout_embedding=dropout_embedding,
                          dropout_mlp=dropout_mlp,
                          batch_size=batch_size,
                          use_positional_encoding=use_positional_encoding)
        z_decoder = decoder([h_output, z_encoder])

        dense = Dense(units=d_model,
                      activation=None,
                      name='dense_post_decoder_stack')(z_decoder)
        y = ReverseEmbedding(embedder.embedding_layer[0],
                             activation='softmax',
                             name='output')(dense)

        inputs = [x, x_output],
        outputs = y

        super().__init__(inputs=inputs,
                         outputs=outputs,
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
        self.use_positional_encoding = use_positional_encoding

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
                                      d_q=self.d_q,
                                      d_k=self.d_k,
                                      d_v=self.d_v,
                                      d_mlp_hidden=self.d_mlp_hidden,
                                      dropout_embedding=self.dropout_embedding,
                                      dropout_mlp=self.dropout_mlp,
                                      use_positional_encoding=self.use_positional_encoding))
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        """Builds a Transformer model from a config.
        """
        assert 'attributes' in config, \
            f'expected `attributes` to be in config. Received: {config.keys()}'

        model = Transformer(**config['attributes'], name=config['name'])

        return model

    @staticmethod
    def load(filepath,
             custom_objects=None,
             compile=True):
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
                              'Encoder': Encoder,
                              'Decoder': Decoder,
                              'PositionalEncoding': PositionalEncoding,
                              'ScaledDotProductAttention': ScaledDotProductAttention,
                              'MultiHeadAttention': MultiHeadAttention,
                              'LayerNormalization': LayerNormalization,
                              'ReverseEmbedding': ReverseEmbedding}

        model = load_model(filepath, custom_objects=custom_objects, compile=compile)

        return model


class Encoder(Model):
    """The encoding model of the Transformer.
    """

    def __init__(self,
                 sequence_length,
                 d_layers=1,
                 d_heads=2,
                 d_model=128,
                 d_k=16,
                 d_v=128,
                 d_mlp_hidden=1024,
                 dropout_embedding=0.1,
                 dropout_mlp=0.1,
                 batch_size=None,
                 name='Encoder',
                 use_positional_encoding=True,
                 **kwargs):
        batch_size = batch_size or sequence_length
        if batch_size != sequence_length:
            warnings.warn('batch_size and sequence_length have to be of the same size to '
                          'correctly train on all data.')
        if d_layers <= 0:
            warnings.warn('d_layers is 0, not using any layers of the Encoder.')
        if use_positional_encoding is None:
            use_positional_encoding = config.use_positional_encoding

        d_q = d_k

        # Build the internal model structure
        h_L0 = Input(shape=(sequence_length,),
                     name='h_L0')

        h = h_L0
        for i in range(d_layers):
            # #### Multi Head Attention #####
            sdpa_layers = [ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_model) for _ in range(d_heads)]  # TODO: d_q, d_k, d_v
            sdpa = [sdpa_layer([h, h]) for sdpa_layer in sdpa_layers]  # TODO: check [h, h]

            mha = MultiHeadAttention(d_heads=d_heads,
                                     d_model=d_model,
                                     d_k=d_k,
                                     d_v=d_model,
                                     name=f'multihead_attention_L{i}')(sdpa)
            mha_skip = Add(name=f'mha_skip_L{i}')([h, mha])
            a = LayerNormalization(name=f'mha_layer_norm_L{i}')(mha_skip)
            # # #### #################### #####

            mlp_hidden = Dense(units=d_mlp_hidden,
                               activation='relu',
                               name=f'mlp_hidden_0_L{i}')(a)
            mlp = Dense(units=d_model,
                        activation=None,
                        name=f'mlp_no_activation_L{i}')(mlp_hidden)
            mlp_drop = Dropout(rate=dropout_mlp, name=f'dropout_L{i}')(mlp)
            mlp_skip = Add(name=f'mlp_skip_L{i}')([mlp_drop, a])

            h = LayerNormalization(name=f'h_L{i+1}')(mlp_skip)  # h, for L_{i+1}

        inputs = h_L0
        outputs = h

        super().__init__(inputs=inputs,
                         outputs=outputs,
                         name=name,
                         **kwargs)

        # settings
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.d_layers = d_layers
        self.d_model = d_model
        self.d_heads = d_heads
        self.d_q = d_q
        self.d_k = d_k
        self.d_v = d_v
        self.d_mlp_hidden = d_mlp_hidden
        self.dropout_embedding = dropout_embedding
        self.dropout_mlp = dropout_mlp
        self.use_positional_encoding = use_positional_encoding

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        model = super().from_config(config=config, custom_objects=custom_objects)
        return model


class Decoder(Model):
    """The decoding model of the Transformer.
    """

    def __init__(self,
                 sequence_length,
                 d_layers=1,
                 d_heads=2,
                 d_model=128,
                 d_k=16,
                 d_v=128,
                 d_mlp_hidden=1024,
                 dropout_embedding=0.1,
                 dropout_mlp=0.1,
                 batch_size=None,
                 name='Encoder',
                 use_positional_encoding=True,
                 **kwargs):
        batch_size = batch_size or sequence_length
        if batch_size != sequence_length:
            warnings.warn('batch_size and sequence_length have to be of the same size to '
                          'correctly train on all data.')
        if d_layers <= 0:
            warnings.warn('d_layers is 0, not using any layers of the Encoder.')
        if use_positional_encoding is None:
            use_positional_encoding = config.use_positional_encoding

        d_q = d_k

        # Build the internal model structure
        h_output = Input(shape=(sequence_length,),
                         name='h_output')
        z_encoder = Input(shape=(sequence_length,),
                          name='z_encoder')

        h = h_output
        for i in range(d_layers):
            # Masked Multi-Head Attention
            masked_sdpa_layers = [ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_model) for _ in range(d_heads)]  # TODO: d_q, d_k, d_v
            masked_sdpa = [sdpa_layer([h, h]) for sdpa_layer in masked_sdpa_layers]  # TODO: check [h_output, h_output]

            masked_mha = MaskedMultiHeadAttention(d_heads=d_heads,
                                                  d_model=d_model,
                                                  d_k=d_k,
                                                  d_v=d_model,
                                                  name=f'masked_multihead_attention_L{i}')(masked_sdpa)
            masked_mha_skip = Add(name=f'masked_mha_skip_L{i}')([h, masked_mha])
            masked_a = LayerNormalization(name=f'masked_mha_layer_norm_L{i}')(masked_mha_skip)

            # Multi-Head Attention
            sdpa_layers = [ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_model) for _ in range(d_heads)]  # TODO: d_q, d_k, d_v
            sdpa = [sdpa_layer([z_encoder, masked_a]) for sdpa_layer in sdpa_layers]  # TODO: check [h, h]

            mha = MultiHeadAttention(d_heads=d_heads,
                                     d_model=d_model,
                                     d_k=d_k,
                                     d_v=d_model,
                                     name=f'multihead_attention_L{i}')(sdpa)
            mha_skip = Add(name=f'mha_skip_L{i}')([h, mha])
            a = LayerNormalization(name=f'mha_layer_norm_L{i}')(mha_skip)
            # # #### #################### #####

            mlp_hidden = Dense(units=d_mlp_hidden,
                               activation='relu',
                               name=f'mlp_hidden_0_L{i}')(a)
            mlp = Dense(units=d_model,
                        activation=None,
                        name=f'mlp_no_activation_L{i}')(mlp_hidden)
            mlp_drop = Dropout(rate=dropout_mlp, name=f'dropout_L{i}')(mlp)
            mlp_skip = Add(name=f'mlp_skip_L{i}')([mlp_drop, a])

            h = LayerNormalization(name=f'h_L{i+1}')(mlp_skip)

        inputs = [h_output, z_encoder]
        outputs = h

        super().__init__(inputs=inputs,
                         outputs=outputs,
                         name=name,
                         **kwargs)

        # settings
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.d_layers = d_layers
        self.d_model = d_model
        self.d_heads = d_heads
        self.d_q = d_q
        self.d_k = d_k
        self.d_v = d_v
        self.d_mlp_hidden = d_mlp_hidden
        self.dropout_embedding = dropout_embedding
        self.dropout_mlp = dropout_mlp
        self.use_positional_encoding = use_positional_encoding

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        model = super().from_config(config=config, custom_objects=custom_objects)
        return model


class Embedder(Model):

    def __init__(self,
                 sequence_length,
                 vocab_size=16384,
                 d_model=128,
                 dropout=0.1,
                 batch_size=None,
                 name='Embedder',
                 use_positional_encoding=True,
                 **kwargs):
        x = Input(shape=(sequence_length,),
                  name='x')
        x_output = Input(shape=(sequence_length,),
                         name='x_output')

        # Embedding
        embedding_layer = Embedding(input_dim=vocab_size,
                                    output_dim=d_model,
                                    embeddings_initializer='uniform',
                                    name='word_embedding')

        encoder_embedding = embedding_layer(x)
        decoder_embedding = embedding_layer(x_output)

        # Postional Encoding
        if use_positional_encoding:
            pe_layer = PositionalEncoding(batch_size=batch_size,
                                          verbose=True,
                                          name='positional_encoding')

            encoder_pos = pe_layer(encoder_embedding)
            decoder_pos = pe_layer(decoder_embedding)

            encoder_embedding = Add(name='encoder_total_embedding')([encoder_embedding, encoder_pos])
            decoder_embedding = Add(name='decoder_total_embedding')([decoder_embedding, decoder_pos])

        # Dropout
        h_L0 = Dropout(rate=dropout, name='h_L0')(encoder_embedding)
        h_output = Dropout(rate=dropout, name='h_output')(decoder_embedding)

        inputs = [x, x_output]
        outputs = [h_L0, h_output]

        super().__init__(inputs=inputs,
                         outputs=outputs,
                         name=name,
                         **kwargs)
        self.embedding_layer = [embedding_layer]

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        model = super().from_config(config=config, custom_objects=custom_objects)
        return model
