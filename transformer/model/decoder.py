import warnings

from typing import Optional
from keras.layers import Input, \
                         Add, \
                         Dropout, \
                         Dense
from keras.models import Model

from transformer.model.layers import ScaledDotProductAttention, \
                                     MultiHeadAttention, \
                                     LayerNormalization


class Decoder(Model):
    """The decoding model of the Transformer.
    """

    def __init__(self,
                 sequence_length: int,
                 d_layers: int = 1,
                 d_heads: int = 2,
                 d_model: int = 128,
                 d_k: int = 16,
                 d_v: int = 128,
                 d_mlp_hidden: int = 1024,
                 dropout_embedding: float = 0.1,
                 dropout_mlp: float = 0.1,
                 batch_size: Optional[int] = None,
                 name: str = 'Decoder',
                 **kwargs):
        batch_size = batch_size or sequence_length
        if batch_size != sequence_length:
            warnings.warn('batch_size and sequence_length have to be of the same size to '
                          'correctly train on all data.')
        if d_layers <= 0:
            warnings.warn('d_layers is 0, not using any layers of the Encoder.')

        d_q = d_k

        # Build the internal model structure
        h_output = Input(shape=(sequence_length, d_model),
                         name='h_output')
        z_encoder = Input(shape=(sequence_length, d_model),
                          name='z_encoder')

        h = h_output
        for i in range(d_layers):
            # Masked Multi-Head Attention [masked before embedding h_output]
            masked_sdpa_layers = [ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_v) for _ in range(d_heads)]  # TODO: d_q, d_k, d_v
            masked_sdpa = [sdpa_layer([h, h]) for sdpa_layer in masked_sdpa_layers]

            masked_mha = MultiHeadAttention(d_heads=d_heads,
                                            sequence_length=sequence_length,
                                            d_model=d_model,
                                            d_k=d_k,
                                            d_v=d_v,
                                            name=f'masked_multihead_attention_L{i}')(masked_sdpa)
            masked_mha_skip = Add(name=f'masked_mha_skip_L{i}')([h, masked_mha])
            masked_a = LayerNormalization(name=f'masked_mha_layer_norm_L{i}')(masked_mha_skip)

            # Multi-Head Attention
            sdpa_layers = [ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_v) for _ in range(d_heads)]  # TODO: d_q, d_k, d_v. before d_model, d_k, d_model
            sdpa = [sdpa_layer([z_encoder, masked_a]) for sdpa_layer in sdpa_layers]  # TODO: check [h, h]

            mha = MultiHeadAttention(d_heads=d_heads,
                                     sequence_length=sequence_length,
                                     d_model=d_model,
                                     d_k=d_k,
                                     d_v=d_v,
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

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        model = super().from_config(config=config, custom_objects=custom_objects)
        return model
