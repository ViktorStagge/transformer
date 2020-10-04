from typing import Optional
from keras.layers import Input, \
                         Embedding, \
                         Add, \
                         Dropout
from keras.models import Model

from transformer.model.layers import PositionalEncoding


class Embedder(Model):

    def __init__(self,
                 sequence_length: int,
                 vocab_size: int = 16384,
                 d_model: int = 128,
                 dropout: float = 0.1,
                 batch_size: Optional[int] = None,
                 name: str = 'Embedder',
                 use_positional_encoding: bool = True,
                 use_mask: bool = True,
                 **kwargs):
        x = Input(batch_shape=(batch_size, sequence_length),
                  name='x')
        x_output = Input(batch_shape=(batch_size, sequence_length),
                         name='x_output')

        # Embedding
        embedding_layer = Embedding(input_dim=vocab_size,
                                    output_dim=d_model,
                                    embeddings_initializer='uniform',
                                    mask_zero=use_mask,
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
