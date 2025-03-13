import tensorflow as tf


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_head, d_model, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_head = num_head
        self.d_model = d_model
        self.head_dim = d_model // num_head
        self.qkv_layer = tf.keras.layers.Dense(3 * d_model, activation=None)
        self.linear_layer = tf.keras.layers.Dense(d_model)

    @staticmethod
    def self_attention(query, key, value, mask=None):
        d_k = tf.cast(tf.shape(query)[-1], tf.float32)
        scaled = tf.divide(tf.matmul(query, tf.einsum('ijkl->ijlk', key)),
                           tf.sqrt(d_k))

        if mask:
            tensor = tf.ones_like(scaled)
            mask = tf.linalg.band_part(tensor, -1, 0)  # lower triangular
            new_mask = tf.where(mask == 1, 0., float('-inf'))
            scaled += new_mask

        attention = tf.math.softmax(scaled, axis=-1)
        out = tf.matmul(attention, value)
        return out, attention

    def call(self, x, mask=None, to_print=False):
        batch_size, sequence_length, input_dimension = tf.shape(x)
        if to_print:
            print('batch_size --> ', batch_size, end='\n')
            print('sequence_length --> ', sequence_length, end='\n')
            print('input_dimension --> ', input_dimension, end='\n')

        query_key_value = self.qkv_layer(x)

        if to_print:
            print('query_key_value --> ', tf.shape(query_key_value), end='\n')

        qkv = tf.reshape(query_key_value,
                         shape=[batch_size, sequence_length, self.num_head,
                                3 * self.head_dim])
        if to_print:
            print('Reshaped query_key_value --> ', tf.shape(qkv), end='\n')

        qkv = tf.einsum('ijkl->ikjl', qkv)
        q, k, v = tf.split(qkv, num_or_size_splits=3, axis=-1)

        if to_print:
            print('query --> ', tf.shape(q), end='\n')
            print('key --> ', tf.shape(k), end='\n')
            print('value --> ', tf.shape(v), end='\n')

        values, attention = self.self_attention(q, k, v, mask)
        if to_print:
            print('The Attention: ', tf.shape(attention))
        values = tf.reshape(values, shape=[batch_size, sequence_length,
                                           self.num_head * self.head_dim])
        if to_print:
            print('Final value shape --> ', tf.shape(values), end='\n')

        out = self.linear_layer(values)
        return out

    def get_config(self):
        base_config = super(MultiHeadAttention, self).get_config()
        base_config.update({'d_model': self.d_model,
                            'num_head': self.num_head
                            })
        return base_config


class SentenceEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_sequence_length, d_model, vocab_size, drop_rate, **kwargs):
        super(SentenceEmbedding, self).__init__(**kwargs)
        self.max_sequence = max_sequence_length
        self.vocab_size = vocab_size
        self.drop_rate = drop_rate
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(input_dim=self.vocab_size,
                                                   output_dim=self.d_model)
        self.positional_encoder = PositionalEncoding(max_sequence_length=self.max_sequence,
                                                     d_model=self.d_model)
        self.dropout = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, x, *args, **kwargs):
        x = self.embedding(x)
        x = self.positional_encoder(x)
        x = self.dropout(x)
        return x

    def get_config(self):
        base_config = super(SentenceEmbedding, self).get_config()
        base_config.update({'d_model': self.d_model,
                            'max_sequence_length': self.max_sequence_length,
                            'vocab_size': self.vocab_size,
                            'drop_rate': self.drop_rate
                            })
        return base_config


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_sequence_length, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.denominator = None
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
        self.positional_encoding = self.return_encoding()
        # This will ensure that this won't be considered during the training process.
        self.positional_encoding = tf.Variable(self.positional_encoding, trainable=False, dtype=tf.float32)

    def return_encoding(self):
        # Calculating the denominator
        indices = tf.range(start=0, limit=self.d_model, delta=2)
        self.denominator = tf.math.pow(10000, indices / self.d_model)

        position = tf.reshape(tf.range(self.max_sequence_length,
                                       dtype=tf.float64),
                              shape=[self.max_sequence_length, 1])

        even_pe = tf.math.sin(position / self.denominator)
        odd_pe = tf.math.cos(position / self.denominator)
        stacked = tf.stack(values=[even_pe, odd_pe], axis=2)

        positional_encoding = tf.reshape(stacked, shape=[tf.shape(stacked)[0], -1])
        return tf.cast(positional_encoding, dtype=tf.float32)

    def call(self, inputs, *args, **kwargs):
        return inputs + self.positional_encoding

    def get_config(self):
        base_config = super(PositionalEncoding, self).get_config()
        base_config.update({'d_model': self.d_model,
                            'max_sequence_length': self.max_sequence_length
                            })
        return base_config


class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self, param_shape, eps=1e-5, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.param_shape = param_shape
        self.eps = eps
        self.gamma = tf.Variable(tf.ones(self.param_shape))
        self.beta = tf.Variable(tf.ones(self.param_shape))

    def call(self, inputs, **kwargs):
        dims = [-(i + 1) for i in range(len(self.param_shape))]
        mean = tf.reduce_mean(inputs, axis=dims, keepdims=True)
        variance = tf.reduce_mean((inputs - mean) ** 2, axis=dims, keepdims=True)
        std = tf.math.sqrt(variance + self.eps)
        normalization = tf.divide(tf.subtract(inputs, mean), std)
        out = tf.add(tf.multiply(self.gamma, normalization), self.beta)
        return out

    def get_config(self):
        base_config = super(LayerNormalization, self).get_config()
        base_config.update({'param_shape': self.param_shape,
                            'eps': self.eps
                            })
        return base_config


class FeedForwardNetwork(tf.keras.layers.Layer):
    def __init__(self, d_model, hidden, drop_prob=0.1, **kwargs):
        super(FeedForwardNetwork, self).__init__(**kwargs)
        self.d_model = d_model
        self.hidden = hidden
        self.drop_prob = drop_prob
        self.linear1 = tf.keras.layers.Dense(hidden, activation='relu')
        self.linear2 = tf.keras.layers.Dense(d_model, activation='relu')
        self.dropout = tf.keras.layers.Dropout(rate=drop_prob)

    def call(self, x, **kwargs):
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

    def get_config(self):
        base_config = super(FeedForwardNetwork, self).get_config()
        base_config.update({'d_model': self.d_model,
                            'hidden': self.hidden,
                            'drop_prob': self.drop_prob
                            })
        return base_config


class MultiHeadCrossAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super(MultiHeadCrossAttention, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.kv_layer = tf.keras.layers.Dense(2 * d_model)  # Key and Value will come through Encoder
        self.q_layer = tf.keras.layers.Dense(d_model)   # Made from the target variable
        self.linear_layer = tf.keras.layers.Dense(d_model)

    @staticmethod
    def self_attention(query, key, value, mask=None):
        d_k = tf.cast(tf.shape(query)[-1], tf.float32)
        scaled = tf.divide(tf.matmul(query, tf.einsum('ijkl->ijlk', key)),
                           tf.sqrt(d_k))

        if mask:
            tensor = tf.ones_like(scaled)
            mask = tf.linalg.band_part(tensor, -1, 0)  # lower triangular
            new_mask = tf.where(mask == 1, 0., float('-inf'))
            scaled += new_mask

        attention = tf.math.softmax(scaled, axis=-1)
        out = tf.matmul(attention, value)
        return out, attention

    def call(self, input_, mask=None, to_print=False, **kwargs):
        x, y = input_
        batch_size, sequence_length, input_dimension = tf.shape(x)
        if to_print:
            print('batch_size --> ', batch_size, end='\n')
            print('sequence_length --> ', sequence_length, end='\n')
            print('input_dimension --> ', input_dimension, end='\n')

        key_value = self.kv_layer(x)

        if to_print:
            print('key_value --> ', tf.shape(key_value), end='\n')

        q = self.q_layer(y)

        if to_print:
            print('query size --> ', tf.shape(q), end='\n')

        kv = tf.reshape(key_value,
                        shape=[batch_size, sequence_length, self.num_heads,
                               2 * self.head_dim])

        q = tf.reshape(q, shape=[batch_size, sequence_length,
                                 self.num_heads, self.head_dim])
        kv = tf.einsum('ijkl->ikjl', kv)
        q = tf.einsum('ijkl->ikjl', q)

        k, v = tf.split(kv, num_or_size_splits=2, axis=-1)

        values, attention = self.self_attention(q, k, v, mask)
        if to_print:
            print('The Attention: ', tf.shape(attention))
        values = tf.reshape(values, shape=[batch_size, sequence_length,
                                           self.num_heads * self.head_dim])
        if to_print:
            print('Final value shape --> ', tf.shape(values), end='\n')

        out = self.linear_layer(values)
        return out

    def get_config(self):
        base_config = super(MultiHeadCrossAttention, self).get_config()
        base_config.update({'d_model': self.d_model,
                            'num_heads': self.num_heads
                            })
        return base_config



class DecoderLayer(tf.keras.layers.Layer):

    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, print_=False, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        self.to_print = print_
        self.d_model = d_model
        self.ffn_hidden = ffn_hidden
        self.num_heads = num_heads
        self.drop_prob = drop_prob
        self.self_attention = MultiHeadAttention(d_model=d_model, num_head=num_heads)
        self.norm1 = LayerNormalization(param_shape=[d_model])
        self.dropout1 = tf.keras.layers.Dropout(rate=drop_prob)
        self.encoder_decoder_attention = MultiHeadCrossAttention(d_model=d_model, num_heads=num_heads)
        self.norm2 = LayerNormalization(param_shape=[d_model])
        self.dropout2 = tf.keras.layers.Dropout(rate=drop_prob)
        self.ffn = FeedForwardNetwork(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNormalization(param_shape=[d_model])
        self.dropout3 = tf.keras.layers.Dropout(rate=drop_prob)

    def call(self, new_input, **kwargs):
        x, y = tf.split(new_input, num_or_size_splits=2, axis=0)
        x = tf.squeeze(x, axis=0)
        y = tf.squeeze(y, axis=0)

        residual_y = y

        if self.to_print:
            print(" ---- MASKED SELF ATTENTION ---- ")

        y = self.self_attention(y, mask=True)

        if self.to_print:
            print(" ---- DROP OUT 1 ---- ")

        y = self.dropout1(y)

        if self.to_print:
            print(" ---- ADD + LAYER NORMALIZATION 1 ---- ")

        y = self.norm1(y + residual_y)

        residual_y = y

        if self.to_print:
            print(" ---- CROSS ATTENTION ---- ")

        y = self.encoder_decoder_attention([x, y], mask=None)

        if self.to_print:
            print(" ---- DROP OUT 2 -----")

        y = self.dropout2(y)

        if self.to_print:
            print(" ---- ADD + LAYER NORMALIZATION 2 ---- ")

        y = self.norm2(y + residual_y)

        residual_y = y

        if self.to_print:
            print(" ---- FEED FORWARD 1 ---- ")

        y = self.ffn(y)

        if self.to_print:
            print(" ---- DROP OUT 3 ---- ")

        y = self.dropout3(y)

        if self.to_print:
            print(" ---- ADD + LAYER NORMALIZATION 3 ---- ")

        y = self.norm3(y + residual_y)
        return y

    def get_config(self):
        base_config = super(DecoderLayer, self).get_config()
        base_config.update({'d_model': self.d_model,
                            'ffn_hidden': self.ffn_hidden,
                            'num_heads': self.num_heads,
                            'drop_prob': self.drop_prob,
                            'print_': self.to_print
                            })
        return base_config


class Decoder(tf.keras.layers.Layer):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob,
                 num_layers, max_sequence_length, target_vocab, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.d_model = d_model
        self.ffn_hidden = ffn_hidden
        self.num_heads = num_heads
        self.drop_prob = drop_prob
        self.num_layers = num_layers
        self.max_sequence_length = max_sequence_length
        self.target_vocab = target_vocab

        self.layers = []
        self.data_layer = SentenceEmbedding(max_sequence_length=max_sequence_length,
                                            d_model=d_model, vocab_size=target_vocab,
                                            drop_rate=drop_prob)
        for _ in range(num_layers):
            self.layers.append(DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob))

    def call(self, input_, **kwargs):
        x, y = input_
        y = self.data_layer(y)
        for layer in self.layers:
            new_input = tf.stack([x, y], axis=0)
            y = layer(new_input)
        return y

    def get_config(self):
        base_config = super(Decoder, self).get_config()
        base_config.update({'d_model': self.d_model,
                            'ffn_hidden': self.ffn_hidden,
                            'num_heads': self.num_heads,
                            'drop_prob': self.drop_prob,
                            'num_layers': self.num_layers,
                            'max_sequence_length': self.max_sequence_length,
                            'target_vocab': self.target_vocab})

        return base_config


if __name__ == '__main__':
    d_model = 512
    num_heads = 8
    drop_prob = 0.1
    batch_size = 32
    max_sequence_length = 128
    ffn_hidden = 2048
    num_layers = 5
    target_vocab = 1000

    x = tf.random.uniform(shape=(batch_size, max_sequence_length, d_model), minval=0, maxval=1000)
    y = tf.random.uniform(shape=(batch_size, max_sequence_length), minval=0, maxval=1000)
    decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, target_vocab)

    out = decoder(x, y)
