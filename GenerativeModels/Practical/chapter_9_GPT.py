import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package='SelfAttention')
class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, multi_head_block=False):
        super().__init__()
        self.is_multihead = multi_head_block

    def call(self, inputs, mask=False):
        query, key, value = inputs
        d_k = tf.cast(tf.shape(query)[-1], tf.float32)
        
        if not self.is_multihead:  # This case is for simplest attention mechanism [sequence_len, input_dim]
            attention = tf.divide(tf.matmul(query, tf.transpose(key)), tf.sqrt(d_k))
        else:   # Here i am doing transpose using tf.einsum() since batch dimension is also used
            attention = tf.divide(tf.matmul(query, tf.einsum('ijkl -> ijlk', key)), tf.sqrt(d_k))

        if mask:
            ones = tf.ones_like(attention)
            mask = tf.linalg.band_part(ones, -1, 0)   # lower triangular matrix, query on rows, keys on columns
            mask = tf.where(mask == 1, 0., float('-inf'))
            attention += mask
        
        attention = tf.math.softmax(attention, axis=-1)
        out = tf.matmul(attention, value)
        return out, attention
    
    def get_config(self):
        base_config = super().get_config()
        base_config.update({'multi_head_block': self.is_multihead})
        return base_config


@tf.keras.utils.register_keras_serializable(package='MultiHeadAttention')
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_head, d_model):
        super().__init__()
        self.num_head = num_head
        self.d_model = d_model
        self.attention = SelfAttention(multi_head_block=True)
        self.head_dim = self.d_model // self.num_head
        self.linear_layer = tf.keras.layers.Dense(self.d_model)   # This is final W_o
        self.qkv_layer = tf.keras.layers.Dense(3 * d_model, activation=None)

    def call(self, input, mask=False):
        batch_size, sequence_length, input_dimension = tf.shape(input)
        # Lets first create query, key and value pair
        qkv = self.qkv_layer(input)
        qkv = tf.reshape(qkv, shape=[batch_size, sequence_length, self.num_head, 3 * self.head_dim])

        # We manage the dimension of the qkv so that we can parallelize it [batch, heads, sequence_len, dimension], every head can be parallelized.
        qkv = tf.einsum('ijkl->ikjl', qkv)
        query, key, value = tf.split(qkv, num_or_size_splits=3, axis=-1)
        values, attention = self.attention([query, key, value], mask=mask)
        values = tf.reshape(values, shape=[batch_size, sequence_length,
                                           self.num_head * self.head_dim])
        output = self.linear_layer(values)
        return output
    
    def get_config(self):
        base_config = super().get_config()
        base_config.update({'num_head': self.num_head,
                            'd_model': self.d_model})
        return base_config


if __name__ == '__main__':
    input_dim = 1024 
    d_model = 512
    num_heads = 8
    batch_size = 30
    sequence_length = 5
    input_x = tf.random.normal(shape=(batch_size, sequence_length, input_dim))
    model = MultiHeadAttention(num_head=num_heads, d_model=d_model)
    out = model(input_x, mask=True)
    print('Final output: ', out)
