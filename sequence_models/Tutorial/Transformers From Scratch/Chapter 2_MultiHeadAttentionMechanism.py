import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

"""
MultiHead Attention: (refer to figure multi_head_attention_chapter2.png)

Let's assume we have a vector in the start (at left, say after positional encoding) of size 512x1 (maybe it's for 
a word 'Name' in the previous example 'My Name is Gopal'. This vector breaks down into 3 components vector which are:
query, key and value (each of them are 512x1 vectors). 

Now every single vector (i.e. key, query and value vector) is broken down further to 8 pieces (as shown in figure), and
each piece is going to be a part of creating an attention head. Hence we have 8 attention head per vector. Now each
of them will be passed to some attention unit (which we will see in a while), along with all the other words too.
Remember that we have other words like 'My', 'is', and 'Gopal'. Here we just talked about the breakdown of the word 
'Name'. All these other respective word vectors will also be broken down in a similar way and passed to an attention
unit. And for each head we are going to generate an attention matrix as shown in the right hand side of dimension
[sequence_len x sequence_len] in our this example case it will be [4x4]. As we have 4 words in sequence. Each of these
rows are going to add up to 1. And we are going to have 8 such attention matrix corresponding to every head. These
matrices are then going to generate other output vectors that are concatenated inorder to generate vectors that has
pretty good contextual awareness. The goal of the attention is to make our input word vectors(after positional encoding)
more contextually aware.

"""

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


if __name__ == '__main__':
    sequence_length = 4
    batch_size = 1
    input_dimension = 512   # The vector dimension of every word that goes into the attention unit.
    # (i.e. after positional encoding)
    d_model = 512    # The output of the attention unit for every single word.

    # This input_x is just after the positional encoding and before we input it inside the multi head attention.
    input_x = tf.random.normal(shape=(batch_size, sequence_length, input_dimension))
    print('The shape of the input vector x: ', tf.shape(input_x), end='\n\n')

    # Now we have to map this input_x from this dimension to 3 * d_model to create query, key, and value vector
    # all concatenated, and all of them has the 8 attention head, which we will split up later.
    mapping_layer = tf.keras.layers.Dense(3 * d_model)
    query_key_value = mapping_layer(input_x)
    print('The mapping from input to 3*d_model to get query, key and value: ', query_key_value, end='\n\n')

    # Defining the Variables
    num_heads = 8
    head_dim = d_model // num_heads   # Each of the attention head size will be 512 // 8. Remember, we said that
    # every query, key and value vector is split into 8 attention head. This is exactly what we are doing here.

    # Here 3, means it's the combination of query vector, key vector and value vector.
    query_key_value = tf.reshape(query_key_value, shape=(batch_size, sequence_length, num_heads, 3 * head_dim))
    print('The reshaped query_key_value vector: ', query_key_value, end='\n\n')

    # Now I am going to switch the last two dimension such that the head is in the second position. So that we
    # can easily parallelize each individual head.
    query_key_value = tf.einsum('ijkl->ikjl', query_key_value)
    print('The permuted query_key_value vector: ', query_key_value, end='\n\n')

    # Now we can obtain query key and value vector individually by basically breaking down this entire tensor by last
    # dimension.

    # If num_or_size_splits is an int, then it splits value along the dimension axis into num_or_size_splits
    # smaller tensors. This requires that value.shape[axis] is divisible by num_or_size_splits.
    query, key, value = tf.split(query_key_value, num_or_size_splits=3, axis=-1)
    print('After splitting: ', end='\n')
    print('query --> ', tf.shape(query), end='\n')
    print('key --> ', tf.shape(key), end='\n')
    print('value --> ', tf.shape(value), end='\n\n')

    # Now we are going to do the Attention mechanism. Like we did in the chapter 1. See the function.

    def self_attention_fn(query, key, value, mask=None):
        d_k = tf.cast(tf.shape(query)[-1], tf.float32)

        # Note, here since we have already split our key, query and value into 8 heads, and permuted so
        # that now head is in the second axis, therefore can be parallelized. Now, to do tf.transpose(key),
        # we know that we need to transpose the last two axis. Therefore, I used tf.einsum().
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

    values, attention = self_attention_fn(query, key, value, mask=True)
    # Now these new values vector will be much more context aware, then the original value vector and the original
    # input vector.
    print('The Attention: ', tf.shape(attention), end='\n')  # max_seq_len x max_seq_len
    print('The Value: ', tf.shape(values), end='\n\n')
    # For each batch, each head, each word we have 64 dimensional vector.

    # Now we are going to combine/concatenate all of these 8 heads together. So for 8 heads we are going to make
    # them 512 dimensional vectors, which is exactly the input dimension.
    values = tf.reshape(values, shape=[batch_size, sequence_length,
                                       num_heads * head_dim])
    print('Final value shape --> ', tf.shape(values), end='\n\n')

    # Now so that these heads can be communicated with each other, all the information they have learned. We pass
    # them through a dense layer, from same dimension to same dimension.
    linear_layer = tf.keras.layers.Dense(d_model)

    # Now, this output vector will be much more context aware then the input vector.
    final_output = linear_layer(values)
    print('Final output: ', final_output, end='\n')

    # So we can put now all this, to a Multi-head Attention layer. As shown in the above code.
    # Let's define some reasonable params.
    print('The output of the Layer: ', end='\n\n')
    input_dim = 1024
    d_model = 512
    num_heads = 8
    batch_size = 30
    sequence_length = 5
    input_x = tf.random.normal(shape=(batch_size, sequence_length, input_dim))
    model = MultiHeadAttention(num_head=num_heads, d_model=d_model)
    out = model(input_x, to_print=True, mask=True)
