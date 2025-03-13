import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

"""
Positional Encoding: (Refer to the positional_encoding_*.png)

Why Needed?
Positional encoding is a crucial component in Transformer architectures, addressing the model's lack of
inherent understanding of the order or position of tokens in a sequence. Unlike recurrent or convolutional
neural networks, Transformers process input data in parallel, which means they don't inherently capture 
the sequential order of tokens.

The positional encoding is added to the input embeddings of the tokens to give the model 
information about the relative or absolute position of the tokens in the sequence.
 This helps the Transformer understand the sequential relationships between different elements in the input.
 
Let's study the flow:

In the figure, we start with the sentence that we want to translate, we post-pad the sentence to make it fix length as
shown, reaching it to the max_len_sequence. Hence the matrix will be = [max_len_sequence, vocab_size]. Now this is 
passed to the Embedding which has the learnable weight dimension [vocab_size x embedding_size (i.e. 512)). Therefore, in the 
output we get out = [max_len_sequence x 512], which is then added to the positional encoding.

In the second figure, We can see that after adding the positional encoding we get another set of word vectors of same
512 dimension. Now as we did in chapter 2 for each and every word vector we want to generate query, key and value vector
all of 512 dimension each. Therefore, we have 3 learnable param matrices as shown, which is responsible to map
the word vector to query, key and value vector. So total number of vectors = 3 * max_seq_len,
because it's 3 for every word. Now for each we can split it into multiple heads.

The formula for positional encoding is shown in figure 3.

PE(pos,2i): Is the value at position pos and 2i in the positional encoding matrix.
PE(pos,2i+1): Is the value at position pos and 2i+1 in the positional encoding matrix.
pos: is the position of the token in the sequence.
i: is the dimension/index of the positional encoding.
d: is the dimension of the model/embeddings.

Why?
1. Periodicity = Sin and Cosine is periodic function they will repeat after sometime. Let's focus on the second image,
Let's see the third row of the positional encoding which is the added encoding to the third row of the input X. Now,
when we compute attention later to see how much focus this word has to other words. During this phase, because of 
periodicity this word is able to pay attention to let's say 5 words after it and then 10 words after it and then 15
words after it in much more tractable way

2. Constrained Values = Sin and Cosine will constrain the value between +1 and -1. Without it, the values wont be 
bounded. Which means that if we consider the third row, its value will be smaller then the fourth row, which itself
will be smaller then the fifth row and so on... Because of it, when we compute the attention you will find that say
the third row wont be able to capture the context of words at larger locations. i.e. one word wont be able to capture
the context of the later words at larger index locations.

3. Easy to extrapolate for long sequence = The formula is deterministic and easy to compute, and even if we haven't
seen certain sequence length in our train set, we will always be able to interpret in test set. 

We will implement the rewritten version of positional encoding.

"""

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


if __name__ == '__main__':
    max_sequence_len = 10
    d_model = 6

    even_i = tf.range(0, d_model, delta=2, dtype=tf.float32)
    print('even_i: ', even_i, end='\n')

    even_i_denominator = tf.math.pow(10000, even_i/d_model)
    print('even_i_denominator: ', even_i_denominator, end='\n\n')

    odd_i = tf.range(1, d_model, delta=2, dtype=tf.float32)
    print('odd_i: ', odd_i, end='\n')

    odd_i_denominator = tf.math.pow(10000, (odd_i-1)/d_model)
    print('odd_i_denominator: ', odd_i_denominator, end='\n\n')

    # The odd_i_denominator will be equal to even_i_denominator. Therefore, you can simply store it in one variable
    # called the denominator.
    denominator = even_i_denominator

    positions = tf.reshape(tf.range(0, max_sequence_len, dtype=tf.float32), shape=(max_sequence_len,1))
    print('positions: ', positions, end='\n\n')   # We will get a two-dimensional vector one for every word.

    even_PE = tf.math.sin(positions / denominator)
    odd_PE = tf.math.cos(positions / denominator)

    print('even_PE: ', even_PE, end='\n')
    print('odd_PE: ', odd_PE, end='\n\n')

    # Now we have got to interleave these both two. So we want our output to be: even_PE[0][0], odd_PE[0][0],
    # even_PE[0][1], odd_PE[0][1]...

    stacked = tf.stack(values=[even_PE, odd_PE], axis=2)
    positional_encoding = tf.reshape(stacked, shape=[tf.shape(stacked)[0], -1])
    print('Positional Encoding: ', positional_encoding, end='\n')
    # Now the first row is the positional encoding for the first word, similarly the second row is for second word...
    # If you notice every positional encoding is different which will help to differentiate between the respective
    # ordering.
