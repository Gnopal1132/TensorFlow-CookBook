import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf

"""
Here we are going to learn about transformers.

Here we will study:
    1. Motivation for Attention.
    2. Transformer Architecture.
    3. Self Attention

In traditional RNN for language models. We have inputs, the hidden states and the outputs. RNN uses this to do
Sequence to Sequence modelling. Sequence can be set of words to form a sentence etc. They suffer from two main
disadvantages:

    1. They are slow. We need to feed these inputs one at a time in order to generate outputs sequentially one
       at a time. Also, the training algorithm is also slow. We use the truncated version of backpropagation, also
       called BP through time.

       These hidden states that get generated for every word (in case of word language model), we are not
       truly sure that they represent the context of the word itself. (After all, the context of the current
       word depends on the word before it and words that come after it). Now here in RNN it's clear we
       simply get the context of the word that comes before it not after.

       Even in bidirectional RNN, they see left and then right separately and then concatenate them, not at once. Hence,
       the disadvantage persist. So there is some true meaning or context lost while generating these vector words.

    2. They cannot be Parallelized, because the architecture is purely sequential
       and won't allow any parallel operation.

       One way to improve the quality of the vector words generated is by an Attention mechanism. Let's say we have
       an input sentence as shown below. It's four words but each of them can be represented by their own vectors. Using
       attention we can decide which part each word needs to focus on. In this case we have a table 4x4. And score
       represents how much focus that word is putting on the neighboring words. For example, the word 'my' is putting
       a lot of focus on the word 'Name', and it's going to use the context of the word 'Name', in order to incorporate
       in its own feature vector. Similarly, the word 'Gopal' is putting more emphasis on the word 'Name', therefore
       the vector that corresponds to 'Gopal' is going to incorporate some more context information from the word 'Name'
       which comes before it.

                    My    Name    is    Gopal
               My   8      7      3       4
              Name  7      6      4       5
               is   3      4      8       2
              Gopal 3      6      2       7

       Hence, using attention we can have every word vector better incorporate the context either before it or after it.
       All of it can be incorporated better in their vector then the corresponding RNN counterpart.

       This form of attention is called Self-Attention, because we are attending on the same sentence as we are using
       as input. The self attention can also be used in the computer vision application as well.

       The transformer Architecture has the Attention piece at its crux. Now if we focus on the encoder part (refer to
       img). If I pass the sentence 'My Name is Gopal' simultaneously then it will generate four vectors, one for each
       word. We then pass them simultaneously into the decoder. Now into the decoder we will pass as input out first
       word '<SOS>', the decoder is supposed to generate the first word (say Eng-De machine translation) 'Mein', which
       will be then passes again to the decoder input and will generate 'Name', which again passed to decoder and it 
       will generate 'ist' and again then 'Gopal'...
 
"""
"""
    Let's focus on the Encoder piece (refer to the architecture). In the encoder, we have four inputs 'My Name is Gopal'.
    After positional encoding we will get four vectors whose size is 512 (according to the paper). Then it is passed
    through our encoder. And the encoder is going to generate another set of vectors. The idea is that due to the 
    attention mechanism these vectors are more context aware. Hence, higher quality then the vectors we got after the 
    positional encodings. The main crux being the multi-headed attention part. Now we will see exactly how we can
    implement this multi-head attention mechanism. Before that, let's understand how transformer architecture overcomes
    RNN Disadvantages.
    
    1. Here we can process the stages, an encoder or decoder stage (notice Nx) in parallel, so we can make use of
    modern GPU's, and to make these outcoming vectors more higher quality we have these attention mechanism built in.

"""
"""
Multi Head Attention:

Essentially, every single word that is input to our transformer is going to have three vectors.
        1. A query vector: Indicates what am I looking for: [seq_len x dimension_k]
        2. A key vector: Indicates what I can offer: [seq_len x dimension_k]
        3. A value vector: Indicates what I actually offer: [seq_len x dimension_v]

"""


def self_attention_fn(query, key, value, mask=None):
    d_k = tf.cast(tf.shape(query)[-1], tf.float32)
    scaled = tf.divide(tf.matmul(query, tf.transpose(key)),
                       tf.sqrt(d_k))

    if mask:
        tensor = tf.ones_like(scaled)
        mask = tf.linalg.band_part(tensor, -1, 0)  # lower triangular
        new_mask = tf.where(mask == 1, 0., float('-inf'))
        scaled += new_mask

    attention = tf.math.softmax(scaled, axis=-1)
    out = tf.matmul(attention, value)

    # This out is the contextual value vector now.
    return out, attention


if __name__ == '__main__':
    length_input_sequence, dimension_k, dimension_v = 4, 8, 8
    # length_input_sequence = 4, because the input sentence = 'My name is Gopal'
    # query = np.random.normal(size=(length_input_sequence, dimension_k))
    # key = np.random.normal(size=(length_input_sequence, dimension_k))
    # values = np.random.normal(size=(length_input_sequence, dimension_v))

    query = tf.random.normal(shape=(length_input_sequence, dimension_k))
    key = tf.random.normal(shape=(length_input_sequence, dimension_k))
    values = tf.random.normal(shape=(length_input_sequence, dimension_v))
    print('Query: ', query)
    print('Key: ', key)
    print('Values: ', values, end='\n')

    # For every single word, it has one key, query, and value associated with it. i.e. one 8x1 vector for query, one
    # one 8x1 vector for key, and one 8x1 vector for value.

    # Self Attention
    # For self attention, i.e. to create an initial self attention matrix, we need every word to look at every single
    # other word, just to see if it has a higher affinity towards it or not. Which is given by Query(i.e. What I am
    # looking for), and the Key (i.e. What I currently have). The dot product of these two will lead to a 4x4 matrix
    # because we had a sequence of 4 words. In the dot product we are going to compare one query vector to every other
    # key vector and see it's affinity, as in how much important or focus to put on. In each case the entry of the
    # matrix will be proportional to exactly how much attention we want to focus on each word.
    # For example the first row of the matrix will be for the word 'My' and the entries tells us how much it is
    # going to focus on the other vectors. (Just like the above example).
    #
    # Now why do we need this division by sqrt(dimension_k)?. This is because we want to minimize the variance and
    # stabilize the value of the above dot product. We want the key, query and this dot product in similar range.
    #
    # Masking: Especially required, during the decoder part of the network, so that we don't see the future word
    # while trying to generate the context of the current word. However, in the encoder it's not really required
    # because every word is passed to the encoder simultaneously.
    #
    # In the end, we now have softmax, which converts a vector into probability distribution. So they add up to 1.
    # And they are more interpretable and stable.
    #
    #   Equation: Self-Attention = Softmax(Query.np.transpose(Key) / np.sqrt(dimension_k) + M).Value
    #

    # self_attention = np.matmul(query, key.T)
    self_attention = tf.matmul(query, tf.transpose(key))
    print('Shape of Self-Attention: ', self_attention.shape, end='\n')
    print('Self-Attention: ', self_attention, end='\n\n')

    print('Before division by sqrt(dimension_k): ', end='\n')
    print('Variance of Query: ', np.var(query), end='\n')
    print('Variance of Key: ', np.var(key), end='\n')
    print('Variance of Multiplication: ', np.var(self_attention), end='\n\n')

    normalize_self_attention = self_attention / np.sqrt(dimension_k)
    print('After division by sqrt(dimension_k): ', end='\n')
    print('Variance of Query: ', np.var(query), end='\n')
    print('Variance of Key: ', np.var(key), end='\n')
    print('Variance of Multiplication: ', np.var(normalize_self_attention), end='\n\n')

    # Masking
    mask = np.tril(np.ones((length_input_sequence, length_input_sequence)))
    # If you see the output, the first row is [1, 0, 0, 0]. Because the word 'My' can look only to itself.
    # Second row [1, 1, 0, 0]. The word 'Name' can look only the word 'My' and 'Name'...
    mask[mask == 0] = -np.inf
    mask[mask == 1] = 0
    print('Mask numpy:', mask, end='\n')
    print('Scaled masked numpy:', normalize_self_attention + mask, end='\n\n')
    # Placing -infinity, means we are not going to get any context from it. And since after we will do softmax,
    # it will make exp(-inf) zero.

    tensor = tf.ones_like(normalize_self_attention)  # Will generate a matrix of 1'
    """
    tf.linalg.band_part(input, 0, -1) ==> Upper triangular part.
    tf.linalg.band_part(input, -1, 0) ==> Lower triangular part.
    tf.linalg.band_part(input, 0, 0) ==> Diagonal.
    """
    tf2_mask = tf.linalg.band_part(tensor, -1, 0)  # lower triangular
    tf2_mask = tf.where(tf2_mask == 1, 0., float('-inf'))
    print('Mask TF2:', tf2_mask, end='\n')
    print('Scaled masked TF2:', normalize_self_attention + tf2_mask, end='\n\n')


    def softmax(vector):
        return (np.exp(vector).T / np.sum(np.exp(vector), axis=-1)).T


    numpy_attention_softmax = softmax(normalize_self_attention + mask)
    print('Numpy Softmax: ', numpy_attention_softmax, end='\n\n')

    tf2_attention_softmax = tf.math.softmax(normalize_self_attention + tf2_mask)
    print('TF2 Softmax: ', tf2_attention_softmax, end='\n\n')

    # These matrices now should better encapsulate the context of the word. If you compare these to the original
    # value matrix. You will find that the first row is the same in both because of masking, as for the first
    # word we can only see that word only, and for second, it can look to itself and the previous words.
    new_value_numpy = np.matmul(numpy_attention_softmax, values)
    new_value_tf2 = tf.matmul(tf2_attention_softmax, values)
    print('Original Value', values, end='\n')

    print('New value numpy: ', new_value_numpy, end='\n')
    print('New value tensorflow: ', new_value_tf2, end='\n\n')

    # Hence the final Multi-Head attention can be written as above shown in tensorflow. Either you can
    # pass the mask by yourself. But I have encoded the mask already within the function
    new_value_tf2_function = self_attention_fn(query, key, values, mask=True)
    print('The function New Value: ', new_value_tf2_function, end='\n\n')
