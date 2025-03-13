import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

"""
Layer Normalization. Why? How? and When?

I. Why?

Layer Normalization = Layer wise + Normalization. Normalization, Initially in a neural network the activation of these
neurons will be a wide range of positive and negative values. Normalization encapsulates these values in much more of
a smaller range typically centered around 0. This allows for stable training and convergence.

Now layer normalization is the strategy we apply the normalization in a neural network, refer to the image of 
layer_normalization.png.  In the figure, we can see X is the input neurons: Now to get the output for next layer
we will first calculate the activation as simply, the linear combination of X and then activation function on top of it.
Now to get Y, we will apply normalization for that layer. Here micro_1 and sigma_1 are the mean and std of that layer
output (i.e. the second layer), and r1 and ß1 are the learnable params. It's just like Batch normalization but instead
of doing on batch we do on the output of a layer.

II. When?

In chapter 2, we saw how the multihead attention works, refer to its figure. After dividing the key, query and value
to 8 heads, we apply the formula of self attention. We take query and key do the dot product, apply the formula and get
the value of attention which is then multiplied with values to get the final output of dimension [max_seq_len x 64], all
these 8 vectors are then concatenated. To get the final output of [max_sequence_len x 512], now this is passed to 
Add and Normalization layer as shown in the tranformer architecture. Note, we are also using the residual connections 
for better flow of gradient and to reduce the problem of vanishing and exploding gradient. Then we apply the layer norm.

III. How? (Refer to the figure of layer_normalization_part_*.png)

Initially, we have the matrix where each row represents one word and the word dimension is the columns. In this case,
we have 2 words and 3 dimensions. Now, this matrix can be the initial configuration before passing it to the NN,
or after say one layer. How? Imagine, we have a layer with 3 neurons, every neuron will generate an output, 
hence in total 3 outputs, and those three outputs will be rows in this matrix.

Calculation is simple self explanatory and shown in the images.

"""


class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self, param_shape, eps=1e-5, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.param_shape = param_shape
        self.eps = eps
        self.gamma = tf.Variable(tf.ones(self.param_shape))
        self.beta = tf.Variable(tf.zeros(self.param_shape))

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


if __name__ == '__main__':
    input_ = tf.constant(value=[[[0.2, 0.1, 0.3], [0.5, 0.1, 0.1]]])
    batch, words, embedding = tf.shape(input_)
    # During training, we have a batch dimension, so that we can easily parallelize the computation.
    # So here I switch the dimension making the batch dimension in second. Hence note, we are going to use layer norm
    # not just for the last layer but on that last layer across some batches. Here the batch size is 1, so it wont
    # make too much of a difference.
    input_ = tf.einsum('ijk->jik', input_)
    param_shape = tf.shape(input_)[-2:]
    gamma = tf.Variable(tf.ones(shape=param_shape))
    beta = tf.Variable(tf.zeros(shape=param_shape))

    print('Shape of gamma: ', tf.shape(gamma), end='\n')
    print('Shape of beta: ',tf.shape(beta), end='\n\n')

    # Now let's calculate the dimension across which we are going to perform layer normalization.
    # i.e. the batch dimension as well as the embedding dimension.
    dims = [-(i+1) for i in range(len(param_shape))]

    # Now we simply calculate mean and std across these dimension
    mean = tf.reduce_mean(input_, axis=dims, keepdims=True)
    variance = tf.reduce_mean((input_-mean)**2, keepdims=True, axis=dims)
    std = tf.sqrt(variance + 1e-5)

    print('Mean: ', mean, end='\n')
    print('Variance: ', variance, end='\n')
    print('Standard Deviation: ', std, end='\n')

    y = (input_ - mean) / std
    print('Z-score:', y, end='\n')

    out = gamma * y + beta
    print('Output: ', out, end='\n\n')

    layer_norm = LayerNormalization(param_shape=param_shape)
    print('Layer Output:', layer_norm(input_), end='\n')