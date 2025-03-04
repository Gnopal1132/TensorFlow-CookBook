import tensorflow as tf


# One approach is to simply use function.

def softplus(value):
    return tf.math.log(tf.math.exp(value) + 1.0)


# Note, mean = 0, variance = 1/Fan_avg
# shape[0] = fan_in, shape[1] = fan_out; They are simply
# dimension of the weights.
def glorot_normal_initializer(shape, dtype=tf.float32):
    stddev = tf.math.sqrt(2. / shape[0] + shape[1])
    return tf.random.normal(shape, stddev=stddev, dtype=dtype)


def glorot_uniform_initializer(shape, dtype=tf.float32):
    fan_avg = 1. * (shape[0] + shape[1]) / 2.
    temp = tf.math.sqrt(3. / fan_avg)
    return tf.random.uniform(shape, minval=-temp, maxval=temp,
                             dtype=dtype)


def l1_regularizer(weights):
    return tf.math.reduce_sum(tf.abs(0.01 * weights))


def positive_weight_constraint(weights):  # You apply constraints on weights and just return the constrained weights.
    return tf.where(weights < 0, tf.zeros_like(weights), weights)


# Simply implement __call__() method.
# Exactly same for constraints and initializers
class L1Regularizer(tf.keras.regularizers.Regularizer):

    def __init__(self, factor, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor

    def __call__(self, weights):
        return tf.math.reduce_sum(tf.abs(self.factor * weights))

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'factor': self.factor}


if __name__ == '__main__':
    layer = tf.keras.layers.Dense(100, activation=softplus, kernel_initializer=glorot_normal_initializer,
                                  kernel_regularizer=l1_regularizer, kernel_constraint=positive_weight_constraint)
    print(layer)
