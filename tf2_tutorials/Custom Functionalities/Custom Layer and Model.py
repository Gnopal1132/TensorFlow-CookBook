import tensorflow as tf

"""
There are two ways of doing it. Either you subclass the ''layer class'' or ''model class''.
The only methods you need to implement are:
    
    1. def __init__(self): Here just initialize the constant parameters. You can also initialize weights here but its not
        recommended to do so. For initializing the weights use the build method.
    
    2. def build(self, inputs_shape): Initialize your variables here.
    
    3. def call(self, inputs): It does the computation here on the inputs
        
Note: In cases where you know the input_dimension in advance you can initialize them in constructor 
and in that case you need not to implement build() function.
 But if you dont know always use build() it will give u the size.
"""

"""
add_weight(
    name=None,
    shape=None,
    dtype=None,
    initializer=None,
    regularizer=None,
    trainable=None,
    constraint=None,
    use_resource=None,
    synchronization=tf.VariableSynchronization.AUTO,
    aggregation=tf.VariableAggregation.NONE,
    **kwargs
)
"""


class DenseLayer(tf.keras.layers.Layer):
    def __init__(self, units=100, **kwargs):
        super(DenseLayer, self).__init__(**kwargs)
        self.bias = None
        self.kernel = None
        self.units = units

    def build(self, input_shape):
        self.kernel = self.add_weight('Kernel',
                                      shape=(input_shape[-1], self.units),
                                      initializer=tf.keras.initializers.he_normal,
                                      dtype=tf.float32,
                                      trainable=True)
        self.bias = self.add_weight('Bias',
                                    shape=(self.units,),
                                    initializer=tf.keras.initializers.zeros,
                                    dtype=tf.float32,
                                    trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel) + self.bias

    # Optional
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'units': self.units}


# We could have also subclassed tf.keras.layers.Layer

class ResnetIdentityBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters):
        super(ResnetIdentityBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters

        self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
        self.bn2c = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        x += input_tensor
        return tf.nn.relu(x)


if __name__ == '__main__':
    layer = DenseLayer(32)
    X = tf.reshape(tf.range(10, dtype='float'), [5, 2])
    block = ResnetIdentityBlock(1, [1, 2, 3])
    print(layer(X))
    print(block.trainable_variables)
