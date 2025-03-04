import tensorflow as tf

generator_input = tf.keras.Input(shape=(128,), name='generator_input')
reshape_input = tf.keras.layers.Reshape((1, 1, 128))(generator_input)
x = tf.keras.layers.Conv2DTranspose(512, 4, strides=1, padding='valid', use_bias=False)(reshape_input)
x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
x = tf.keras.layers.LeakyReLU(0.2)(x)
x = tf.keras.layers.Conv2DTranspose(
    256, kernel_size=4, strides=2, padding="same", use_bias=False
)(x)
x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
x = tf.keras.layers.LeakyReLU(0.2)(x)
x = tf.keras.layers.Conv2DTranspose(
    128, kernel_size=4, strides=2, padding="same", use_bias=False
)(x)
x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
x = tf.keras.layers.LeakyReLU(0.2)(x)
x = tf.keras.layers.Conv2DTranspose(
    64, kernel_size=4, strides=2, padding="same", use_bias=False
)(x)
x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
x = tf.keras.layers.LeakyReLU(0.2)(x)
generator_output = tf.keras.layers.Conv2DTranspose(
    3, kernel_size=4, strides=2, padding="same", activation="tanh"
)(x)
generator_model = tf.keras.Model(generator_input, generator_output)


print(generator_model.summary())
input = tf.random.uniform((1, 128))
output = generator_model(input)
print(output.shape)