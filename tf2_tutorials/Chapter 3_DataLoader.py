import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    train = train.cache().shuffle(buffer_size=100).batch(32).prefetch(tf.data.AUTOTUNE)
    test = test.cache().shuffle(buffer_size=100).batch(32).prefetch(tf.data.AUTOTUNE)

    for img, label in train.take(1):
        print(img.shape)
        print(label.shape)
        