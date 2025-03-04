import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd


def plot_dataset(x_train, y_train):
    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    n_rows = 5
    n_cols = 5
    plt.figure(figsize=(20, 20))
    for row in range(n_rows):
        for col in range(n_cols):
            index = row * n_cols + col
            plt.subplot(n_rows, n_cols, index + 1)
            plt.imshow(x_train[index], cmap='binary', interpolation='nearest')
            plt.title(class_names[y_train[index]], fontsize=12)
            plt.axis('off')
    plt.subplots_adjust(wspace=1.2, hspace=1.2)
    plt.show()


if __name__ == '__main__':
    # tf.debugging.set_log_device_placement(True)
    (x_train, y_train), (x_val, y_val) = tf.keras.datasets.fashion_mnist.load_data()

    # Printing the shape
    print(x_train.shape)
    print(y_train.shape)

    plot_dataset(x_train, y_train)
    # Let's create a sequential model
    """model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=x_train.shape[1:]),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10)
    ])"""
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=x_train.shape[1:]),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                        epochs=10)

    pd.DataFrame(history.history).plot(figsize=(10, 10))
    plt.grid()
    plt.show()
