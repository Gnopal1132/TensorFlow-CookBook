import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # Let's first simply do an image classification
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    NUM_CLASSES = 10

    # Converting into float values and Normalizing
    x_train = tf.divide(tf.cast(x_train, tf.float32), 255.0)
    x_test = tf.divide(tf.cast(x_test, tf.float32), 255.0)

    # x_train = x_train.astype('float32') / 255.
    # x_test = x_test.astype('float32') / 255.

    # Converting labels into categorical values, i.e. one hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    # print(x_train[0])
    # print(y_train)

    # Sequential Model
    sequential_model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(x_train.shape[1:])),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    # Functional Model
    input_layer = tf.keras.layers.Input(shape=(x_train.shape[1:]))
    flatten_layer = tf.keras.layers.Flatten()(input_layer)
    dense_layer = tf.keras.layers.Dense(128, activation='relu')(flatten_layer)
    dense_layer = tf.keras.layers.Dense(64, activation='relu')(dense_layer)
    output_layer = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(dense_layer)
    functional_model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # Compiling the model
    functional_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
                             loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                             metrics=['accuracy'])
    # history = functional_model.fit(x_train, y_train, batch_size=32,
    #                                epochs=10, validation_data=(x_test, y_test), shuffle=True)
    # pd.DataFrame(history.history).plot(figsize=(10, 5))
    # plt.grid(True)
    # plt.show()

    functional_model.evaluate(x_test, y_test, verbose=2)

    CLASSES = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
    prediction = functional_model.predict(x_test)
    prediction_labels = CLASSES[tf.argmax(prediction, axis=-1)]
    actual = CLASSES[tf.argmax(y_test, axis=-1)]


    def plot_predictions(rows, columns, x_test, prediction_labels):
        rows = rows
        columns = columns
        for row in range(rows):
            for column in range(columns):
                index = row * columns + column
                ax = plt.subplot(rows, columns, index + 1)
                label = str(prediction_labels[index])
                plt.imshow(x_test[index], cmap='gray')
                plt.title(label)
                plt.axis('off')

        plt.subplots_adjust(hspace=0.4, wspace=0.4)
        plt.show()


    # Let's solve it using CNN with BN and dropout #
    input_cnn = tf.keras.layers.Input(shape=(x_train.shape[1:]))
 
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(input_cnn)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    output = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = tf.keras.Model(inputs=input_cnn, outputs=output)
    print(model.summary())

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=32,
                        epochs=10, validation_data=(x_test, y_test), shuffle=True)
    pd.DataFrame(history.history).plot(figsize=(10, 5))
    plt.grid(True)
    plt.show()

    prediction = model.predict(x_test)
    labels = CLASSES[tf.argmax(prediction, axis=-1)]
    plot_predictions(5, 5, x_test, labels)
