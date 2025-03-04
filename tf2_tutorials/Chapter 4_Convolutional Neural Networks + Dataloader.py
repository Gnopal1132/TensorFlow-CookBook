import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf


class Dataloader:
    def __init__(self, data_path: str, absolute_path: str):
        self.path = data_path
        self.abs_path = absolute_path
        self.images = None
        self.labels = None

        self.load_absolute_image_path()

    def load_absolute_image_path(self):
        dataframe = pd.read_csv(self.path)
        self.images = dataframe.values[:, 0]
        self.labels = dataframe.values[:, 1].astype(np.int32)

    def return_absolute_path(self, instance, label):
        instance = tf.strings.join([self.abs_path, instance], separator=os.sep)
        return instance, tf.cast(label, tf.int8)

    @staticmethod
    def load_image(instance, label):
        image = tf.io.read_file(instance)
        image = tf.image.decode_jpeg(image)
        return image, label

    def return_loader(self, batch_size=4, buffer_size=1000, parallel_calls=tf.data.AUTOTUNE):
        dataset = tf.data.Dataset.from_tensor_slices((self.images, self.labels))
        dataset = dataset.map(self.return_absolute_path, num_parallel_calls=parallel_calls)
        dataset = dataset.map(self.load_image, num_parallel_calls=parallel_calls)
        dataset = dataset.cache().shuffle(buffer_size).batch(batch_size, num_parallel_calls=parallel_calls)

        return dataset.prefetch(parallel_calls)


class ConvolutionalNeuralNetwork:
    def __init__(self, input_shape: tuple, classes: int, get_logits: bool = False):
        self.input_shape = input_shape
        self.classes = classes
        self.get_logits = get_logits

    def return_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=self.input_shape),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
        ])
        if not self.get_logits:
            model.add(tf.keras.layers.Dense(self.classes, activation='sigmoid'))
        else:
            model.add(tf.keras.layers.Dense(self.classes))

        return model


if __name__ == '__main__':
    path = os.path.join(os.curdir, 'Custom Dataset')
    dataloader = Dataloader(data_path=os.path.join(path, 'cats_dogs.csv'),
                            absolute_path=os.path.join(path, 'cats_dogs_resized'))

    # Loading the dataloader
    train_loader = dataloader.return_loader()
    for image, label in train_loader.take(1):
        print(image.shape)
        print(label.shape)

    conv_model = ConvolutionalNeuralNetwork(input_shape=(224, 224, 3), classes=1, get_logits=True)
    model = conv_model.return_model()

    # Compile the model
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])
    history = model.fit(train_loader, epochs=10)

    plt.Figure(figsize=(10, 10))
    plt.plot(pd.DataFrame(history.history))
    plt.axis('off')
    plt.grid(True)
    plt.show()
