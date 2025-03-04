import tensorflow as tf
import os
import pandas as pd
import numpy as np


def read_image(image_address, label):
    image = tf.io.read_file(image_address)
    image = tf.image.decode_jpeg(image)
    image = tf.cast(image, dtype=tf.float32)
    return image, label


if __name__ == '__main__':

    # Scenario 1: Where the lego_dataset is present in structured format
    # Cat ---> cat images, Dog ---> Dog images, and root folder = [Dog, Cat]

    root_path = os.path.join(os.curdir, 'lego_dataset', 'cats_and_dogs')

    # Let's read the lego_dataset
    BATCH = 32
    IMG_HEIGHT = IMG_WIDTH = 224
    BUFFER = 1000
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train = tf.keras.preprocessing.image_dataset_from_directory(
        root_path,
        batch_size=BATCH,
        labels='inferred',
        label_mode='binary',  # int for multiclass, 'categorical' means that the labels are encoded as a categorical
        # vector (e.g. for categorical_cross-entropy loss).
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        validation_split=0.2,
        subset='training'
    )
    val = tf.keras.preprocessing.image_dataset_from_directory(
        root_path,
        batch_size=BATCH,
        labels='binary',
        label_mode='int',
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        validation_split=0.2,
        subset='validation'
    )

    train = train.cache().shuffle(BUFFER).prefetch(AUTOTUNE)
    val = val.cache().prefetch(AUTOTUNE)

    for img, label in train.take(1):
        print(img.shape)
        print(label.shape)

    # Scenario 2: Manually reading the datapoints from folder
    # and say labels from a CSV file
    path = os.path.join(os.curdir, 'Custom Dataset')
    dataframe = pd.read_csv(os.path.join(path, 'cats_dogs.csv'))

    address_name = dataframe.values[:, 0]
    labels = dataframe.values[:, 1].astype(np.int32)

    dataset = tf.data.Dataset.from_tensor_slices((address_name, labels))


    def join_string(name_file, label):
        new_string = tf.strings.join([os.path.join(path, 'cats_dogs_resized'), name_file],
                                     separator=os.sep)
        return new_string, label


    dataset = dataset.map(join_string, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(read_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache().shuffle(buffer_size=100).batch(4)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

