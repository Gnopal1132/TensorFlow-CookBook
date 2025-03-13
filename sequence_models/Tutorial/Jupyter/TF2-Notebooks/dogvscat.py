import matplotlib.pyplot as plt
import tensorflow as tf
import os
import pandas as pd

if __name__ == '__main__':
    root_path = os.path.join(os.curdir, 'lego_dataset', 'cats_and_dogs')

    # Let's read the lego_dataset
    BATCH = 32
    IMG_HEIGHT = IMG_WIDTH = 224
    BUFFER = 1000
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(root_path, 'train'),
        batch_size=BATCH,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
    )
    val = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(root_path, 'validation'),
        batch_size=BATCH,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
    )

    train = train.cache().shuffle(BUFFER).prefetch(AUTOTUNE)
    val = val.cache().prefetch(AUTOTUNE)

    for img, label in train.take(1):
        print(img.shape)
        print(label.shape)

    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(factor=0.2),
        tf.keras.layers.RandomContrast(factor=0.4)
    ])

    # Always use rescaling layer instead of manually dividing.
    rescaled = tf.keras.layers.Rescaling(1. /255)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        rescaled,
        augmentation,
        tf.keras.layers.Conv2D(16, 3, padding='SAME'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='SAME'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='SAME'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    print(model.summary())

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy'])

    earlystop = tf.keras.callbacks.EarlyStopping(patience=15)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('best_classifier/', save_best_only=True)

    history = model.fit(train, validation_data=val, epochs=50, callbacks=[earlystop, checkpoint])

    pd.DataFrame(history.history).plot(figsize=(10, 10))
    plt.grid()
    plt.show()
    # Here the model achieves validation accuracy greater than 70% if ran for number of epochs

    # Let's try transfer learning, Always include the image shape as well.
    basemodel = tf.keras.applications.MobileNetV2(include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    basemodel.trainable = False
    basemodel.summary()

    preprocess_layer = tf.keras.applications.mobilenet_v2.preprocess_input
    # Note this preprocess layer wants input in range [0,255]. That's why we used rescaling layer above instead of
    # manually dividing it. But here the preprocess_layer will do that for ya!

    # First Augment the image and then apply preprocess_layer on it.

    NUM_CLASSES = 1
    input_layer = tf.keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = augmentation(input_layer)
    x = preprocess_layer(x)
    x = basemodel(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    out = tf.keras.layers.Dense(NUM_CLASSES)(x)

    model = tf.keras.Model(inputs=input_layer, outputs=out)

    print(model.summary())

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  metrics=['accuracy'])

    earlystop = tf.keras.callbacks.EarlyStopping(patience=15)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('best_classifier/', save_best_only=True)

    history = model.fit(train, validation_data=val, epochs=50, callbacks=[earlystop, checkpoint])

    pd.DataFrame(history.history).plot(figsize=(10, 10))
    plt.grid()
    plt.show()


