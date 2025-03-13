import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

tf.keras.backend.clear_session()

if __name__ == '__main__':
    (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.fashion_mnist.load_data()

    print(X_train.shape)
    print(Y_train.shape)

    # Let's first split the lego_dataset
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2)

    train = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    test = tf.data.Dataset.from_tensor_slices(X_test)
    val = tf.data.Dataset.from_tensor_slices((X_val, Y_val))

    autotune = tf.data.AUTOTUNE
    train = train.cache().shuffle(1000).batch(32).prefetch(autotune)
    val = val.cache().batch(32).prefetch(autotune)
    test = test.cache().batch(32).prefetch(autotune)

    num_classes = 10
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=[28, 28]),
        tf.keras.layers.Rescaling(1. / 255),
        tf.keras.layers.Dense(100, activation='elu', kernel_initializer='he_normal'),
        tf.keras.layers.Dense(num_classes)
    ])

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer='adam',
                  metrics=['accuracy'])
    earlystop = tf.keras.callbacks.EarlyStopping(patience=10)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('best_mnist/', save_best_only=True)

    # history = model.fit(train, validation_data=val, epochs=100, callbacks=[earlystop, checkpoint])
    model = tf.keras.models.load_model('best_mnist/')

    prediction = model.predict(test)
    output_class = np.argmax(prediction, axis=1)
    print(tf.keras.metrics.Accuracy()(Y_test, output_class))
