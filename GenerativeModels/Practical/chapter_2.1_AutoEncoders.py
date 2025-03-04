import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

K = tf.keras.backend
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def preprocess(image):
    image = image.astype('float32') / 255.
    # We are padding so that we can make it 32x32, for easy processing
    image = np.pad(image, ((0, 0), (2, 2), (2, 2)), mode='constant', constant_values=0.0)
    image = tf.expand_dims(image, -1)
    return image


def plot_reconstructions(predictions, actual, n_rows=4, n_cols=4):
    """Plots actual and predicted images side-by-side."""
    plt.figure(figsize=(n_cols * 2, n_rows * 2))

    for i in range(n_rows * n_cols):
        # Plot Actual Image
        plt.subplot(n_rows, n_cols * 2, 2 * i + 1)  # 2*i + 1 for the left image
        plt.imshow(actual[i], cmap='gray')
        plt.title("Actual")
        plt.axis('off')

        # Plot Predicted Image
        plt.subplot(n_rows, n_cols * 2, 2 * i + 2)  # 2*i + 2 for the right image
        plt.imshow(predictions[i], cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # We will first encode AutoEncoder
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = preprocess(x_train)
    x_test = preprocess(x_test)

    encoder_input = tf.keras.Input(shape=x_train.shape[1:])
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')(encoder_input)
    x = tf.keras.layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(128, 3, strides=2, padding='same', activation='relu')(x)
    shape_before_flattening = K.int_shape(x)[1:]

    x = tf.keras.layers.Flatten()(x)
    encoder_output = tf.keras.layers.Dense(2, name='encoder_output')(x)

    encoder_model = tf.keras.Model(encoder_input, encoder_output)
    print(encoder_model.summary())

    # Decoder being just the opposite
    decoder_input = tf.keras.Input(shape=(2,), name='decoder_input')
    x = tf.keras.layers.Dense(np.prod(shape_before_flattening), activation='relu')(decoder_input)
    x = tf.keras.layers.Reshape(shape_before_flattening)(x)
    x = tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)
    decoder_output = tf.keras.layers.Conv2D(1, 3, strides=1, padding='same', activation='sigmoid',
                                            name='decoder_output')(x)
    decoder = tf.keras.Model(decoder_input, decoder_output)
    print(decoder.summary())

    autoencoder = tf.keras.Model(encoder_input, decoder(encoder_output))
    print(autoencoder.summary())

    # The loss function is between the individual pixels of the original image and the reconstruction.
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    history = autoencoder.fit(x_train, x_train, epochs=20, batch_size=128,
                              shuffle=True, validation_data=(x_test, x_test))

    model_path = os.path.join(os.getcwd(), 'saved_model', 'autoencoder.h5')
    # Save the model
    autoencoder.save(model_path)

    # Showing the trajectory
    pd.DataFrame(history.history).plot(figsize=(8, 8))
    plt.grid(True)
    plt.show()

    # Reconstructing images
    example_image = x_test[:5000]
    # Getting the latent space variables
    predictions_latent_variables = encoder_model.predict(example_image)

    # Let's see the latent space variable for the variables
    sample_latent_variables = predictions_latent_variables[:10]
    sample_labels = y_test[:10]

    for latent_vector, label in zip(sample_latent_variables, sample_labels):
        print('For the label {} The latent space vector is: {}'.format(label, latent_vector))

    # Let's see some predictions
    predictions = autoencoder.predict(example_image)
    # plot_reconstructions(predictions, example_image)

    # Let's plot the latent space.
    plt.figure(figsize=(8, 8))
    plt.scatter(predictions_latent_variables[:, 0], predictions_latent_variables[:, 1], c=y_test[:5000], alpha=0.5, s=3)
    plt.colorbar()
    plt.grid(True)
    plt.show()

    # Let's Generate New Images
    # Step 1: Get the range
    minimum, maximum = np.min(predictions_latent_variables), np.max(predictions_latent_variables)
    # Step 2: Sample from this range. Let's sample 18 such images
    sample = np.random.uniform(low=minimum, high=maximum, size=(18, 2))

    # Step 3: Do the Prediction
    reconstruction = decoder.predict(sample)

    # Step 4: Plot them
    plt.figure(figsize=(10, 8))
    for i in range(18):
        sample_image = reconstruction[i]
        plt.subplot(3, 6, i+1)
        plt.imshow(sample_image, cmap='gray')
        plt.axis('off')
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.show()

