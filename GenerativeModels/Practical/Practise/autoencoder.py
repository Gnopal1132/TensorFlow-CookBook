import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

K = tf.keras.backend
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def preprocess(image):
    image = image.astype('float32') / 255.

    # Padding image to 32x32. Understand this
    image = np.pad(image, ((0, 0), (2, 2), (2, 2)), mode='constant', constant_values=0.)
    image = np.expand_dims(image, axis=-1)
    return image


def create_validation_set(x_set, y_set, train_split=0.8):
    indices = np.arange(x_set.shape[0])
    train_length = int(train_split * x_set.shape[0])
    np.random.shuffle(indices)

    x_train = x_set[indices[:train_length]]
    y_train = y_set[indices[:train_length]]
    x_val = x_set[indices[train_length:]]
    y_val = y_set[indices[train_length:]]

    return x_train, y_train, x_val, y_val


def plot_reconstruction(prediction, target, n_rows=5, n_cols=5):
    plt.figure(figsize=(n_cols * 2, n_rows * 2))
    for i in range(n_rows * n_cols):
        # Plot Actual Image
        plt.subplot(n_rows, n_cols * 2, 2 * i + 1)  # 2*i + 1 for the left image
        plt.imshow(target[i], cmap='gray')
        plt.title("Actual")
        plt.axis('off')

        # Plot Predicted Image
        plt.subplot(n_rows, n_cols * 2, 2 * i + 2)  # 2*i + 2 for the right image
        plt.imshow(prediction[i], cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), 'plots', 'reconstructions.png'))
    plt.show()
    plt.close()


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = preprocess(x_train)
    x_test = preprocess(x_test)

    x_train, y_train, x_val, y_val = create_validation_set(x_train, y_train)
    print('Training instance shape: {}, training label shape: {}'.format(x_train.shape, y_train.shape))
    print('Validation instance shape: {}, validation label shape: {}'.format(x_val.shape, y_val.shape))

    # Let's create the encoder
    encoder_input = tf.keras.Input(shape=x_train.shape[1:], name='encoder_input')
    x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=2, padding='same', activation='relu')(encoder_input)
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=2, padding='same', activation='relu')(x)
    shape_before_flattening = K.int_shape(x)[1:]

    x = tf.keras.layers.Flatten()(x)
    encoder_out = tf.keras.layers.Dense(2, name='encoder_out')(x)
    encoder_model = tf.keras.Model(inputs=encoder_input, outputs=encoder_out)
    print(encoder_model.summary())

    # Decoder
    decoder_input = tf.keras.Input(shape=(2,), name='decoder_input')
    x = tf.keras.layers.Dense(np.prod(shape_before_flattening), activation='relu')(decoder_input)
    x = tf.keras.layers.Reshape(shape_before_flattening)(x)
    x = tf.keras.layers.Conv2DTranspose(128, kernel_size=(3, 3), strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(64, kernel_size=(3, 3), strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(32, kernel_size=(3, 3), strides=2, padding='same', activation='relu')(x)
    decoder_out = tf.keras.layers.Conv2D(1, kernel_size=3, strides=1, padding='same', activation='sigmoid',
                                         name='decoder_output')(x)
    decoder = tf.keras.Model(decoder_input, decoder_out)

    autoencoder = tf.keras.Model(encoder_input, decoder(encoder_out))

    print(autoencoder.summary())

    # autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    # history = autoencoder.fit(x_train, x_train, epochs=20, validation_data=(x_val, x_val), shuffle=True, batch_size=128)
    #
    model_path = os.path.join(os.getcwd(), 'saved_model', 'autoencoder.keras')
    # autoencoder.save(model_path)
    #
    # # Showing the trajectory
    # pd.DataFrame(history.history).plot(figsize=(8, 8))
    # plt.grid(True)
    # plt.savefig(os.path.join(os.getcwd(), 'plots', 'history_autoencoder.png'))
    # plt.show()

    autoencoder = tf.keras.models.load_model(model_path)
    # Let's checkout the encodings for x_test

    # This is the correct way to get the output from a layer.
    # encoder_model = autoencoder.get_layer('encoder_out')
    # encoder_model = tf.keras.Model(autoencoder.input, encoder_model.output)
    #
    test_predict = autoencoder.predict(x_test)
    test_encoding = encoder_model.predict(x_test)
    # print(test_predict.shape)
    # print(test_encoding.shape)
    #
    # plot_reconstruction(test_predict[:25], x_test[:25], n_rows=5, n_cols=5)
    #
    # plt.Figure(figsize=(8, 8))
    # plt.scatter(test_encoding[:, 0], test_encoding[:, 1], c=y_test, s=3, alpha=0.5)
    # plt.colorbar()
    # plt.grid(True)
    # plt.savefig(os.path.join(os.getcwd(), 'plots', 'test_encoder_embedding_space.png'))
    # plt.show()
    # plt.close()

    # Generate new images

    for layer in autoencoder.layers:
        print(layer.name)  # decoder full model named as functional_1

    minimum, maximum = np.min(test_encoding), np.max(test_encoding)
    new_embedding_samples = np.random.uniform(low=minimum, high=maximum, size=(25, 2))

    # We don't have to get the layers in this because it is the full decoder model in itself.
    decoder = autoencoder.get_layer('functional_1')
    print(type(decoder))

    new_samples = decoder.predict(new_embedding_samples)
    # Step 4: Plot them
    plt.figure(figsize=(10, 8))
    for i in range(25):
        sample_image = new_samples[i]
        plt.subplot(5, 5, i + 1)
        plt.imshow(sample_image, cmap='gray')
        plt.axis('off')
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.savefig(os.path.join(os.getcwd(), 'plots', 'new_samples.png'))
    plt.show()
    plt.close()

    # To store everything separately:
    """
    # Save the encoder, decoder, and autoencoder weights separately
    encoder_model.save_weights('encoder_weights.h5')
    decoder.save_weights('decoder_weights.h5')
    autoencoder.save_weights('autoencoder_weights.h5')

    # Reinitialize models with the same architecture, i.e. define the model architecture
    # Make sure the architecture of the models matches exactly!
    
    encoder_model = tf.keras.Model(encoder_input, encoder_out)
    decoder = tf.keras.Model(decoder_input, decoder_out)
    autoencoder = tf.keras.Model(encoder_input, decoder(encoder_out))
    
    # Load the weights
    encoder_model.load_weights('encoder_weights.h5')
    decoder.load_weights('decoder_weights.h5')
    autoencoder.load_weights('autoencoder_weights.h5')

    """
