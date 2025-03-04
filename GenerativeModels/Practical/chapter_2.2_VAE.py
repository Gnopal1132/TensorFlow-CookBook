import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import keras

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


# Sampling the latent variable
class Sampling(tf.keras.layers.Layer):
    # Optional
    def __init__(self, **kwargs):
        super(Sampling, self).__init__(**kwargs)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        # tf.exp(), we use of numerical stability, and we avoid the negative values for sigma.
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


if __name__ == '__main__':
    # We will first encode AutoEncoder
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = preprocess(x_train)
    x_test = preprocess(x_test)

    encoder_input = tf.keras.layers.Input(shape=x_train.shape[1:], name='encoder_input')
    x = tf.keras.layers.Conv2D(32, (3, 3), strides=2, padding='same', activation='relu')(encoder_input)
    x = tf.keras.layers.Conv2D(64, (3, 3), strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), strides=2, padding='same', activation='relu')(x)
    shape_before_flattening = K.int_shape(x)[1:]

    x = tf.keras.layers.Flatten()(x)
    z_mean = tf.keras.layers.Dense(2, name='z_mean')(x)
    z_log_var = tf.keras.layers.Dense(2, name='z_log_var')(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = tf.keras.Model(encoder_input, [z_mean, z_log_var, z], name='encoder')
    print(encoder.summary())

    decoder_input = tf.keras.Input(shape=(2, ), name='decoder_input')
    x = tf.keras.layers.Dense(np.prod(shape_before_flattening), activation='relu')(decoder_input)
    x = tf.keras.layers.Reshape(shape_before_flattening)(x)
    x = tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)
    decoder_output = tf.keras.layers.Conv2D(1, 3, strides=1, padding='same', activation='sigmoid',
                                            name='decoder_output')(x)
    decoder = tf.keras.Model(decoder_input, decoder_output)

    print(decoder.summary())

    # Some defaults
    BETA = 500
    IMAGE_SIZE = 32
    BATCH_SIZE = 100
    VALIDATION_SPLIT = 0.2
    EMBEDDING_DIM = 2
    EPOCHS = 5


    """
    1. Firstly explain me the loss fucntion here in detail. Why KL divergence and cross entropy both are used.
    2. Explain the axis part while calculating the loss function. How its interpreted with a small example ok
    3. What does this update_Stat does? How can i use it for myself? Is it just for metrics? When to use it?
    4. Explain the output {m.name: m.result() for m in self.metrics}, why it is the way it is. 
    5. Why defining another function metrics() as property??? What is @property here means?
    6. while defining a custom model. We need to define a call function like we do in layers. Why and what is this
       train_Step means?? why it is here? the significance of it? why are we defining the loss inside? How can i use it
       for myself?
    7. Like in genereal I wonder, sometimes other use the keras backend (i.e. K =tf.keras.backend), and perform operation
    like K.int_shape() (in encoder), why cant we use tf.shape(). How do i know when to use K and when to use tf??. Similarly, there
    is tf.keras.losses.binary_crossentropy and tf.keras.losses.BinaryCrossentropy. Why two versions?? Whats the difference?
    Which one to use and when?
    
    """

    class VAE(tf.keras.Model):
        def __init__(self, encoder, decoder, **kwargs):
            super(VAE, self).__init__(**kwargs)
            self.encoder = encoder
            self.decoder = decoder
            self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
            self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
                name="reconstruction_loss"
            )
            self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

        @property
        def metrics(self):
            return [
                self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker,
            ]

        def call(self, inputs):
            """Call the model on a particular input."""
            z_mean, z_log_var, z = encoder(inputs)
            reconstruction = decoder(z)
            return z_mean, z_log_var, reconstruction

        def train_step(self, data):
            """Step run during training."""
            with tf.GradientTape() as tape:
                # self(data) will call the __call__
                z_mean, z_log_var, reconstruction = self(data)
                reconstruction_loss = tf.reduce_mean(
                    BETA
                    * tf.keras.losses.binary_crossentropy(
                        data, reconstruction, axis=(1, 2, 3)
                    )
                )
                kl_loss = tf.reduce_mean(
                    tf.reduce_sum(
                        -0.5
                        * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)),
                        axis=1,
                    )
                )
                total_loss = reconstruction_loss + kl_loss

            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

            self.total_loss_tracker.update_state(total_loss)
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            self.kl_loss_tracker.update_state(kl_loss)

            return {m.name: m.result() for m in self.metrics}

        def test_step(self, data):
            """Step run during validation."""
            if isinstance(data, tuple):
                data = data[0]

            z_mean, z_log_var, reconstruction = self(data)
            reconstruction_loss = tf.reduce_mean(
                BETA
                * tf.keras.losses.binary_crossentropy(data, reconstruction, axis=(1, 2, 3))
            )
            kl_loss = tf.reduce_mean(
                tf.reduce_sum(
                    -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)),
                    axis=1,
                )
            )
            total_loss = reconstruction_loss + kl_loss

            return {
                "loss": total_loss,
                "reconstruction_loss": reconstruction_loss,
                "kl_loss": kl_loss,
            }


    # Create a variational autoencoder
    vae = VAE(encoder, decoder)
    print(vae.summary())

    # Compile the variational autoencoder
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    vae.compile(optimizer=optimizer)

    # Create a model save checkpoint
    checkpoint_path = os.path.join(os.curdir, 'checkpoints', 'celeb_vae_checkpoint.keras')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=False,
        save_freq="epoch",
        monitor="loss",
        mode="min",
        save_best_only=True,
        verbose=0,
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")

    history = vae.fit(
        x_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        validation_data=(x_test, x_test),
        callbacks=[model_checkpoint_callback, tensorboard_callback],
    )
    # Showing the trajectory
    pd.DataFrame(history.history).plot(figsize=(8, 8))
    plt.grid(True)
    plt.show()

    model_path = os.path.join(os.getcwd(), 'models')
    # Save the final models
    vae.save(os.path.join(model_path, 'vae.keras'))
    encoder.save(os.path.join(model_path, 'vae_encoder.keras'))
    decoder.save(os.path.join(model_path, 'vae_decoder.keras'))
