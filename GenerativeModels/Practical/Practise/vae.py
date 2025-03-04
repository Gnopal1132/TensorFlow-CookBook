import datetime

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
K = tf.keras.backend


@tf.keras.utils.register_keras_serializable(package='SamplingLayer')
class SamplingLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        pass

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch_size = tf.shape(z_mean)[0]
        dimension = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch_size, dimension))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def get_config(self):
        base_config = super().get_config()
        return base_config


@tf.keras.utils.register_keras_serializable(package='VAE')
class VAE(tf.keras.models.Model):
    def __init__(self, input_shape, BETA=500, **kwargs):
        super().__init__(**kwargs)

        self.shape_before_flattening = None
        self.kl_metric = tf.keras.metrics.Mean(name='kl_loss')
        self.bce_metric = tf.keras.metrics.Mean(name='bce_loss')
        self.total_metric = tf.keras.metrics.Mean(name='total_loss')

        # Initializing Variables
        self.input_shape = input_shape

        # Some constants
        self.BETA = BETA

        # Initializing the Encoder and Decoder model.
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def build_encoder(self):
        encoder_input = tf.keras.Input(shape=self.input_shape, name='encoder_input')
        x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=2, padding='same', activation='relu')(encoder_input)
        x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=2, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=2, padding='same', activation='relu')(x)
        self.shape_before_flattening = K.int_shape(x)[1:]
        x = tf.keras.layers.Flatten()(x)
        z_mean = tf.keras.layers.Dense(2, name='z_mean')(x)
        z_log_var = tf.keras.layers.Dense(2, name='z_log_var')(x)
        z = SamplingLayer()([z_mean, z_log_var])
        encoder = tf.keras.Model(inputs=encoder_input, outputs=[z_mean, z_log_var, z])
        return encoder

    def build_decoder(self):
        decoder_input = tf.keras.Input(shape=(2,), name='decoder_input')
        x = tf.keras.layers.Dense(np.prod(self.shape_before_flattening), activation='relu')(decoder_input)
        x = tf.keras.layers.Reshape(self.shape_before_flattening)(x)
        x = tf.keras.layers.Conv2DTranspose(128, kernel_size=(3, 3), strides=2, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2DTranspose(64, kernel_size=(3, 3), strides=2, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2DTranspose(32, kernel_size=(3, 3), strides=2, padding='same', activation='relu')(x)
        decoder_out = tf.keras.layers.Conv2D(1, kernel_size=(3, 3), strides=1, padding='same', activation='sigmoid',
                                             name='decoder_output')(x)
        decoder = tf.keras.Model(inputs=decoder_input, outputs=decoder_out)
        return decoder

    @property
    def metrics(self):
        return [self.bce_metric, self.kl_metric, self.total_metric]

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructions = self.decoder(z)
        return z_mean, z_log_var, reconstructions

    def train_step(self, inputs):
        # Defining custom training loop
        with tf.GradientTape() as tape:
            # This will call the call() method
            z_mean, z_log_var, reconstruction = self(inputs)
            # Now we calculate the loss
            bce_loss = tf.reduce_mean(
                self.BETA * tf.keras.losses.binary_crossentropy(inputs, reconstruction, axis=(1, 2, 3))
            )
            kl_loss = tf.reduce_mean(
                tf.reduce_sum(
                    -0.5
                    * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)),
                    axis=1,
                )
            )
            total_loss = bce_loss + kl_loss

        gradients = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        self.bce_metric.update_state(bce_loss)
        self.kl_metric.update_state(kl_loss)
        self.total_metric.update_state(total_loss)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, inputs):
        """Step run during validation."""
        if isinstance(inputs, tuple):
            inputs = inputs[0]

        z_mean, z_log_var, reconstruction = self(inputs)
        reconstruction_loss = tf.reduce_mean(
            self.BETA
            * tf.keras.losses.binary_crossentropy(inputs, reconstruction, axis=(1, 2, 3))
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

    def get_config(self):
        base_config = super().get_config()
        base_config.update({'BETA': self.BETA, 'input_shape': self.input_shape})
        return base_config


def preprocess(image):
    image = image.astype('float32') / 255.
    image = np.pad(image, ((0, 0), (2, 2), (2, 2)), mode='constant', constant_values=0.)
    image = np.expand_dims(image, axis=-1)
    return image


def tensorboard_id(log_path):
    current_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return log_path + '/' + current_id


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = preprocess(x_train)
    x_test = preprocess(x_test)

    print('Training instance shape: {}, training label shape: {}'.format(x_train.shape, y_train.shape))
    print('Validation instance shape: {}, validation label shape: {}'.format(x_test.shape, y_test.shape))

    # Let's first define the optimizer
    adam = tf.keras.optimizers.Adam(learning_rate=5e-4)
    vae = VAE(input_shape=x_train.shape[1:])
    print(vae.summary())

    BATCH_SIZE = 128
    EPOCHS = 50
    vae.compile(optimizer=adam)

    checkpoint_path = os.path.join(os.curdir, 'checkpoints', 'celeb_vae_checkpoint.keras')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=False,
        save_freq="epoch",
        monitor="total_loss",
        mode="min",
        save_best_only=True,
        verbose=0,
    )
    tensorboard_path = os.path.join(os.curdir, 'TensorBoard', 'vae', 'logs')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_id(tensorboard_path))

    history = vae.fit(
        x_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        validation_data=(x_test, y_test),
        callbacks=[model_checkpoint_callback, tensorboard_callback],
    )
    # Showing the trajectory
    pd.DataFrame(history.history).plot(figsize=(8, 8))
    plt.grid(True)
    plt.show()

    # Let's load the model
    vae = tf.keras.models.load_model(checkpoint_path)
    encoder_model = vae.encoder
    decoder_model = vae.decoder

    plots = os.path.join(os.curdir, 'plots')
    # Let's first see the embedding space
    # z_mean, z_log_var, z = encoder_model.predict(x_test)
    # plt.figure(figsize=(8, 8))
    # plt.scatter(z[:, 0], z[:, 1], c=y_test, s=3, alpha=0.5)
    # plt.colorbar()
    # plt.axis('off')
    # plt.title('Test Embedding latent space')
    # plt.savefig(os.path.join(plots, 'test_embedding_space.png'))
    # plt.show()
    # plt.close()

    # Let's generate new images
    new_embedding_samples = np.random.normal(size=(25, 2))
    new_samples = decoder_model.predict(new_embedding_samples)
    # Step 4: Plot them
    plt.figure(figsize=(10, 8))
    for i in range(25):
        sample_image = new_samples[i]
        plt.subplot(5, 5, i + 1)
        plt.imshow(sample_image, cmap='gray')
        plt.axis('off')
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.savefig(os.path.join(plots, 'new_samples.png'))
    plt.suptitle('Generative Samples VAE')
    plt.show()
    plt.close()
