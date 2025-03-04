import numpy as np
import tensorflow as tf
import os
import pandas as pd
import matplotlib.pyplot as plt


def preprocess_image(image):
    return (tf.cast(image, tf.float32) - 127.5) / 127.5


@tf.keras.utils.register_keras_serializable(package='WGAN')
class WGAN(tf.keras.models.Model):
    def __init__(self, image_size, channels, critic_steps, gp_weight,
                 latent_dim=100, **kwargs):
        super(WGAN, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.channels = channels
        self.critic_steps = critic_steps
        self.gp_weight = gp_weight

        # critic and generator model
        self.critic = self.build_critic()
        self.generator = self.build_generator()

        # Metrics to store
        self.c_wass_loss_metric = tf.keras.metrics.Mean(name="c_wass_loss")
        self.c_gp_metric = tf.keras.metrics.Mean(name="c_gp")
        self.c_loss_metric = tf.keras.metrics.Mean(name="c_loss")
        self.g_loss_metric = tf.keras.metrics.Mean(name="g_loss")

    def build_critic(self):
        critic_input = tf.keras.Input(shape=self.image_size, name='critic_input')
        x = tf.keras.layers.Conv2D(32, 4, strides=2, padding='same')(critic_input)
        x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
        x = tf.keras.layers.Dropout(0.4)(x)

        x = tf.keras.layers.Conv2D(64, 4, strides=2, padding='same')(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
        x = tf.keras.layers.Dropout(0.4)(x)

        x = tf.keras.layers.Conv2D(128, 4, strides=2, padding='same')(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
        x = tf.keras.layers.Dropout(0.4)(x)

        x = tf.keras.layers.Conv2D(256, 4, strides=2, padding='same')(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
        x = tf.keras.layers.Dropout(0.4)(x)

        x = tf.keras.layers.Conv2D(1, kernel_size=4, strides=1, padding='valid')(x)
        x = tf.keras.layers.Flatten()(x)
        critic_output = tf.keras.layers.Dense(1)(x)

        critic_model = tf.keras.Model(inputs=critic_input, outputs=critic_output)
        return critic_model

    def build_generator(self):
        generator_input = tf.keras.Input(shape=(self.latent_dim,), name='generator_input')
        x = tf.keras.layers.Reshape((1, 1, self.latent_dim))(generator_input)
        x = tf.keras.layers.Conv2DTranspose(256, 4, strides=1, padding='valid', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = tf.keras.layers.Conv2DTranspose(
            128, kernel_size=4, strides=2, padding="same", use_bias=False
        )(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = tf.keras.layers.Conv2DTranspose(
            64, kernel_size=4, strides=2, padding="same", use_bias=False
        )(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = tf.keras.layers.Conv2DTranspose(
            32, kernel_size=4, strides=2, padding="same", use_bias=False
        )(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        generator_output = tf.keras.layers.Conv2DTranspose(
            self.channels, kernel_size=4, strides=2, padding="same", activation="tanh"
        )(x)
        generator_model = tf.keras.Model(generator_input, generator_output)
        return generator_model

    @property
    def metrics(self):
        return [
            self.c_loss_metric,
            self.c_wass_loss_metric,
            self.c_gp_metric,
            self.g_loss_metric,
        ]

    def gradient_penalty(self, batch_size, real_images, fake_images):
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        difference = real_images - fake_images
        interpolation = real_images + alpha * difference

        with tf.GradientTape() as tape:
            tape.watch(interpolation)
            prediction = self.critic(interpolation, training=True)

        gradients = tape.gradient(prediction, [interpolation])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def compile(self, g_optimizer, c_optimizer):
        super(WGAN, self).compile()
        self.g_optimizer = g_optimizer
        self.c_optimizer = c_optimizer

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]

        for i in range(self.critic_steps):
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )

            with tf.GradientTape() as tape:
                fake_images = self.generator(
                    random_latent_vectors, training=True
                )
                fake_predictions = self.critic(fake_images, training=True)
                real_predictions = self.critic(real_images, training=True)

                c_wass_loss = tf.reduce_mean(fake_predictions) - tf.reduce_mean(
                    real_predictions
                )
                c_gp = self.gradient_penalty(
                    batch_size, real_images, fake_images
                )
                c_loss = c_wass_loss + c_gp * self.gp_weight

            c_gradient = tape.gradient(c_loss, self.critic.trainable_variables)
            self.c_optimizer.apply_gradients(
                zip(c_gradient, self.critic.trainable_variables)
            )

        random_latent_vectors = tf.random.normal(
            shape=(batch_size, self.latent_dim)
        )
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_latent_vectors, training=True)
            fake_predictions = self.critic(fake_images, training=True)
            g_loss = -tf.reduce_mean(fake_predictions)

        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )

        self.c_loss_metric.update_state(c_loss)
        self.c_wass_loss_metric.update_state(c_wass_loss)
        self.c_gp_metric.update_state(c_gp)
        self.g_loss_metric.update_state(g_loss)

        return {m.name: m.result() for m in self.metrics}

    def get_config(self):
        base_config = super().get_config()

        # Here how to save the optimizer setting
        g_optimizer_config = self.g_optimizer.get_config()
        c_optimizer_config = self.c_optimizer.get_config()

        base_config.update({
            "critic_steps": self.critic_steps,
            "latent_dim": self.latent_dim,
            "image_size": self.image_size,
            "channels": self.channels,
            "gp_weight": self.gp_weight,
            "g_optimizer_config": g_optimizer_config,
            "c_optimizer_config": c_optimizer_config,
        })
        return base_config

    @classmethod
    def from_config(cls, config):
        # Remove optimizer-related keys from the config
        config_copy = config.copy()
        g_optimizer_config = config_copy.pop('g_optimizer_config', None)
        c_optimizer_config = config_copy.pop('c_optimizer_config', None)

        # Re-create the model with the remaining configuration
        model = super(WGAN, cls).from_config(config_copy)

        # Re-create the optimizers from the saved configuration
        g_optimizer = tf.keras.optimizers.Adam.from_config(g_optimizer_config)
        c_optimizer = tf.keras.optimizers.Adam.from_config(c_optimizer_config)

        # Re-compile the model with the optimizers
        model.compile(g_optimizer=g_optimizer, c_optimizer=c_optimizer)

        return model


class ImageGenerator(tf.keras.callbacks.Callback):
    def __init__(self, latent_dim, batch_size):
        self.latent_dim = latent_dim
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        noise_normal = tf.random.normal(shape=(self.batch_size, self.latent_dim))
        prediction = self.model.generator.predict(noise_normal)
        prediction = prediction * 127.5 + 127.5
        for i in range(self.batch_size):
            image = tf.keras.utils.array_to_img(prediction[i])
            image.save(os.path.join(os.curdir, 'checkpoint_decoded_image', 'wgan_images', f'decoded_{epoch}_{i}.jpg'))


if __name__ == '__main__':
    IMAGE_SIZE = (64, 64)
    BATCH_SIZE = 128

    data_path = os.path.join(os.curdir, 'dataset', 'lego_dataset')
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_path,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        seed=123,
        interpolation='bilinear',
        labels=None,
        shuffle=True,
        color_mode='grayscale'
    )
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    for instance in dataset.take(1):
        print(instance.shape)
    # Hyperparameters
    IMAGE_SIZE = (64, 64)
    CHANNELS = 1
    BATCH_SIZE = 512
    NUM_FEATURES = 64
    Z_DIM = 128
    LEARNING_RATE = 0.0002
    ADAM_BETA_1 = 0.5
    ADAM_BETA_2 = 0.999
    EPOCHS = 200
    CRITIC_STEPS = 3
    GP_WEIGHT = 10.0
    LOAD_MODEL = False
    ADAM_BETA_1 = 0.5
    ADAM_BETA_2 = 0.9
    # # Create a GAN
    wgang = WGAN(
        image_size=IMAGE_SIZE + (1,),
        latent_dim=Z_DIM,
        critic_steps=CRITIC_STEPS,
        gp_weight=GP_WEIGHT,
        channels=CHANNELS
    )
    c_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=ADAM_BETA_1, beta_2=ADAM_BETA_2)
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=ADAM_BETA_1, beta_2=ADAM_BETA_2)
    wgang.compile(g_optimizer=g_optimizer, c_optimizer=c_optimizer)
    decoded_image_checkpoint = ImageGenerator(latent_dim=Z_DIM, batch_size=10)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(os.getcwd(), 'checkpoints', 'wgan_lesson_3_1_checkpoint.keras'),
        save_weights_only=False,
        save_freq="epoch",
        monitor="g_loss",
        mode="min",
        save_best_only=True,
        verbose=0,
    )
    history = wgang.fit(
        dataset,
        epochs=300,
        callbacks=[model_checkpoint_callback,
                   decoded_image_checkpoint], )
    # Showing the trajectory
    pd.DataFrame(history.history).plot(figsize=(8, 8))
    plt.grid(True)
    plt.show()

    wgan_model = tf.keras.models.load_model(os.path.join(os.curdir, 'checkpoints', 'wgan_lesson_3_1_checkpoint.keras'))
    generator = wgan_model.generator
    print(generator.summary())

    num_images = 15
    samples = np.random.normal(size=(num_images, Z_DIM))
    reconstruction = generator.predict(samples)

    for i in range(num_images):
        plt.subplot(5, 3, i + 1)
        plt.imshow(reconstruction[i])  # Show the reconstructed image
        plt.axis('off')

    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.suptitle('WGAN Generated New Images', fontsize=14, fontweight='bold')
    plt.show()

