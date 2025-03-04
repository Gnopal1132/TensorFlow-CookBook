import numpy as np
import tensorflow as tf
import os
import pandas as pd
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def preprocess_image(image):
    return tf.divide(tf.cast(image, tf.float32) - 127.5, 127.5)


class ConditionalGAN(tf.keras.Model):
    def __init__(self, image_size, channels, classes, latent_dim, gp_weight, critic_step, **kwargs):
        super(ConditionalGAN, self).__init__(**kwargs)
        self.image_size = image_size
        self.channels = channels
        self.classes = classes
        self.latent_dim = latent_dim
        self.gp_weight = gp_weight
        self.critic_step = critic_step

        # Metrics
        self.gp_loss = tf.keras.metrics.Mean(name='gradient_penalty_loss')
        self.c_loss = tf.keras.metrics.Mean(name='critic_loss')
        self.c_wass_loss = tf.keras.metrics.Mean(name='c_wasserstein_loss')
        self.g_loss = tf.keras.metrics.Mean(name='gradient_loss')

        # Critic and Discriminator
        self.critic = self.build_critic()
        self.generator = self.build_generator()

        # Optimizers
        self.c_optimizer = None
        self.g_optimizer = None

    def build_critic(self):
        critic_input = tf.keras.Input(shape=self.image_size + (self.channels,), name='critic_input')
        critic_label = tf.keras.Input(shape=self.image_size + (self.classes,), name='critic_label')
        input_critic = tf.keras.layers.concatenate([critic_input, critic_label], axis=-1)
        x = tf.keras.layers.Conv2D(32, 4, strides=2, padding='same')(input_critic)
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

        critic_model = tf.keras.Model(inputs=[critic_input, critic_label], outputs=critic_output)
        return critic_model

    def build_generator(self):
        generator_input = tf.keras.Input(shape=(self.latent_dim,), name='generator_input')
        generator_label = tf.keras.Input(shape=(self.classes,), name='generator_label')

        input_generator = tf.keras.layers.concatenate([generator_input, generator_label], axis=-1)
        x = tf.keras.layers.Reshape(target_shape=(1, 1, self.latent_dim + self.classes))(input_generator)
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
        generator_model = tf.keras.Model([generator_input, generator_label], generator_output)
        return generator_model

    def compile(self, c_optimizer, g_optimizer):
        super(ConditionalGAN, self).compile()
        self.c_optimizer = c_optimizer
        self.g_optimizer = g_optimizer

    @property
    def return_metrics(self):
        return [self.g_loss, self.c_loss, self.c_wass_loss, self.gp_loss]

    def gradient_penalty(self, batch_size, real_images, fake_images, one_hot_encoded_vectors):
        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
        difference = real_images - fake_images
        interpolation = real_images + alpha * difference

        with tf.GradientTape() as tape:
            tape.watch(interpolation)
            prediction = self.critic([interpolation, one_hot_encoded_vectors], training=True)

        gradients = tape.gradient(prediction, [interpolation])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
        return tf.reduce_mean((norm - 1) ** 2)

    def train_step(self, data_points):
        real_images, real_hot_labels = data_points
        images_one_hot = real_hot_labels[:, tf.newaxis, tf.newaxis, :]
        images_one_hot = tf.repeat(images_one_hot, self.image_size[0], axis=1)
        images_one_hot = tf.repeat(images_one_hot, self.image_size[0], axis=2)

        batch_size = tf.shape(real_images)[0]
        for _ in range(self.critic_step):
            latent_random_vector = tf.random.normal([batch_size, self.latent_dim])
            with tf.GradientTape() as c_tape:
                generator_prediction = self.generator([latent_random_vector, real_hot_labels], training=True)

                # Get the predictions
                fake_prediction = self.critic([generator_prediction, images_one_hot], training=True)
                real_prediction = self.critic([real_images, images_one_hot], training=True)

                # Let's find the loss
                c_wass_loss = tf.reduce_mean(fake_prediction) - tf.reduce_mean(real_prediction)
                gp_loss = self.gradient_penalty(batch_size, real_images, generator_prediction, images_one_hot)

                c_loss = c_wass_loss + self.gp_weight * gp_loss

            c_gradient = c_tape.gradient(c_loss, self.critic.trainable_variables)
            self.c_optimizer.apply_gradients(
                zip(c_gradient, self.critic.trainable_variables)
            )

        latent_random_vector = tf.random.normal([batch_size, self.latent_dim])
        with tf.GradientTape() as g_tape:
            generator_prediction = self.generator([latent_random_vector, real_hot_labels], training=True)
            fake_prediction = self.critic([generator_prediction, images_one_hot], training=True)

            generator_loss = -tf.reduce_mean(fake_prediction)

        gen_gradient = g_tape.gradient(generator_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))

        self.g_loss.update_state(generator_loss)
        self.c_loss.update_state(c_loss)
        self.c_wass_loss.update_state(c_wass_loss)
        self.gp_loss.update_state(gp_loss)

        return {m.name: m.result() for m in self.return_metrics}

    def get_config(self):
        base_config = super().get_config()

        c_optimizer_config = self.c_optimizer.get_config()
        g_optimizer_config = self.g_optimizer.get_config()

        base_config.update({
            'image_size': self.image_size,
            'channels': self.channels,
            'classes': self.classes,
            'latent_dim': self.latent_dim,
            'gp_weight': self.gp_weight,
            'critic_step': self.critic_step,
            'c_optimier_config': c_optimizer_config,
            'g_optimizer_config': g_optimizer_config,
        })
        return base_config

    @classmethod
    def from_config(cls, config):
        base_config = config.copy()
        c_optimizer_config = base_config.pop('c_optimizer_config', None)
        g_optimizer_config = base_config.pop('g_optimizer_config', None)

        model = super(ConditionalGAN, cls).from_config(base_config)
        c_optimizer = tf.keras.optimizers.Adam.from_config(c_optimizer_config)
        g_optimizer = tf.keras.optimizers.Adam.from_config(g_optimizer_config)

        model.compile(c_optimizer, g_optimizer)
        return model


class ImageGenerator(tf.keras.callbacks.Callback):
    def __init__(self, latent_dim, batch_size):
        self.latent_dim = latent_dim
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(
            shape=(self.batch_size, self.latent_dim)
        )
        # 0 label
        zero_label = tf.convert_to_tensor(np.repeat([[1, 0]], self.batch_size, axis=0), dtype=tf.float32)
        generated_images = self.model.generator(
            [random_latent_vectors, zero_label]
        )
        generated_images = generated_images * 127.5 + 127.5
        generated_images = generated_images.numpy()

        for i in range(self.batch_size):
            image = tf.keras.utils.array_to_img(generated_images[i])
            image.save(os.path.join(os.curdir, 'checkpoint_decoded_image', 'cgan_images', 'label_0',
                                    f'decoded_{epoch}_{i}.jpg'))

        # 1 label
        one_label = tf.convert_to_tensor(np.repeat([[0, 1]], self.batch_size, axis=0), dtype=tf.float32)
        generated_images = self.model.generator(
            [random_latent_vectors, one_label]
        )
        generated_images = generated_images * 127.5 + 127.5
        generated_images = generated_images.numpy()

        for i in range(self.batch_size):
            image = tf.keras.utils.array_to_img(generated_images[i])
            image.save(os.path.join(os.curdir, 'checkpoint_decoded_image', 'cgan_images', 'label_1',
                                    f'decoded_{epoch}_{i}.jpg'))


if __name__ == '__main__':
    dataset_path = os.path.join(os.curdir, 'dataset', 'img_align_celeba')
    label_path = os.path.join(os.curdir, 'dataset', 'list_attr_celeba.csv')
    data_info = pd.read_csv(label_path)

    # Hyperparameters
    GP_WEIGHT = 0.01
    BATCH_SIZE = 128
    IMG_SIZE = (64, 64)
    CHANNELS = 3
    ADAM_BETA_1 = 0.5
    LEARNING_RATE = 0.0002
    ADAM_BETA_2 = 0.99
    LATENT_DIM = 128
    CRITIC_STEP = 3
    CLASS_LABEL = 'Blond_Hair'  # We will make class label binary
    CLASSES = 2
    EPOCHS = 20

    # Changing the labels into binary labels.
    labels = data_info[CLASS_LABEL].tolist()
    labels = [1 if label_instance == 1 else 0 for label_instance in labels]

    dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        labels=labels,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        interpolation='bilinear',
        shuffle=True,
        seed=123,
    )
    dataset = dataset.map(lambda x, y: (preprocess_image(x), tf.one_hot(y, depth=CLASSES)))
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    for images, labels in dataset.take(1):
        print(images.shape)
        print(labels.shape)

    conditionalGAN = ConditionalGAN(image_size=IMG_SIZE, channels=CHANNELS,
                                    classes=CLASSES, latent_dim=LATENT_DIM,
                                    gp_weight=GP_WEIGHT, critic_step=CRITIC_STEP)

    c_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=ADAM_BETA_1, beta_2=ADAM_BETA_2)
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=ADAM_BETA_1, beta_2=ADAM_BETA_2)
    conditionalGAN.compile(c_optimizer=c_optimizer, g_optimizer=g_optimizer)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(os.getcwd(), 'checkpoints', 'cgan_celebA_checkpoint.keras'),
        save_weights_only=False,
        save_freq="epoch",
        save_best_only=True,
        verbose=0,
    )

    history = conditionalGAN.fit(
        dataset,
        epochs=EPOCHS * 100,
        steps_per_epoch=50,
        callbacks=[
            model_checkpoint_callback,
            ImageGenerator(batch_size=10, latent_dim=LATENT_DIM),
        ],
    )
    # Showing the trajectory
    pd.DataFrame(history.history).plot(figsize=(8, 8))
    plt.grid(True)
    plt.show()
