import numpy as np
import tensorflow as tf
import os
import pandas as pd
from matplotlib import pyplot as plt


# For GAN, the images must be in the range of [-1, 1], so that we can use tanh activation function on final layer of
# generator, which tends to provide stronger gradients than the sigmoid function.
def preprocess_image(image):
    image = (tf.cast(image, tf.float32) - 127.5) / 127.5
    return image


@tf.keras.utils.register_keras_serializable(package='GenerativeModels')
class DeepGenerativeAdversarialModel(tf.keras.models.Model):
    def __init__(self, image_size, latent_dim, **kwargs):
        super(DeepGenerativeAdversarialModel, self).__init__(**kwargs)
        self.g_optimizer = None
        self.d_optimizer = None
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.d_loss_metric = tf.keras.metrics.Mean(name='d_loss')
        self.g_loss_metric = tf.keras.metrics.Mean(name='g_loss')
        self.loss_function = tf.keras.losses.BinaryCrossentropy()

        # These both are decoder and generator model.
        self.discriminator_model = self.build_discriminator()
        self.generator_model = self.build_generator()

    @staticmethod
    def build_generator():
        generator_input = tf.keras.Input(shape=(100,))
        x = tf.keras.layers.Reshape((1, 1, 100))(generator_input)
        x = tf.keras.layers.Conv2DTranspose(512, kernel_size=4, strides=1, padding='valid', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)

        x = tf.keras.layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)

        x = tf.keras.layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)

        x = tf.keras.layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)

        generator_output = tf.keras.layers.Conv2DTranspose(1, kernel_size=4, strides=2, padding='same', use_bias=False,
                                                           activation='tanh')(x)

        generator = tf.keras.Model(inputs=generator_input, outputs=generator_output)
        return generator

    # A simple image classification model, to detect whether the image is real or fake. Sigmoid output 0/1.
    def build_discriminator(self):
        discriminator_input = tf.keras.Input(shape=self.image_size, name='discriminator_input')
        x = tf.keras.layers.Conv2D(32, kernel_size=4, strides=2, padding='same', use_bias=False)(discriminator_input)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Dropout(0.6)(x)

        x = tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Dropout(0.6)(x)

        x = tf.keras.layers.Conv2D(128, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Dropout(0.6)(x)

        x = tf.keras.layers.Conv2D(256, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Dropout(0.6)(x)

        x = tf.keras.layers.Conv2D(1, kernel_size=4, strides=1, padding='valid', use_bias=False, activation='sigmoid')(
            x)
        discriminator_output = tf.keras.layers.Flatten()(x)
        discriminator_model = tf.keras.Model(inputs=discriminator_input, outputs=discriminator_output)
        return discriminator_model

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def compile(self, d_optimizer, g_optimizer):
        super(DeepGenerativeAdversarialModel, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            generated_images = self.generator_model(random_latent_vectors, training=True)
            real_predictions = self.discriminator_model(real_images, training=True)
            fake_predictions = self.discriminator_model(generated_images, training=True)

            """
            Adding noise to the labels is a technique often used in Generative Adversarial Networks (GANs), 
            and it is known as label smoothing (for real labels) and label noise (for fake labels). 
            Here's why this is done:

            For real labels (label smoothing):
            
            Purpose: To avoid the discriminator becoming overconfident and trying to predict real labels as exactly 1 
            (which can lead to instability during training). By making the real labels slightly less than 1, the 
            discriminator is encouraged to be more flexible and not too confident, 
            making it harder for the model to get stuck.
            Implementation: 
            real_noisy_labels = real_labels + 0.1 * tf.random.uniform(tf.shape(real_predictions)) 
            adds random noise to the real labels, effectively smoothing the labels to a value in the range [0.9, 1.1]
            
            For fake labels (label noise):

            Purpose: To prevent the discriminator from being overly confident when classifying fake images as 0. 
            Adding noise here helps the discriminator to learn better and not be fooled easily by the generator’s output.
            Implementation: 
            fake_noisy_labels = fake_labels - 0.1 * tf.random.uniform(tf.shape(fake_predictions)) slightly decreases 
            the fake labels below 0 (in the range [−0.1,0.1]) to create noise in the fake labels.
            
            """
            # Creating the labels
            real_labels = tf.ones_like(real_predictions)
            real_noisy_labels = real_labels + 0.1 * tf.random.uniform(tf.shape(real_predictions))

            fake_labels = tf.zeros_like(fake_predictions)
            fake_noisy_labels = fake_labels - 0.1 * tf.random.uniform(tf.shape(fake_predictions))

            d_real_loss = self.loss_function(real_noisy_labels, real_predictions)
            d_fake_loss = self.loss_function(fake_noisy_labels, fake_predictions)

            d_loss = (d_real_loss + d_fake_loss) / 2.

            g_loss = self.loss_function(real_labels, fake_predictions)

        gradients_of_discriminator = d_tape.gradient(d_loss, self.discriminator_model.trainable_variables)
        gradients_of_generator = g_tape.gradient(g_loss, self.generator_model.trainable_variables)

        self.d_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator_model.trainable_variables))
        self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator_model.trainable_variables))

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)

        return {m.name: m.result() for m in self.metrics}

    def get_config(self):
        base_config = super().get_config()
        base_config.update({
            'latent_dim': self.latent_dim,
            'image_size': self.image_size,
        })
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class ImageGenerator(tf.keras.callbacks.Callback):
    def __init__(self, latent_dim, batch_size):
        self.latent_dim = latent_dim
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        noise_normal = tf.random.normal(shape=(self.batch_size, self.latent_dim))
        prediction = self.model.generator_model.predict(noise_normal)
        prediction = prediction * 127.5 + 127.5
        for i in range(self.batch_size):
            image = tf.keras.utils.array_to_img(prediction[i])
            image.save(os.path.join(os.curdir, 'checkpoint_decoded_image', 'gan_images', f'decoded_{epoch}_{i}.jpg'))


if __name__ == '__main__':
    # Parameters
    IMAGE_SIZE = (64, 64)
    BATCH_SIZE = 128

    data_path = os.path.join(os.curdir, 'dataset', 'lego_dataset')
    # dataset = tf.keras.utils.image_dataset_from_directory(
    #     data_path,
    #     image_size=IMAGE_SIZE,
    #     labels=None,
    #     color_mode='grayscale',
    #     batch_size=BATCH_SIZE,
    #     seed=42,
    #     interpolation='bilinear',
    #     shuffle=True,
    # )
    # dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    # for images in dataset.take(1):
    #     print(images.shape)

    d_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5, beta_2=0.999)
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5, beta_2=0.999)
    beta_1 = 0.5
    beta_2 = 0.999

    dcgan = DeepGenerativeAdversarialModel(image_size=IMAGE_SIZE + (1,), latent_dim=100)
    dcgan.compile(d_optimizer=d_optimizer, g_optimizer=g_optimizer)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(os.getcwd(), 'checkpoints', 'gan_lesson_3_checkpoint.keras'),
        save_weights_only=False,
        save_freq="epoch",
        monitor="g_loss",
        mode="min",
        save_best_only=True,
        verbose=0,
    )
    decoded_image_checkpoint = ImageGenerator(latent_dim=100, batch_size=10)
    # history = dcgan.fit(
    #     dataset,
    #     epochs=300,
    #     callbacks=[model_checkpoint_callback,
    #                decoded_image_checkpoint], )
#
    # # Showing the trajectory
    # pd.DataFrame(history.history).plot(figsize=(8, 8))
    # plt.grid(True)
    # plt.show()

    # load the model
    dcgan = tf.keras.models.load_model(os.path.join(os.curdir, 'checkpoints', 'gan_lesson_3_checkpoint.keras'))
    generator = dcgan.generator_model
    r, c = 3, 5
    fig, axs = plt.subplots(r, c, figsize=(10, 6))
    fig.suptitle("Generated images", fontsize=20)

    noise = np.random.normal(size=(r * c, 100))
    gen_imgs = generator.predict(noise)

    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt], cmap="gray_r")
            axs[i, j].axis("off")
            cnt += 1
    plt.show()