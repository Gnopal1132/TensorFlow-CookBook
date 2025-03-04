import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import datetime
import numpy as np
import pandas as pd
import glob

import matplotlib.pyplot as plt

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

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable(package='VAE')
class VAE(tf.keras.models.Model):
    def __init__(self, input_shape, CHANNELS, NUM_FEATURES, BETA=2000, Z_DIM=200, **kwargs):
        super().__init__(**kwargs)

        self.shape_before_flattening = None
        self.kl_metric = tf.keras.metrics.Mean(name='kl_loss')
        self.bce_metric = tf.keras.metrics.Mean(name='bce_loss')
        self.total_metric = tf.keras.metrics.Mean(name='total_loss')

        # Initializing Variables
        self.input_shape = input_shape

        # Some constants
        self.BETA = BETA
        self.num_features = NUM_FEATURES
        self.z_dim = Z_DIM
        self.channels = CHANNELS

        # Initializing the Encoder and Decoder model.
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def build_encoder(self):
        encoder_input = tf.keras.Input(shape=self.input_shape, name='encoder_input')
        x = tf.keras.layers.Conv2D(self.num_features, kernel_size=(3, 3), strides=2, padding='same')(encoder_input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv2D(self.num_features, kernel_size=(3, 3), strides=2, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv2D(self.num_features, kernel_size=(3, 3), strides=2, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        self.shape_before_flattening = K.int_shape(x)[1:]
        x = tf.keras.layers.Flatten()(x)
        z_mean = tf.keras.layers.Dense(self.z_dim, name='z_mean')(x)
        z_log_var = tf.keras.layers.Dense(self.z_dim, name='z_log_var')(x)
        z = SamplingLayer()([z_mean, z_log_var])
        encoder = tf.keras.Model(inputs=encoder_input, outputs=[z_mean, z_log_var, z])
        return encoder

    def build_decoder(self):
        decoder_input = tf.keras.Input(shape=(self.z_dim,), name='decoder_input')
        x = tf.keras.layers.Dense(np.prod(self.shape_before_flattening))(decoder_input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Reshape(self.shape_before_flattening)(x)
        x = tf.keras.layers.Conv2DTranspose(self.num_features, kernel_size=(3, 3), strides=2, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv2DTranspose(self.num_features, kernel_size=(3, 3), strides=2, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv2DTranspose(self.num_features, kernel_size=(3, 3), strides=2, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        decoder_out = tf.keras.layers.Conv2D(self.channels, kernel_size=(3, 3), strides=1, padding='same',
                                             activation='sigmoid',
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

    """
    What Does get_config() Do?
    get_config() is like a blueprint of your model/layer. It tells TensorFlow:

    What parameters were used to create the model/layer.
    How to save these parameters so that the model can be recreated later.
    When you save a model using model.save(), TensorFlow calls get_config() on all the custom components (like your 
    VAE class or SamplingLayer) to store their configuration in the saved file. This allows the model to be fully reconstructed later.
    
    In simple words:
    
    It saves the “recipe” for creating your model or layer.
    What Does from_config() Do?
    from_config() is the chef that uses the recipe. When you load a saved model using tf.keras.models.load_model(), TensorFlow:
    
    Reads the configuration stored in the saved file (from get_config()).
    Calls from_config() to rebuild the model or layer using that configuration.
    In simple words:
    
    It uses the saved parameters to recreate the model or layer.
    Why Do You Need Both?
    get_config() is for saving the parameters of your model or layer.
    from_config() is for using those saved parameters to recreate it.
    Together, they allow your custom objects to be saved and loaded seamlessly.
    
     
    Parameters to Include in get_config
    
    You should include parameters that:

    1. Define the model's architecture: These are things like layer configurations, dimensions,
       activation functions, number of units, etc.
       
    2. Impact the model's behavior: Hyperparameters such as learning_rate, beta, or dropout_rate
       that influence the training or inference process.
       
    3. Custom properties: Any custom attribute you’ve defined (like CHANNELS, 
       NUM_FEATURES, or Z_DIM in your case).
       
    Parameters You Can Omit
    You don’t need to include attributes like self.encoder or self.decoder explicitly, because:

    These are objects built dynamically during model initialization (__init__ method).
    Instead, you include the parameters used to create these objects (e.g., input_shape, NUM_FEATURES, etc.).
    TensorFlow will use these parameters to reconstruct the encoder and decoder when the model is loaded.
    
    """

    def get_config(self):
        base_config = super().get_config()
        base_config.update({
            'input_shape': self.input_shape,
            'CHANNELS': self.channels,
            'NUM_FEATURES': self.num_features,
            'BETA': self.BETA,
            'Z_DIM': self.z_dim,
        })
        return base_config

    """
    While TensorFlow handles most of this for you when using @register_keras_serializable, 
    it’s good practice to provide an explicit from_config method to ensure that all arguments are handled correctly.
    """

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Creating a custom callback to visually see how the decoder is creating new images
class DecoderImageTransformation(tf.keras.callbacks.Callback):
    def __init__(self, num_images, latent_dimension):
        self.num_images = num_images
        self.latent_dimension = latent_dimension

    # On the end of every epoch
    def on_epoch_end(self, epoch, logs=None):
        random_vector = tf.random.normal(shape=(self.num_images, self.latent_dimension))
        generated_images = self.model.decoder(random_vector)
        generated_images *= 255.
        for i in range(self.num_images):
            img = tf.keras.utils.array_to_img(generated_images[i])
            img.save("/content/decoder_output/generated_img_%03d_%d.png" % (epoch, i))


def process_image(image):
    image = tf.cast(image, tf.float32) / 255.
    return image


def tensorboard_id(log_path):
    current_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return log_path + '/' + current_id


def get_vector_from_label(data, vae, embedding_dim, label):
    # Firstly we define the variables to keep track of sum and mean of the positive and negative vectors
    current_sum_POS = np.zeros(shape=embedding_dim, dtype="float32")
    current_n_POS = 0
    current_mean_POS = np.zeros(shape=embedding_dim, dtype="float32")

    current_sum_NEG = np.zeros(shape=embedding_dim, dtype="float32")
    current_n_NEG = 0
    current_mean_NEG = np.zeros(shape=embedding_dim, dtype="float32")

    current_vector = np.zeros(shape=embedding_dim, dtype="float32")
    current_dist = 0

    print("label: " + label)
    print("images : POS move : NEG move :distance : 𝛥 distance")

    # We dont have to go through entire lego_dataset to find a stabilizing vector for the label.
    while current_n_POS < 10000:
        batch = list(data.take(1).get_single_element())

        # im = (128, 64, 64, 3)   and   attribute = (128, )
        im = batch[0]
        attribute = batch[1]

        # We get the encoding of the array.
        _, _, z = vae.encoder.predict(np.array(im), verbose=0)

        # We separate out the encodings corresponding to +1 and -1.
        z_POS = z[attribute == 1]
        z_NEG = z[attribute == -1]

        # Then we calculate the sum and the mean for it
        if len(z_POS) > 0:
            current_sum_POS = current_sum_POS + np.sum(z_POS, axis=0)
            current_n_POS += len(z_POS)
            new_mean_POS = current_sum_POS / current_n_POS

            # np.linalg.norm() returns the magnitude of the vector = sqrt(a1**2 + a2**2 + ... + an**2)
            # We are subtracting because, as we get close to the stabilizing vector this movement_POS will become small.
            movement_POS = np.linalg.norm(new_mean_POS - current_mean_POS)

        if len(z_NEG) > 0:
            current_sum_NEG = current_sum_NEG + np.sum(z_NEG, axis=0)
            current_n_NEG += len(z_NEG)
            new_mean_NEG = current_sum_NEG / current_n_NEG
            movement_NEG = np.linalg.norm(new_mean_NEG - current_mean_NEG)

        # Then we get the new values for it.
        current_vector = new_mean_POS - new_mean_NEG
        new_dist = np.linalg.norm(current_vector)
        dist_change = new_dist - current_dist

        print(
            str(current_n_POS)
            + "    : "
            + str(np.round(movement_POS, 3))
            + "    : "
            + str(np.round(movement_NEG, 3))
            + "    : "
            + str(np.round(new_dist, 3))
            + "    : "
            + str(np.round(dist_change, 3))
        )

        current_mean_POS = np.copy(new_mean_POS)
        current_mean_NEG = np.copy(new_mean_NEG)
        current_dist = np.copy(new_dist)

        # Adding them together gives the total "movement" or instability in the latent space
        # representation for this attribute. If this is smaller than 0.08 we say that they are stabilized enough.
        # So we break the loop.
        if np.sum([movement_POS, movement_NEG]) < 0.08:
            current_vector = current_vector / current_dist
            print("Found the " + label + " vector")
            break

    return current_vector


def add_vector_to_images(data, vae, feature_vec):
    n_to_show = 5
    # Different factors for the feature_vector
    factors = [-4, -3, -2, -1, 0, 1, 2, 3, 4]

    example_batch = list(data.take(1).get_single_element())
    example_images = example_batch[0]

    _, _, z_points = vae.encoder.predict(example_images, verbose=0)

    fig = plt.figure(figsize=(18, 10))

    counter = 1

    for i in range(n_to_show):
        img = example_images[i]
        sub = fig.add_subplot(n_to_show, len(factors) + 1, counter)
        sub.axis("off")
        sub.imshow(img)

        counter += 1

        for factor in factors:
            changed_z_point = z_points[i] + feature_vec * factor
            changed_image = vae.decoder.predict(
                np.array([changed_z_point]), verbose=0
            )[0]

            sub = fig.add_subplot(n_to_show, len(factors) + 1, counter)
            sub.axis("off")
            sub.imshow(changed_image)

            counter += 1

    plt.show()


def morph_faces(data, vae):
    factors = np.arange(0, 1, 0.1)

    # The morphing process transitions from the first image to the second image in the batch
    example_batch = list(data.take(1).get_single_element())[:2]
    example_images = example_batch[0]
    _, _, z_points = vae.encoder.predict(example_images, verbose=0)

    fig = plt.figure(figsize=(18, 8))

    counter = 1

    img = example_images[0]
    sub = fig.add_subplot(1, len(factors) + 2, counter)
    sub.axis("off")
    sub.imshow(img)

    counter += 1

    for factor in factors:
        # z_point[0] is for the first image and z_points[1] is for the second image of the first batch. So we are slowly
        # transitioning from image A to image B.
        changed_z_point = z_points[0] * (1 - factor) + z_points[1] * factor
        changed_image = vae.decoder.predict(
            np.array([changed_z_point]), verbose=0
        )[0]
        sub = fig.add_subplot(1, len(factors) + 2, counter)
        sub.axis("off")
        sub.imshow(changed_image)

        counter += 1

    img = example_images[1]
    sub = fig.add_subplot(1, len(factors) + 2, counter)
    sub.axis("off")
    sub.imshow(img)

    plt.show()


if __name__ == '__main__':
    """
    # Trained on colab to extract the data
    
    import zipfile
    
    file_ref = zipfile.ZipFile("/content/drive/MyDrive/img_align_celeba.zip", 'r')
    file_ref.extractall("/content")
    file_ref.close()
    """
    image_path = "/content/img_align_celeba"  # Replace with your path
    # train_data = tf.keras.utils.image_dataset_from_directory(
    #     image_path,
    #     labels=None,
    #     color_mode='rgb',
    #     image_size=(64, 64),
    #     batch_size=128,
    #     shuffle=True,
    #     seed=42,
    #     interpolation='bilinear',
    # )

    # train_data = train_data.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    # for image in train_data.take(1):
    #     print(image.shape)

    input_dimension = (64, 64, 3)
    EPOCHS = 10
    # Let's first define the optimizer
    adam = tf.keras.optimizers.Adam(learning_rate=5e-4)
    vae = VAE(input_shape=input_dimension, CHANNELS=3, NUM_FEATURES=128)

    print(vae.summary())
    print(vae.encoder.summary())
    print(vae.decoder.summary())

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

    # vae.compile(optimizer=adam)
    # history = vae.fit(
    #     train_data,
    #     epochs=EPOCHS,
    #     callbacks=[model_checkpoint_callback, tensorboard_callback,
    #                DecoderImageTransformation(num_images=10, latent_dimension=200)],
    # )

    # Showing the trajectory
    # pd.DataFrame(history.history).plot(figsize=(8, 8))
    # plt.grid(True)
    # plt.show()
    #
    # Let's load the model
    vae_model = tf.keras.models.load_model(checkpoint_path, custom_objects={"VAE": VAE, "SamplingLayer": SamplingLayer})

    encoder_model = vae_model.encoder
    decoder_model = vae_model.decoder

    # Generate new samples
    # plt.figure(figsize=(10, 8))
    # num_images = 25
    # z_dim = 200
    # samples = np.random.normal(size=(num_images, z_dim))
    # reconstruction = decoder_model.predict(samples)
    # reconstruction = np.clip(reconstruction * 255, 0, 255).astype(np.uint8)  # Scale and cast
    #
    # for i in range(num_images):
    #     plt.subplot(5, 5, i + 1)
    #     plt.imshow(reconstruction[i])  # Show the reconstructed image
    #     plt.axis('off')
    #
    # plt.subplots_adjust(hspace=0.4, wspace=0.4)
    # plt.suptitle('VAE Generated New Images', fontsize=14, fontweight='bold')
    # plt.savefig('./plots/vae_generated_new_images_celeba.png')
    # plt.show()

    # Let's manipulate the vectors.
    attributes = pd.read_csv(os.path.join(os.curdir, 'lego_dataset', 'list_attr_celeba.csv'))
    print(attributes.columns)
    attributes.head()

    LABEL = "Wearing_Hat"  # <- Set this label
    labelled_test = tf.keras.utils.image_dataset_from_directory(
        "/content/img_align_celeba",
        labels=attributes[LABEL].tolist(),  # Label for every image. +1 represent the presence and -1 represents
        # the absence.
        color_mode="rgb",
        image_size=(64, 64),
        batch_size=128,
        shuffle=True,
        seed=42,
        validation_split=0.2,
        subset="validation",
        interpolation="bilinear",
    )

    labelled = labelled_test.map(lambda x, y: (process_image(x), y))

    # Find the attribute vector
    attribute_vec = get_vector_from_label(labelled, vae, 200, LABEL)

    # Add vector to images
    add_vector_to_images(labelled, vae, attribute_vec)
