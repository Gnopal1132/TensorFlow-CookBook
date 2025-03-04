import tensorflow as tf
import numpy as np
import os


class MaskedConv2DLayer(tf.keras.layers.Layer):
    def __init__(self, mask_type, **kwargs):
        super(MaskedConv2DLayer, self).__init__()
        self.mask_type = mask_type
        self.conv = tf.keras.layers.Conv2D(**kwargs)

    def build(self, input_shape):
        # Build the conv2d layer to initialize kernel variables
        self.conv.build(input_shape)
        # Use the initialized kernel to create the mask
        kernel_shape = self.conv.kernel._shape
        self.mask = np.zeros(shape=kernel_shape)
        self.mask[: kernel_shape[0] // 2, ...] = 1.0
        self.mask[kernel_shape[0] // 2, : kernel_shape[1] // 2, ...] = 1.0
        if self.mask_type == "B":
            self.mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...] = 1.0

    def call(self, inputs):
        self.conv.kernel.assign(self.conv.kernel * self.mask)
        return self.conv(inputs)

    def get_config(self):
        cfg = super().get_config()
        return cfg


class ResidualClass(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(ResidualClass, self).__init__(**kwargs)
        self.filters = filters
        self.conv1 = tf.keras.layers.Conv2D(filters=filters // 2, kernel_size=1, activation='relu')
        self.pixel_conv = MaskedConv2DLayer(mask_type='B', filters=filters // 2, kernel_size=3, activation='relu',
                                            padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, activation='relu')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pixel_conv(x)
        x = self.conv2(x)
        return x

    def get_config(self):
        old_config = super().get_config()
        old_config.update({'filters': self.filters})
        return old_config


if __name__ == '__main__':
    (x_train, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
    IMAGE_SIZE = 16
    PIXEL_LEVELS = 4
    N_FILTERS = 128
    RESIDUAL_BLOCKS = 5
    BATCH_SIZE = 128
    EPOCHS = 150

    # Preprocess the data
    def preprocess(imgs_int):
        imgs_int = np.expand_dims(imgs_int, -1)
        imgs_int = tf.image.resize(imgs_int, (IMAGE_SIZE, IMAGE_SIZE)).numpy()
        imgs_int = (imgs_int / (256 / PIXEL_LEVELS)).astype(int)
        imgs = imgs_int.astype("float32")
        imgs = imgs / PIXEL_LEVELS
        return imgs, imgs_int


    input_data, output_data = preprocess(x_train)
    inputs = tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
    x = MaskedConv2DLayer(
        mask_type="A",
        filters=N_FILTERS,
        kernel_size=7,
        activation="relu",
        padding="same",
    )(inputs)

    for _ in range(RESIDUAL_BLOCKS):
        x = ResidualClass(filters=N_FILTERS)(x)

    for _ in range(2):
        x = MaskedConv2DLayer(
            mask_type="B",
            filters=N_FILTERS,
            kernel_size=1,
            strides=1,
            activation="relu",
            padding="valid",
        )(x)

    out = tf.keras.layers.Conv2D(
        filters=PIXEL_LEVELS,
        kernel_size=1,
        strides=1,
        activation="softmax",
        padding="valid",
    )(x)

    pixel_cnn = tf.keras.Model(inputs, out)
    print(pixel_cnn.summary())

    adam = tf.keras.optimizers.Adam(learning_rate=0.0005)
    pixel_cnn.compile(optimizer=adam, loss="sparse_categorical_crossentropy")


    class ImageGenerator(tf.keras.callbacks.Callback):
        def __init__(self, num_img):
            self.num_img = num_img

        def sample_from(self, probs, temperature):  # <2>
            probs = probs ** (1 / temperature)
            probs = probs / np.sum(probs)
            return np.random.choice(len(probs), p=probs)

        def generate(self, temperature):
            generated_images = np.zeros(
                shape=(self.num_img,) + (pixel_cnn.input_shape)[1:]
            )
            batch, rows, cols, channels = generated_images.shape

            for row in range(rows):
                for col in range(cols):
                    for channel in range(channels):
                        probs = self.model.predict(generated_images, verbose=0)[
                                :, row, col, :
                                ]
                        generated_images[:, row, col, channel] = [
                            self.sample_from(x, temperature) for x in probs
                        ]
                        generated_images[:, row, col, channel] /= PIXEL_LEVELS

            return generated_images

        def on_epoch_end(self, epoch, logs=None):
            generated_images = self.generate(temperature=1.0)
            for i in range(self.num_img):
                image = tf.keras.utils.array_to_img(generated_images[i])
                image.save(os.path.join(os.curdir, 'checkpoint_decoded_image', 'pixelcnn', f'decoded_{epoch}_{i}.jpg'))


    img_generator_callback = ImageGenerator(num_img=10)
    pixel_cnn.fit(
        input_data,
        output_data,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[img_generator_callback],
    )
