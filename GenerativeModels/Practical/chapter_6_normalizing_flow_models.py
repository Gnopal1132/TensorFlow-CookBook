import tensorflow as tf
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import numpy as np

class RealNVP(tf.keras.models.Model):
    def __init__(self, input_dim, coupling_dim, regularization, coupling_layers):
        super(RealNVP, self).__init__()
        self.input_dim = input_dim
        self.coupling_dim = coupling_dim
        self.regularization = regularization
        self.coupling_layers_count = coupling_layers   # We will stack many coupling layers together, every coupling layer represent a invertible function
        self.coupling_layer_list = [self.coupling_layers() for _ in range(self.coupling_layers_count)]
        
        """
        Since the input vector dimension is 2, hence we use two dimensional vectors shown below.
        For general d-dimension,  
        
        """
        self.distribution = tfp.distributions.MultivariateNormalDiag(loc=[0., 0.], scale_diag=[1., 1.])
        self.masks = np.array([[0, 1], [1, 0]] * (coupling_layers // 2), dtype=np.float32)  # We just need half because, for reversal we can just do 1 - self.mask[i]
        """
        Why do we need masks? Since each coupling layer only transforms one part of the input, we need to alternate which part is transformed. 
        This ensures all dimensions get updated over multiple layers.
        
        Example:
            First layer: [0, 1] → First variable is fixed, second variable is transformed.
            Second layer: [1, 0] → First variable is transformed, second variable is fixed.
            This alternates across layers.

        """

        # Loss tracker
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
    
    @property
    def return_metric(self):
        return [self.loss_tracker]
    
    def coupling_layers(self):
        input_layer = tf.keras.layers.Input(shape=(self.input_dim, ))

        # Let's first create stacked layers for shifting vector.
        s_first = tf.keras.layers.Dense(self.coupling_dim, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(self.regularization))(input_layer)
        s_second = tf.keras.layers.Dense(self.coupling_dim, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(self.regularization))(s_first)
        s_third = tf.keras.layers.Dense(self.coupling_dim, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(self.regularization))(s_second)
        s_fourth = tf.keras.layers.Dense(self.coupling_dim, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(self.regularization))(s_third)
        s_final = tf.keras.layers.Dense(self.input_dim, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(self.regularization))(s_fourth)

        # Let's create for the translation vector.
        t_first = tf.keras.layers.Dense(self.coupling_dim, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(self.regularization))(input_layer)
        t_second = tf.keras.layers.Dense(self.coupling_dim, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(self.regularization))(t_first)
        t_third = tf.keras.layers.Dense(self.coupling_dim, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(self.regularization))(t_second)
        t_fourth = tf.keras.layers.Dense(self.coupling_dim, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(self.regularization))(t_third)
        t_final = tf.keras.layers.Dense(self.input_dim, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(self.regularization))(t_fourth)

        return tf.keras.Model(inputs=input_layer, outputs=[s_final, t_final])

    """
    Recap: What is Happening?
        1️⃣ Masking ensures only half of the input is transformed at a time.
        2️⃣ Affine transformation is applied selectively using s and t.
        3️⃣ Jacobian determinant correction ensures correct probability computation.
        4️⃣ Alternating masks ensure all dimensions are modified over multiple layers.
    """
    def call(self, x, training=True):
        log_det_inv = 0
        direction = 1

        if training:
            direction = -1  # direction = -1 for forward pass (training). +1 inverse pass (sampling)
        
        for i in range(self.coupling_layers_count)[::direction]:
            x_masked = x * self.masks[i]  # This hides one part of x by setting it to zero (using the mask). Suppose x = [3, 4] and mask = [1, 0], x_masked = [3, 0]
            reversed_mask = 1 - self.masks[i]  # This means the transformation is applied only to the part of x that was originally masked out.
            s, t = self.coupling_layer_list[i](x_masked)   # We get the corresponding s, and t vector
            
            # We zero out the part of s and t that corresponds to the unchanged part of x. This ensures that only the correct half of x is modified.
            # We first set the unmodified half of 𝑠, 𝑡 to zero. Then, when we apply the transformation 𝑥′=𝑥⋅𝑒**𝑠 + 𝑡, only the correct part of 𝑥 is modified.
           
            s *= reversed_mask  
            t *= reversed_mask
            """
            So:
                When forward, gate = (-1 - 1) / 2 = -1.
                When inverse, gate = (1 - 1) / 2 = 0.
                💡 Why do we need gate?

                It controls the exponent in t * exp(gate * s).
                In the forward pass, it helps compute the log-determinant.
            """
            gate = (direction - 1) / 2

            """
            This updates only the transformed half of x. The first half remains unchanged (x_masked). The second half is updated using 𝑠 and 𝑡:

                Forward (direction = -1): 𝑥′=𝑥⋅𝑒**(-𝑠) - 𝑡𝑒**(-𝑠)
                Inverse (direction = 1):  𝑥′=𝑥⋅𝑒**𝑠 + 𝑡
            Why use exponentials?
            Ensures invertibility: Multiplication by e**s prevents singularities.
                        
            """
            x = (reversed_mask * (x * tf.exp(direction * s) + direction * t * tf.exp(gate * s)) + x_masked)
            log_det_inv += gate * tf.reduce_sum(s, axis=1)    # This accumulates the log absolute determinant of the Jacobian.
        
        return x, log_det_inv
    
    def log_loss(self, x):
        y, logdet = self(x)
        log_likelihood = self.distribution.log_prob(y) + logdet
        return -tf.reduce_mean(log_likelihood)
    
    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self.log_loss(data)

        gradient = tape.gradient(loss, self.trainable_variables)    
        self.optimizer.apply_gradients(zip(gradient, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
    
    def test_step(self, data):
        loss = self.log_loss(data)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

class ImageGenerator(tf.keras.callbacks.Callback):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def generate(self):
        # From data to latent space.
        z, _ = model(normalized_data)

        # From latent space to data.
        samples = model.distribution.sample(self.num_samples)
        x, _ = model.predict(samples, verbose=0)

        return x, z, samples

    def display(self, x, z, samples, save_to=None):
        f, axes = plt.subplots(2, 2)
        f.set_size_inches(8, 5)

        axes[0, 0].scatter(
            normalized_data[:, 0], normalized_data[:, 1], color="r", s=1
        )
        axes[0, 0].set(title="Data space X", xlabel="x_1", ylabel="x_2")
        axes[0, 0].set_xlim([-2, 2])
        axes[0, 0].set_ylim([-2, 2])
        axes[0, 1].scatter(z[:, 0], z[:, 1], color="r", s=1)
        axes[0, 1].set(title="f(X)", xlabel="z_1", ylabel="z_2")
        axes[0, 1].set_xlim([-2, 2])
        axes[0, 1].set_ylim([-2, 2])
        axes[1, 0].scatter(samples[:, 0], samples[:, 1], color="g", s=1)
        axes[1, 0].set(title="Latent space Z", xlabel="z_1", ylabel="z_2")
        axes[1, 0].set_xlim([-2, 2])
        axes[1, 0].set_ylim([-2, 2])
        axes[1, 1].scatter(x[:, 0], x[:, 1], color="g", s=1)
        axes[1, 1].set(title="g(Z)", xlabel="x_1", ylabel="x_2")
        axes[1, 1].set_xlim([-2, 2])
        axes[1, 1].set_ylim([-2, 2])

        plt.subplots_adjust(wspace=0.3, hspace=0.6)
        if save_to:
            plt.savefig(save_to)
            print(f"\nSaved to {save_to}")
        plt.close()
        # plt.show()

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            x, z, samples = self.generate()
            self.display(
                x,
                z,
                samples,
                save_to="./callback_output/generated_img_%03d.png" % (epoch),
            )


if __name__ == '__main__':
    data = make_moons(3000, noise=0.05)[0].astype('float32')

    # Let's first normalize them to [0, 1]
    normalization = tf.keras.layers.Normalization()
    normalization.adapt(data)
    normalized_data = normalization(data)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(normalized_data.numpy()[:, 0], normalized_data.numpy()[:, 1])
    plt.show()


    # Hyperparameters
    COUPLING_DIM = 256
    COUPLING_LAYERS = 2
    INPUT_DIM = 2
    REGULARIZATION = 0.01
    BATCH_SIZE = 256
    EPOCHS = 1000

    model = RealNVP(
    input_dim=INPUT_DIM,
    coupling_layers=COUPLING_LAYERS,
    coupling_dim=COUPLING_DIM,
    regularization=REGULARIZATION,
    )

    # Compile and train the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./tensorboard/")
    img_generator_callback = ImageGenerator(num_samples=3000)

    model.fit(
    normalized_data,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[tensorboard_callback, img_generator_callback],
    )