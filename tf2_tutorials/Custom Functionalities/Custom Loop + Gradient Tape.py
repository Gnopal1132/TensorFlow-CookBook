import tensorflow as tf
import numpy as np


def f(w1, w2):
    return 3 * w1 ** 2 + 2 * w1 * w2


def f_2(w1, w2, w3):
    return 2 * w1 ** 2 + 3 * w1 * w2 * w3 + 2 * w2 + 3 * w3 ** 0.5


class CustomTraining:
    def __init__(self):
        pass

    def training(self):
        dataset = tf.keras.datasets.fashion_mnist
        (x_train, y_train), (x_valid, y_valid) = dataset.load_data()

        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=x_train.shape[1:]),
            tf.keras.layers.Dense(100, activation='elu', kernel_initializer='he_normal'),
            tf.keras.layers.Dense(100, activation='elu', kernel_initializer='he_normal'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        epochs = 10
        batch_size = 32
        steps = x_train.shape[0] // batch_size
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

        # Training Loop
        for epoch in range(epochs):
            for step in range(steps):
                x, y = self.random_batch(X=x_train, Y=y_train, batch_size=batch_size)
                x = tf.divide(tf.cast(x, tf.float32), 255.)
                with tf.GradientTape() as tape:
                    output = model(x, training=True)
                    out_loss = loss(y, output)

                gradients = tape.gradient(out_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                if step % 10 == 0:
                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, float(out_loss))
                    )
                    print("Seen so far: %s samples" % ((step + 1) * batch_size))

    @staticmethod
    def random_batch(X, Y, batch_size=32):
        idx = np.random.randint(len(X), size=batch_size)
        return X[idx], Y[idx]


if __name__ == '__main__':
    # Chapter 1: Gradient Calculation

    # To calculate gradient you can either manually differentiate and then do it.
    # Or you could also do
    eps = 1e-6
    w1, w2 = 5, 3

    # With respect to w1
    print('w.r.t w1: ', round((f(w1 + eps, w2) - f(w1, w2)) / eps, 2))

    # With respect to w2
    print('w.r.t w2: ', round((f(w1, w2 + eps) - f(w1, w2)) / eps, 2))

    # Enter the Gradient Tape:
    # Lets use gradient tape which will automatically record every operation that involves a variable
    # Must the variable in this case w1 and w2 MUST be a variable.
    w1, w2 = tf.Variable(5.), tf.Variable(3.)

    with tf.GradientTape() as tape:
        output = f(w1=w1, w2=w2)

    # And finally we ask this tape to calculate the gradient
    # of result output w.r.t the variables [w1, w2]
    gradients = tape.gradient(output, [w1, w2])
    print(gradients)

    # If you again write line 81, you will get error. After calculating
    # the gradient it deletes the tape. In order to use it multiple times
    # use persistent tape.
    # To make it persistent make

    with tf.GradientTape(persistent=True) as tape:
        z = f(w1, w2)

    print(tape.gradient(z, w1))
    print(tape.gradient(z, w2))

    # # Note the w1, and w2 must be Variables not Constant
    # if they are, and you try to calculate the gradient on them.
    # You will get [None, None].
    # To calculate gradient on constants c1, and c2
    # you need to watch them within Gradient tape.
    c1, c2 = tf.constant(5.), tf.constant(3.)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(c1)
        tape.watch(c2)
        z = f(c1, c2)

    print('w.r.t c1', tape.gradient(z, c1))
    print('w.r.t c2', tape.gradient(z, c2))

    # Gradient tape just do one forward pass and one backward pass to compute all the gradients. If you want to stop
    # some part so that it doesn't flow the gradient, use tf.stop_gradient() on it. It will return Identity during
    # forward pass but during backward pass it wont let the gradient flow

    def f(w1, w2):
        return 3 * w1 ** 2 + tf.stop_gradient(2 * w1 * w2)


    with tf.GradientTape() as tape:
        z = f(w1, w2)

    print(tape.gradient(z, [w1, w2]), end='\n\n')

    # Exercise: Calculate the Jacobian and Hessian of f_2.
    w1, w2, w3 = tf.Variable(1.), tf.Variable(2.), tf.Variable(3.)

    with tf.GradientTape(persistent=True) as Hessian:
        with tf.GradientTape() as Jacobian:
            out = f_2(w1, w2, w3)
        jacobian = Jacobian.gradient(out, [w1, w2, w3])

    # Remember for hessian you need to differentiate with respect to every variable.
    hessians = [Hessian.gradient(jacob, [w1, w2, w3]) for jacob in jacobian]

    print(jacobian)
    print(hessians)

    custom_train = CustomTraining()
    custom_train.training()
