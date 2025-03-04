import tensorflow as tf

# Here we will define a Custom Huber Loss Function.
"""
To implement a custom loss function you have to simply implement two different methods:
        1. def call(self, y_true, y_pred):  --> Returns the loss
        2. get_config(self):  --> saves the threshold
"""


class HuberLoss(tf.keras.losses.Loss):
    def __init__(self, threshold, **kwargs):
        super(HuberLoss, self).__init__(**kwargs)
        self.threshold = threshold

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        condition = tf.abs(error) < self.threshold
        return tf.where(condition, tf.square(error), tf.abs(error))

    def get_config(self):
        base_config = super(HuberLoss, self).get_config()
        return {**base_config, 'threshold': self.threshold}


class ArcLoss(tf.keras.losses.Loss):
    def __init__(self, weights, margin, hypersphere_radius):
        super(ArcLoss, self).__init__()
        self.margin = margin
        self.weights = weights
        self.hyp_radius = hypersphere_radius

    def call(self, input, label):
        normalized_weights = tf.math.l2_normalize(self.weights, axis=0)
        normalized_embeddings = tf.math.l2_normalize(input)

        cosine_similarity = tf.matmul(normalized_embeddings, normalized_weights)

        angle_theta = tf.math.acos(cosine_similarity)

        margined_theta = tf.math.add(angle_theta, self.margin)
        shifted_theta = tf.math.multiply(self.hyp_radius, tf.math.cos(margined_theta))

        return tf.nn.softmax_cross_entropy_with_logits(label, shifted_theta)

    def get_config(self):
        base_config = super(ArcLoss, self).get_config()
        return {**base_config, 'margin': self.margin, 'radius': self.hyp_radius}


if __name__ == '__main__':
    huber_loss = HuberLoss(threshold=0.01)
    # Now simply you can pass it in model.compile(loss=huber_loss, ...)
    # Note, Now when you save the model the threshold will be saved along
    # with it. So while loading the model, you need to map the class name
    # to class itself.
    # model = tf.keras.models.load_model('model.h5',custom_objects={'HuberLoss':HuberLoss})
