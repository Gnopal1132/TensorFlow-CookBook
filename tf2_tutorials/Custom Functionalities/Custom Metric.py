import tensorflow as tf

"""
When you define a metric using a simple function, Keras automatically calls it for each batch,and it keeps track 
of the mean during each epoch, just like we did manually. So if you define the class for it the only advantage u get 
is get_config().
"""


def create_huber(threshold):
    def huber_metric(y_true, y_pred):
        error = y_true - y_pred
        condition = tf.abs(error) < threshold
        square = 0.5 * tf.square(error)
        absolute = threshold * (tf.abs(error) - threshold * 0.5)
        return tf.where(condition, square, absolute)
    return huber_metric


"""
While implementing it through class its bit different. You need to implement two methods:
    1. def update_state(self, y_true, y_pred, sample_weight=None)   ----> Used to update weights and variables
    2. def result(self)  ----> returns the metric score
"""


class HuberMetric(tf.keras.metrics.Metric):
    def __init__(self, threshold, **kwargs):
        super(HuberMetric, self).__init__(**kwargs)
        self.threshold = threshold
        self.total = self.add_weight('total', initializer='zeros', dtype=tf.float32)
        self.count = self.add_weight('count', initializer='zeros', dtype=tf.float32)

    def huber_metric(self, y_true, y_pred):
        error = y_true - y_pred
        condition = tf.abs(error) < self.threshold
        square = 0.5 * tf.square(error)
        absolute = self.threshold * (tf.abs(error) - self.threshold*0.5)
        return tf.where(condition, square, absolute)

    def update_state(self, y_true, y_pred):
        metric = self.huber_metric(y_true, y_pred)
        self.total.assign_add(tf.reduce_sum(metric))
        self.count.assign_add(tf.size(y_true, out_type=tf.float32))

    def result(self):
        return self.total / self.count

    def get_config(self):
        base_config = super(HuberMetric, self).get_config()
        return {**base_config, 'threshold':self.threshold}


if __name__ == '__main__':
    huber = create_huber(threshold=0.5)
    huber_metric = HuberMetric(threshold=0.5)
    y_true = tf.constant([[0, 1], [0, 0]], dtype=tf.float32)
    y_pred = tf.constant([[0.6, 0.4], [0.4, 0.6]], dtype=tf.float32)
    print(huber(y_true=y_true, y_pred=y_pred))
    print(huber_metric(y_true=y_true, y_pred=y_pred))
