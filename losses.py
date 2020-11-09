import tensorflow as tf


def sum_squared_error(y_true, y_pred):
    return tf.reduce_sum(tf.math.squared_difference(y_true, y_pred))
