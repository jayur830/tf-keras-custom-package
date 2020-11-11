import tensorflow as tf


def sum_squared_error(y_true, y_pred):
    return tf.reduce_sum(tf.math.squared_difference(y_true, y_pred))


def focal_loss(y_true, y_pred):
    alpha, gamma = .25, 2.
    p_t = y_pred[y_true.argmax()]
    return -alpha * (1 - p_t) ** gamma * tf.math.log(p_t)
