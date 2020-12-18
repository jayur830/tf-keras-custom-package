import numpy as np
import tensorflow as tf


class Local(tf.keras.layers.Layer):
    def __init__(self,
                 units: int,
                 activation: str = None,
                 kernel_initializer=None,
                 **kwargs):
        self.__units = units
        self.__activation = Local.__get_activation(activation)
        self.__kernel_initializer = kernel_initializer
        self.__w = self.__b = self.__cond = self.__where_y = None
        super(Local, self).__init__(**kwargs)

    def build(self, input_shape):
        self.__w = self.add_weight(
            name="w",
            shape=(input_shape[1], self.__units),
            initializer=Local.__locally_dense_initializer,
            trainable=True)

        self.__cond = tf.convert_to_tensor(Local.__assign_nan(np.ones(shape=self.__w.shape, dtype=np.bool), self.__w.shape, False))
        self.__where_y = tf.zeros(shape=self.__w.shape)

        self.__b = self.add_weight(
            name="b",
            shape=(self.__units,),
            initializer=tf.keras.initializers.get("zeros"),
            trainable=True)

        super(Local, self).build(input_shape)
        self.built = True

    @tf.autograph.experimental.do_not_convert
    def call(self, x, **kwargs):
        cal_kernel = tf.where(self.__cond, self.__w, self.__where_y)
        output = self.__activation(tf.matmul(x, cal_kernel) + self.__b)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.__units

    @staticmethod
    def __get_activation(activation):
        if activation is None:
            return tf.keras.activations.linear
        elif type(activation) is str:
            if activation == "":
                return tf.keras.activations.linear
            return tf.keras.activations.get(activation)
        else:
            return activation

    @staticmethod
    def __locally_dense_initializer(shape, dtype=None):
        kernel = Local.__assign_nan(np.random.random(shape), shape, np.nan)
        return tf.convert_to_tensor(kernel, dtype=dtype)

    @staticmethod
    def __assign_nan(matrix, shape, value):
        rows, cols = shape[0], shape[1]
        max_length = min(rows, cols) - 1

        for i in range(max_length):
            matrix[i, :max_length - i] = matrix[rows - 1 - i, cols - max_length + i:cols] = value
        return matrix


class ThresholdedLeakyReLU(tf.keras.layers.Layer):
    def __init__(self, alpha=.01, threshold=1., **kwargs):
        super(ThresholdedLeakyReLU, self).__init__(**kwargs)
        self.supports_masking = True
        self.__alpha = K.cast_to_floatx(alpha)
        self.__threshold = K.cast_to_floatx(threshold)

    def build(self, input_shape):
        super(ThresholdedLeakyReLU, self).build(input_shape)
        self.built = True

    @tf.autograph.experimental.do_not_convert
    def call(self, x, **kwargs):
        return tf.where(x < self.__threshold, K.relu(x, alpha=self.__alpha), self.__alpha * (x - self.__threshold) + self.__threshold)

    def compute_output_shape(self, input_shape):
        return input_shape
