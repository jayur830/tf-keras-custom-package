import numpy as np
import tensorflow as tf


class LocallyConnectedDense(tf.keras.layers.Layer):
    def __init__(self,
                 units: int,
                 activation=None,
                 **kwargs):
        super(LocallyConnectedDense, self).__init__(name="locally_connected_dense", **kwargs)
        self.__units = units
        self.__activation = self.__get_activation(activation)
        self.__w = self.__b = self.__cond = self.__where_y = None

    def build(self, input_shape):
        self.__w = self.add_weight(
            name="w",
            shape=(input_shape[1], self.__units),
            initializer=self.__locally_dense_initializer)

        self.__cond = tf.convert_to_tensor(self.__assign_with_bfs(np.ones(shape=self.__w.shape, dtype=np.bool), self.__w.shape, False))
        self.__where_y = tf.zeros(shape=self.__w.shape)

        self.__b = self.add_weight(
            name="b",
            shape=(self.__units,),
            initializer=tf.keras.initializers.get("zeros"))

        super().build(input_shape)
        self.built = True

    def __call__(self, *args, **kwargs):
        self.build((None,) + args[0].shape if args[0].shape[0] is not None else args[0].shape)
        return self.call(args[0])

    @tf.autograph.experimental.do_not_convert
    def call(self, x, **kwargs):
        cal_kernel = tf.where(self.__cond, self.__w, self.__where_y)
        output = self.__activation(tf.matmul(x, cal_kernel, name="locally_dense") + self.__b)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.__units

    def __get_activation(self, activation):
        if activation is None:
            return tf.keras.activations.linear
        elif type(activation) is str:
            if activation == "":
                return tf.keras.activations.linear
            return tf.keras.activations.get(activation)
        else:
            return activation

    def __locally_dense_initializer(self, shape, dtype=None):
        kernel = self.__assign_with_bfs(np.random.random(shape), shape, np.nan)
        return tf.convert_to_tensor(kernel, dtype=dtype)

    def __assign_with_bfs(self, matrix, shape, value):
        rows, cols = shape[0], shape[1]

        queue_0, queue_1 = [[0, 0]], [[rows - 1, cols - 1]]
        while True:
            index_0, index_1 = queue_0.pop(0), queue_1.pop(0)
            matrix[index_0[0], index_0[1]] = value
            matrix[index_1[0], index_1[1]] = value
            if index_0[0] + 1 == rows - 1 or index_0[1] + 1 == cols - 1 or index_1[0] - 1 == 0 or index_1[1] - 1 == 0:
                break
            else:
                queue_0.append([index_0[0] + 1, index_0[1]])
                queue_0.append([index_0[0], index_0[1] + 1])
                queue_1.append([index_1[0] - 1, index_1[1]])
                queue_1.append([index_1[0], index_1[1] - 1])

        return matrix
