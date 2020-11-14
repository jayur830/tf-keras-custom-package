import tensorflow as tf

from tensorflow.core.framework.node_def_pb2 import NodeDef


class Residual:
    def __init__(self, kernel_size, kernel_initializer, activation):
        self.__kernel_size = kernel_size
        self.__kernel_initializer = kernel_initializer
        self.__activation = activation

    def __call__(self, *args, **kwargs):
        if len(args) == 0:
            raise tf.errors.InvalidArgumentError(
                node_def=NodeDef,
                op=tf.Operation,
                message="Arguments is empty.")
        elif len(args) != 1:
            for i in range(len(args) - 1):
                if args[i].output_shape != args[i + 1].output_shape:
                    raise tf.errors.InvalidArgumentError(
                        node_def=NodeDef,
                        op=tf.Operation,
                        message="Invalid shapes.")
                elif len(args[i].output_shape) != 4 or len(args[i].output_shape) != 4:
                    raise tf.errors.InvalidArgumentError(
                        node_def=NodeDef,
                        op=tf.Operation,
                        message="Dimensions of arguments must be 4 Dims.")
                elif not isinstance(args[i], tf.keras.layers.Layer) or not isinstance(args[i + 1], tf.keras.layers.Layer):
                    raise tf.errors.InvalidArgumentError(
                        node_def=NodeDef,
                        op=tf.Operation,
                        message="The argument is not instance of [tf.keras.layers.Layer].")
            layer_1 = tf.keras.layers.Add()(args)
        else:
            if len(args[0].output_shape) != 4:
                raise tf.errors.InvalidArgumentError(
                    node_def=NodeDef,
                    op=tf.Operation,
                    message="Dimensions of arguments must be 4 Dims.")
            layer_1 = args[0]
        layer_0 = tf.keras.layers.Conv2D(
            filters=int(layer_1.output_shape[3] / 4),
            kernel_size=1,
            kernel_initializer=self.__kernel_initializer)(layer_1)
        layer_0 = tf.keras.layers.BatchNormalization()(layer_0)
        layer_0 = tf.keras.layers.Activation(activation=self.__activation)(layer_0)
        layer_0 = tf.keras.layers.Conv2D(
            filters=int(layer_1.output_shape[3] / 4),
            kernel_size=1,
            kernel_initializer=self.__kernel_initializer)(layer_0)
        layer_0 = tf.keras.layers.BatchNormalization()(layer_0)
        layer_0 = tf.keras.layers.Activation(activation=self.__activation)(layer_0)
        layer_0 = tf.keras.layers.Conv2D(
            filters=int(layer_1.output_shape[3]),
            kernel_size=self.__kernel_size,
            padding="same",
            kernel_initializer=self.__kernel_initializer)(layer_0)
        layer_0 = tf.keras.layers.BatchNormalization()(layer_0)
        return tf.keras.layers.Add([layer_0, layer_1])


class InceptionBlock:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass
