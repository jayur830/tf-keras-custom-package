import numpy as np
import tensorflow as tf


class LocallyConnectedDense(tf.keras.layers.Layer):
    def __init__(self,
                 units: int,
                 activation=None,
                 input_shape: tuple = None):
        super(LocallyConnectedDense, self).__init__()
        self.__units = units
        self.__activation = self.__get_activation(activation)
        self.__input_shape = input_shape
        self.__w = self.__b = self.__cond = self.__where_y = None

    def build(self, input_shape):
        self.__w = self.add_weight(
            name="w",
            shape=(input_shape[0], self.__units),
            initializer=self.__locally_dense_initializer)

        self.__cond = tf.convert_to_tensor(self.__assign_with_bfs(np.ones(shape=self.__w.shape, dtype=np.bool), self.__w.shape, False))
        self.__where_y = tf.zeros(shape=self.__w.shape)

        self.__b = self.add_weight(
            name="b",
            shape=(self.__units,),
            initializer=tf.keras.initializers.get("zeros"))

        super().build(input_shape if self.__input_shape is None else self.__input_shape)
        self.built = True

    @tf.autograph.experimental.do_not_convert
    def call(self, x, **kwargs):
        cal_kernel = tf.where(self.__cond, self.__w, self.__where_y)
        output = self.__activation(tf.matmul(tf.reshape(x, (1,) + x.shape), cal_kernel) + self.__b)
        return tf.reshape(output, output.shape[1:])

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
        print(shape)
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


if __name__ == '__main__':
    # (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
    # train_x = train_x.reshape((train_x.shape[0], train_x.shape[1] * train_x.shape[2])) / 255.
    # train_y = tf.keras.utils.to_categorical(train_y).astype("float32")
    # test_x = test_x.reshape((test_x.shape[0], test_x.shape[1] * test_x.shape[2])) / 255.
    # test_y = tf.keras.utils.to_categorical(test_y).astype("float32")
    #
    # print(f"train_x shape: {train_x.shape}")
    # print(f"train_y shape: {train_y.shape}")
    # print(f"test_x shape: {test_x.shape}")
    # print(f"test_y shape: {test_y.shape}")

    model = tf.keras.models.Sequential([
        # tf.keras.layers.Dense(
        #     units=5,
        #     activation="relu",
        #     input_shape=(7,)),
        LocallyConnectedDense(
            units=4,
            activation="linear",
            input_shape=(7,))
    ])
    model.compile(loss="mse")
    # model.build((7,))

    print(f"Before:\n{model.weights}")

    # model.fit(
    #     x=np.array([[1, 2, 3, 4, 5, 6, 7]]).astype("float32"),
    #     y=np.array([[2, 4, 6, 8]]).astype("float32"),
    #     batch_size=1,
    #     epochs=10)
    #
    # print(f"After:\n{model.weights}")

    # model.summary()

    # model.fit(
    #     x=train_x,
    #     y=train_y,
    #     batch_size=32,
    #     epochs=100,
    #     validation_split=0.2
    # )
