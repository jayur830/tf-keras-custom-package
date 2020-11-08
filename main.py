import tensorflow as tf

from custom.locally_connected_dense import LocallyConnectedDense


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
        #     units=4,
        #     activation="linear",
        #     input_shape=(7,)),
        LocallyConnectedDense(
            units=4,
            activation="linear",
            input_shape=(7,))
    ])
    model.compile(loss="mse")
    model.build((7,))
    #
    # print(f"Before:\n{model.weights}")

    # print(model.input_shape)

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
