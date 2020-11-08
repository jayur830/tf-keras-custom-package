import tensorflow as tf
import numpy as np

from custom.locally_connected_dense import LocallyConnectedDense

if __name__ == '__main__':
    # locally_connected_dense = LocallyConnectedDense(units=2, input_shape=(3,))
    # print("Sequential")
    # model = tf.keras.models.Sequential([locally_connected_dense])
    # print("build")
    # model.build((1, 3))
    # # model.summary()
    x = np.array([[1, 2, 3, 4, 5, 6, 7]]).astype("float32")
    y = np.array([[2, 4, 8, 16]]).astype("float32")

    input_layer = tf.keras.layers.Input(shape=(7,))
    dense = LocallyConnectedDense(units=4)
    model = tf.keras.models.Sequential([input_layer, dense])
    model.compile(loss="mae")
    model.summary()

    print(dense.trainable_weights)
    print(model.trainable_weights)

    # print(f"<Before>\n{dense.trainable_weights}")
    # model.fit(x=x, y=y, batch_size=1, epochs=10)
    # print(f"<After>\n{dense.trainable_weights}")
