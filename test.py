import tensorflow as tf

from custom.locally_connected_dense import LocallyConnectedDense

if __name__ == '__main__':
    locally_connected_dense = LocallyConnectedDense(units=2, input_shape=(3,))
    model = tf.keras.models.Sequential([locally_connected_dense])
    model.build((3,))
    model.summary()
