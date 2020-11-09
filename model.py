import tensorflow as tf

from custom.local import Local


def mnist_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        # (28, 28, 1) -> (28, 28, 8)
        tf.keras.layers.Conv2D(
            filters=8,
            kernel_size=3,
            padding="same",
            kernel_initializer="he_normal",
            use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        # (28, 28, 8) -> (14, 14, 8)
        tf.keras.layers.MaxPool2D(),
        # (14, 14, 8) -> (14, 14, 16)
        tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=3,
            padding="same",
            kernel_initializer="he_normal"),
        tf.keras.layers.ReLU(),
        # (14, 14, 16) -> (7, 7, 16)
        tf.keras.layers.MaxPool2D(),
        # (7, 7, 16) -> (5, 5, 32)
        tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=3,
            kernel_initializer="he_normal"),
        tf.keras.layers.ReLU(),
        # (5, 5, 32) -> (3, 3, 64)
        tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=3,
            kernel_initializer="he_normal"),
        tf.keras.layers.ReLU(),
        # Global Average Pooling
        tf.keras.layers.GlobalAvgPool2D(),
        tf.keras.layers.Dropout(.2),
        Local(units=32),
        tf.keras.layers.ReLU(),
        Local(units=10),
        tf.keras.layers.Softmax()
    ])
    model.compile(
        optimizer=tf.optimizers.SGD(
            learning_rate=.005,
            momentum=.9,
            nesterov=True),
        loss=tf.losses.categorical_crossentropy,
        metrics=["acc"])
    model.summary()

    return model
