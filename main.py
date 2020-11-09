import tensorflow as tf
import numpy as np

from custom.local import Local

if __name__ == '__main__':
    local_layer = Local(units=3, input_shape=(3,))
    model = tf.keras.models.Sequential([
        local_layer
    ])
    model.compile(optimizer="sgd", loss="mse")
    model.summary()

    print(local_layer.weights)

    x = np.array([[.1, .2, .3]])
    model.fit(
        x=x,
        y=np.array([[.2, .4, .6]]),
        batch_size=1,
        epochs=500)

    print(local_layer.weights)

    print(f"x >> {x}")
    print(f"predict >> {model.predict(x)}")
