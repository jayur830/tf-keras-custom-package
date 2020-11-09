import tensorflow as tf
import numpy as np

from custom.local import Local

if __name__ == '__main__':
    model = tf.keras.models.Sequential([
        Local(units=3, activation="elu", input_shape=(3,))
    ])
    model.compile(optimizer="sgd", loss="mse")
    model.summary()

    x = np.array([[.1, .2, .3]])
    model.fit(
        x=x,
        y=np.array([[.2, .4, .6]]),
        batch_size=1,
        epochs=200)

    print(f"x >> {x}")
    print(f"predict >> {model.predict(x)}")
