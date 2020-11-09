import tensorflow as tf

from custom.mnist_model import mnist_model

if __name__ == '__main__':
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
    train_x = train_x.reshape(train_x.shape + (1,)) / 255.
    train_y = tf.keras.utils.to_categorical(train_y)
    test_x = test_x.reshape(test_x.shape + (1,)) / 255.
    test_y = tf.keras.utils.to_categorical(test_y)

    model = mnist_model()

    model.fit(
        x=train_x,
        y=train_y,
        batch_size=256,
        epochs=50,
        validation_split=.2)

    model.evaluate(
        x=test_x,
        y=test_y,
        batch_size=256)
