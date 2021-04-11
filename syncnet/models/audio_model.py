import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, ReLU, ZeroPadding2D, BatchNormalization, Flatten, Dense
from syncnet import hyperparameters as hp


class AudioModel(tf.keras.Model):
    def __init__(self):
        super(AudioModel, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate)
        self.architecture = [
            Conv2D(64, kernel_size=(3, 3), strides=(1, 1)),
            ZeroPadding2D(padding=(1, 1)),
            BatchNormalization(),
            ReLU(),
            MaxPool2D(pool_size=(1, 1), strides=(1, 1)),

            Conv2D(192, kernel_size=(3, 3), strides=(1, 1)),
            ZeroPadding2D(padding=(1, 1)),
            BatchNormalization(),
            ReLU(),
            MaxPool2D(pool_size=(3, 3), strides=(1, 2)),

            Conv2D(384, kernel_size=(3, 3)),
            ZeroPadding2D(padding=(1, 1)),
            BatchNormalization(),
            ReLU(),

            Conv2D(256, kernel_size=(3, 3)),
            ZeroPadding2D(padding=(1, 1)),
            BatchNormalization(),
            ReLU(),

            Conv2D(256, kernel_size=(3, 3)),
            ZeroPadding2D(padding=(1, 1)),
            BatchNormalization(),
            ReLU(),
            MaxPool2D(pool_size=(3, 3), strides=(2, 2)),

            Conv2D(512, kernel_size=(5, 4)),
            ZeroPadding2D(padding=(0, 0)),
            BatchNormalization(),
            ReLU(),

            Flatten(),
            Dense(512),
            BatchNormalization(),
            ReLU(),
            Dense(1024),
        ]

    """ Passes input audio through the network. """

    def call(self, x, **kwargs):
        for layer in self.architecture:
            x = layer(x)

        return x
