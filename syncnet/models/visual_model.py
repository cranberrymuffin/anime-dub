import tensorflow as tf
from tensorflow.keras.layers import Conv3D, ReLU, MaxPool3D, Dense, BatchNormalization, Flatten, ZeroPadding3D

import hyperparameters as hp


class VisualModel(tf.keras.Model):
    def __init__(self):
        super(VisualModel, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate)
        self.architecture = [
            Conv3D(96, kernel_size=(5, 7, 7), strides=(1, 2, 2), data_format="channels_last"),
            ZeroPadding3D(padding=0),
            BatchNormalization(),
            ReLU(),
            MaxPool3D(pool_size=(1, 3, 3), strides=(1, 2, 2)),

            Conv3D(256, kernel_size=(1, 5, 5), strides=(1, 2, 2), data_format="channels_last"),
            ZeroPadding3D(padding=(0, 1, 1)),
            BatchNormalization(),
            ReLU(),
            MaxPool3D(pool_size=(1, 3, 3), strides=(1, 2, 2)),
            ZeroPadding3D(padding=(0, 1, 1)),

            Conv3D(256, kernel_size=(1, 3, 3), data_format="channels_last"),
            ZeroPadding3D(padding=(0, 1, 1)),
            BatchNormalization(),
            ReLU(),

            Conv3D(256, kernel_size=(1, 3, 3), data_format="channels_last"),
            ZeroPadding3D(padding=(0, 1, 1)),
            BatchNormalization(),
            ReLU(),

            Conv3D(256, kernel_size=(1, 3, 3), data_format="channels_last"),
            ZeroPadding3D(padding=(0, 1, 1)),
            BatchNormalization(),
            ReLU(),
            MaxPool3D(pool_size=(1, 3, 3), strides=(1, 2, 2)),

            Conv3D(512, kernel_size=(1, 6, 6), data_format="channels_last"),
            ZeroPadding3D(padding=0),
            BatchNormalization(),
            ReLU(),

            Flatten(),
            Dense(512),
            BatchNormalization(),
            ReLU(),
            Dense(1024),
        ]

    """ Passes input video through the network. """

    def call(self, x, **kwargs):
        for layer in self.architecture:
            x = layer(x)

        return x
