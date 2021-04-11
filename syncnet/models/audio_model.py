import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization

import hyperparameters as hp


class AudioModel(tf.keras.Model):
    def __init__(self):
        super(AudioModel, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate)
        self.architecture = [
            # conv1
            Conv2D(96, kernel_size=(3, 3), strides=(1, 1), padding="same", name='conv1_audio',
                   input_shape=(1, 13, 20, 1), activation="relu"),
            BatchNormalization(name='bn1_audio'),

            # conv2
            Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding="same", name='conv2_audio', activation="relu"),
            BatchNormalization(name='bn2_audio'),
            MaxPool2D(pool_size=(1, 3), strides=(1, 2), name='pool2_audio'),

            # conv3
            Conv2D(512, kernel_size=(3, 3), padding="same", name='conv3_audio', activation="relu"),
            BatchNormalization(name='bn3_audio'),

            # conv4
            Conv2D(512, kernel_size=(3, 3), padding="same", name='conv4_audio', activation="relu"),
            BatchNormalization(name='bn4_audio'),

            # conv6
            Conv2D(512, kernel_size=(5, 4), name='conv6_audio', activation="relu"),
            BatchNormalization(name='bn6_audio'),
            MaxPool2D(pool_size=(1, 3), strides=(1, 2), name='pool6_audio'),

            Flatten(name='flatten_audio'),

            # fc7
            Dense(4096, name='fc7_audio', activation="relu"),
            BatchNormalization(name='bn7_audio'),

            # fc8
            Dense(256, name='fc8_audio', activation='relu')
        ]

    """ Passes input audio through the network. """

    def call(self, x, **kwargs):
        for layer in self.architecture:
            x = layer(x)

        return x
