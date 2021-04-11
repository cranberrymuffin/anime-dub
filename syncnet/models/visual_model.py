import tensorflow as tf
from tensorflow.keras.layers import Conv3D, MaxPool3D, Flatten, Dense, BatchNormalization

import hyperparameters as hp


class VisualModel(tf.keras.Model):
    def __init__(self):
        super(VisualModel, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate)
        self.architecture = [
            # Conv1 Lip
            Conv3D(96, kernel_size=(5, 3, 3), strides=(1, 2, 2), input_shape=(1, 5, 120, 120, 1),
                   name="conv1_lip", data_format='channels_last', activation="relu"),
            BatchNormalization(name="batch_norm1_lip"),
            MaxPool3D(pool_size=(1, 3, 3), strides=(1, 2, 2), name="pool1_lip"),

            # Conv2 Lip
            Conv3D(256, kernel_size=(1, 3, 3), strides=(1, 2, 2), padding="same", name="conv2_lip",
                   data_format='channels_last', activation="relu"),
            BatchNormalization(name="batch_norm2_lip"),
            MaxPool3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding="same"),

            # Conv3 Lip
            Conv3D(512, kernel_size=(1, 3, 3), padding="same", name="conv3_lip", data_format='channels_last',
                   activation="relu"),
            BatchNormalization(name="batch_norm3_lip"),

            # Conv4 Lip
            Conv3D(512, kernel_size=(1, 3, 3), padding="same", name="conv4_lip", data_format='channels_last',
                   activation="relu"),
            BatchNormalization(name="batch_norm4_lip"),

            # Conv5 Lip
            Conv3D(512, kernel_size=(1, 3, 3), padding="same", name="conv5_lip", data_format='channels_last',
                   activation="relu"),
            BatchNormalization(name="batch_norm5_lip"),
            MaxPool3D(pool_size=(1, 3, 3), strides=(1, 2, 2), name="pool5_lip"),

            Flatten(name="flatten_lip"),

            Dense(4096, name='fc7_lip', activation="relu"),
            BatchNormalization(name='bn7_lip'),

            Dense(256, name='fc7_lip', activation='relu')
        ]

    """ Passes input video through the network. """

    def call(self, x, **kwargs):
        for layer in self.architecture:
            x = layer(x)

        return x
