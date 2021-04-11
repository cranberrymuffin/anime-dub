import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv3D, MaxPool2D, MaxPool3D, Flatten, Dense, BatchNormalization, ReLU
import hyperparameters as hp


# references: https://medium.com/analytics-vidhya/syncnet-model-with-vidtimit-dataset-dd9de2cb2fb5
#             https://github.com/joonson/syncnet_python/blob/master/SyncNetModel.py

class VisualModel(tf.keras.Model):
    def __init__(self):
        super(VisualModel, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate)
        self.architecture = [
            # Conv1 Lip
            Conv3D(96, kernel_size=(5, 7, 7), strides=(1, 2, 2), padding="valid", input_shape=(1, 120, 120, 5, 1),
                   name="conv1_lip"),
            BatchNormalization(name="batch_norm1_lip"),
            ReLU(name="relu1_lip"),
            MaxPool3D(pool_size=(1, 3, 3), strides=(1, 2, 2), name="pool1_lip"),

            # Conv2 Lip
            Conv3D(256, kernel_size=(1, 5, 5), strides=(1, 2, 2), padding="same", name="conv2_lip"),
            BatchNormalization(name="batch_norm2_lip"),
            ReLU(name="relu2"),
            MaxPool3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding="same"),

            # Conv3 Lip
            Conv3D(256, kernel_size=(1, 3, 3), padding="same", name="conv3_lip"),
            BatchNormalization(name="batch_norm3_lip"),
            ReLU(name="relu3"),

            # Conv4 Lip
            Conv3D(256, kernel_size=(1, 3, 3), padding="same", name="conv4_lip"),
            BatchNormalization(name="batch_norm4_lip"),
            ReLU(name="relu4"),

            # Conv5 Lip
            Conv3D(256, kernel_size=(1, 3, 3), padding="same", name="conv5_lip"),
            BatchNormalization(name="batch_norm5_lip"),
            ReLU(name="relu5"),
            MaxPool3D(pool_size=(1, 3, 3), strides=(1, 2, 2), name="pool5_lip"),
            Flatten(name="flatten_lip"),

            # Conv6 Lip
            Conv3D(512, kernel_size=(1, 6, 6), padding="valid", name="conv6_lip"),
            BatchNormalization(name="batch_norm6_lip"),
            ReLU(name="relu6"),

            Flatten(name="flatten_lip"),

            Dense(512, name='fc7_lip'),
            BatchNormalization(name='bn7_lip'),
            ReLU(name='relu7_lip'),

            Dense(1024, name='fc7_lip')
        ]

    """ Passes input video through the network. """
    def call(self, x):
        for layer in self.architecture:
            x = layer(x)

        return x


class AudioModel(tf.keras.Model):
    def __init__(self):
        super(AudioModel, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate)
        self.architecture = [
            # conv1
            Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), name='conv1_audio',
                   input_shape=(13, 20, 1)),
            BatchNormalization(name='bn1_audio'),
            ReLU(name='relu1_audio'),
            MaxPool2D(pool_size=(1, 1), strides=(1, 1)),

            # conv2
            Conv2D(192, kernel_size=(3, 3), strides=(1,1), padding=(1, 1), name='conv2_audio'),
            BatchNormalization(name='bn2_audio'),
            ReLU(name='relu2_audio'),
            MaxPool2D(pool_size=(3, 3), strides=(1, 2), name='pool2_audio'),

            # conv3
            Conv2D(384, kernel_size=(3, 3), padding=(1, 1), name='conv3_audio'),
            BatchNormalization(name='bn3_audio'),
            ReLU(name='relu3_audio'),

            # conv4
            Conv2D(256, kernel_size=(3, 3), padding=(1, 1), name='conv4_audio'),
            BatchNormalization(name='bn4_audio'),
            ReLU(name='relu4_audio'),

            # conv5
            Conv2D(256, kernel_size=(3, 3), padding=(1, 1), name='conv5_audio'),
            BatchNormalization(name='bn5_audio'),
            ReLU(name='relu5_audio'),
            MaxPool2D(pool_size=(3, 3), strides=(2, 2), name='pool5_audio'),

            # conv6
            Conv2D(512, kernel_size=(5, 4), padding=(0, 0), name='conv6_audio'),
            BatchNormalization(name='bn6_audio'),
            ReLU(name='relu6_audio'),

            Flatten(name='flatten_audio'),

            # fc7
            Dense(512, name='fc7_audio'),
            BatchNormalization(name='bn7_audio'),
            ReLU(name='relu7_audio'),

            # fc8
            Dense(1024, name='fc8_audio')
        ]

    """ Passes input audio through the network. """
    def call(self, x):
        for layer in self.architecture:
            x = layer(x)

        return x
