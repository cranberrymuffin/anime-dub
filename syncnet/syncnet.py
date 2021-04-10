import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv3D, MaxPool2D, MaxPool3D, Dropout, Flatten, Dense, BatchNormalization, ReLU, Activation
import hyperparameters as hp

## reference: https://medium.com/analytics-vidhya/syncnet-model-with-vidtimit-dataset-dd9de2cb2fb5

class VisualModel(tf.keras.Model):
    def __init__(self):
        super(VisualModel, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate)
        self.input_shape = ( 111, 111, 5)
        self.architecture = [
            # Conv1 Lip
            Conv2D(96, (3, 3), padding="valid", input_shape=input_shape, name="conv1_lip"),
            BatchNormalization(name="batch_norm1_lip"),
            ReLU(name="relu1_lip"),
            MaxPool2D(3, 2, padding="valid", name="pool1_lip"),

            # Conv2 Lip
            Conv2D(256, (5, 5), padding="valid", name="conv2_lip"),
            BatchNormalization(name="batch_norm2_lip"),
            ReLU(name="relu2"),
            MaxPool2D(3, 2, padding="valid", name="pool2_lip"),

            # Conv3 Lip
            Conv2D(512, (3, 3), padding="valid", name="conv3_lip"),
            BatchNormalization(name="batch_norm3_lip"),
            ReLU(name="relu3"),

            # Conv4 Lip
            Conv2D(512, (3, 3), padding="valid", name="conv4_lip"),
            BatchNormalization(name="batch_norm4_lip"),
            ReLU(name="relu4"),

            # Conv5 Lip
            Conv2D(512, (3, 3), padding="valid", name="conv5_lip"),
            BatchNormalization(name="batch_norm5_lip"),
            ReLU(name="relu5"),
            MaxPool2D(3, 3, padding="valid", name="pool5_lip"),
            Flatten(name="flatten_lip"),

            # fc6
            Dense(256, name="dense1_lip"),
            BatchNormalization(name="batch_norm6_lip"),
            ReLU(name="relu6"),

            # fc7
            Dense(128, name="dense2_lip"),
            BatchNormalization(name="batch_norm7_lip"),
            ReLU(name="relu7")
        ]

    def forward_pass(self, x):
        for layer in self.architecture:
            x = layer(x)

        return x


class AudioModel(tf.keras.Model):
    def __init__(self):
        super(AudioModel, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate)
        self.input_shape = (13, 20, 1)
        self.architecture = [
            # conv1
            Conv2D(64, (3, 3), padding='same', name='conv1_audio', input_shape=self.input_shape),
            BatchNormalization(name='bn1_audio'),
            ReLU(name='relu1_audio'),
            
            # conv2
            Conv2D(128, (3, 3), padding='same', name='conv2_audio'),
            BatchNormalization(name='bn2_audio'),
            ReLU(name='relu2_audio'),
            
            # pool2
            MaxPool2D(pool_size=(1, 3), strides=(1, 2), padding='valid', name='pool2_audio'),
            
            # conv3
            Conv2D(256, (3, 3), padding='same', name='conv3_audio'),
            BatchNormalization(name='bn3_audio'),
            ReLU(name='relu3_audio'),
            
            # conv4
            Conv2D(256, (3, 3), padding='same', name='conv4_audio'),
            BatchNormalization(name='bn4_audio'),
            ReLU(name='relu4_audio'),
            
            # conv5
            Conv2D(256, (3, 3), padding='same', name='conv5_audio'),    
            BatchNormalization(name='bn5_audio'),   
            ReLU(name='relu5_audio'),

            # pool5
            MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool5_audio'),

            Flatten(name='flatten_audio'),

            # fc6
            Dense(256, name='fc6_audio'),
            BatchNormalization(name='bn6_audio'),
            ReLU(name='relu6_audio'),

            # fc7
            Dense(128, name='fc7_audio'),
            BatchNormalization(name='bn7_audio'),
            ReLU(name='relu7_audio'),
        ]

    """ Passes input audio through the network. """
    def forward(self, x):

        for layer in self.architecture:
            x = layer(x)

        return x
