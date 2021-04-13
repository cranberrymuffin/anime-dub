import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, ZeroPadding2D, Conv3D, ReLU, MaxPool3D, Dense, \
    BatchNormalization, Flatten, ZeroPadding3D
import tensorflow.keras.backend as K

from hyperparameters import learning_rate

# https://medium.com/predict/face-recognition-from-scratch-using-siamese-networks-and-tensorflow-df03e32f8cd0

audio_architecture = [
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

visual_architecture = [
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

visual_model = tf.keras.Sequential(visual_architecture)
audio_model = tf.keras.Sequential(audio_architecture)

visual_input = tf.keras.Input((5, 224, 224, 1))
audio_input = tf.keras.Input((13, 20, 1))

visual_output = visual_model(visual_input)
audio_output = audio_model(audio_input)

euclidean_distance = tf.keras.Lambda(lambda tensors: K.l2_normalize(tensors[0] - tensors[1]))([visual_output, audio_output])

outputs = Dense(1, activation=tf.keras.activations.sigmoid)(euclidean_distance)

sync_net = tf.keras.models.Model([visual_input, audio_input], outputs)
sync_net.compile( loss=tf.keras.losses.binary_crossentropy, optimizer=tf.keras.optimizers.Adam(lr=learning_rate))
