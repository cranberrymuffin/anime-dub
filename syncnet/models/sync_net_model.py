import datetime

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, ZeroPadding2D, Conv3D, ReLU, MaxPool3D, Dense, \
    BatchNormalization, Flatten, ZeroPadding3D
import tensorflow.keras.backend as K
import time
import tensorflow.keras.metrics as metrics
from .hyperparameters import learning_rate, batch_size, epochs


# https://medium.com/predict/face-recognition-from-scratch-using-siamese-networks-and-tensorflow-df03e32f8cd0

class SyncNet(object):
    def __init__(self, checkpoint_path=None):
        if checkpoint_path is not None:
            print("Setting model from saved checkpoint at " + checkpoint_path)
            self.model = tf.keras.models.load_model(checkpoint_path)
        else:
            self.audio_architecture = [
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

            self.visual_architecture = [
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

            visual_model = tf.keras.Sequential(self.visual_architecture)
            audio_model = tf.keras.Sequential(self.audio_architecture)

            visual_input = tf.keras.Input((5, 224, 224, 1))
            audio_input = tf.keras.Input((13, 20, 1))

            visual_output = visual_model(visual_input)
            audio_output = audio_model(audio_input)

            euclidean_distance = tf.keras.layers.Lambda(lambda tensors: K.l2_normalize(tensors[0] - tensors[1], axis=1))(
                [visual_output, audio_output])

            outputs = Dense(1, activation=tf.keras.activations.sigmoid)(0.00000001 + euclidean_distance)

            self.model = tf.keras.models.Model([audio_input, visual_input], outputs)
        self.model.compile(loss=tf.keras.losses.binary_crossentropy,
                           optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                           metrics=['accuracy'])

    def train(self, visual_inputs, audio_inputs, labels):
        print(audio_inputs.shape)
        print(visual_inputs.shape)
        inputs = [audio_inputs, visual_inputs]
        initial_time = time.time()
        self.model.summary()

        #log_dir = "logs/fit/human/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        self.model.fit(inputs, labels,
                       batch_size=batch_size,
                       epochs=epochs,
         #              callbacks=[tensorboard_callback]
                       )
        final_time = time.time()
        eta = (final_time - initial_time)
        time_unit = 'seconds'
        if eta >= 60:
            eta = eta / 60
            time_unit = 'minutes'
        print('Elapsed time acquired for {} epoch(s) -> {} {}'.format(epochs, eta, time_unit))

    def evaluate(self, video_inputs, audio_inputs, labels):
        self.model.evaluate([audio_inputs, video_inputs], labels, batch_size=batch_size)

    def summary(self):
        self.model.summary()

    def save_model(self, file_path):
        self.model.save(file_path)

    def load_model(self, file_path):
        self.model.load_model(file_path)

    def predict(self, input, steps):
        return self.model.predict(input, steps=steps)
