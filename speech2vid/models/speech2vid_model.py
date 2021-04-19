import time
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Conv2DTranspose, concatenate, \
    Reshape, BatchNormalization, Activation
from .hyperparameters import learning_rate, batch_size, epochs
from syncnet.models.sync_net_model import SyncNet
tf.executing_eagerly()
import numpy as np
import tensorflow.keras.backend as K
import math

# tensorflow model lifted from https://github.com/Sindhu-Hegde/you_said_that/blob/master/train.py
class Speech2Vid:
    def __init__(self, checkpoint_path=None, sync_net_path=None):
        self.sync_net = SyncNet(sync_net_path)
        if checkpoint_path is not None:
            print("Setting model from saved checkpoint at " + checkpoint_path)
        else:

            # Audio encoder
            input_audio = Input(shape=(13, 20, 1), batch_size=batch_size,)

            x = self.convolution(input_audio, 64, 3)
            x = self.convolution(x, 128, 3)
            x = MaxPooling2D((3, 3), strides=(1, 2), padding='same')(x)
            x = self.convolution(x, 256, 3)
            x = self.convolution(x, 256, 3)
            x = self.convolution(x, 512, 3)
            x = MaxPooling2D((3, 3), strides=2, padding='same')(x)
            x = Flatten()(x)
            x = Dense(512, activation='relu')(x)
            encoded_audio = Dense(256, activation='relu')(x)

            # Identity encoder
            input_identity = Input(shape=(112, 112, 15), batch_size=batch_size,)

            x = self.convolution(input_identity, 96, 7, 2)
            x_skip1 = MaxPooling2D((3, 3), strides=2, padding='same')(x)
            x_skip2 = self.convolution(x_skip1, 256, 5, 2)
            x_skip3 = MaxPooling2D((3, 3), strides=2, padding='same')(x_skip2)
            x = self.convolution(x_skip3, 512, 3)
            x = self.convolution(x, 512, 3)
            x = self.convolution(x, 512, 3)
            x = Flatten()(x)
            x = Dense(512, activation='relu')(x)
            encoded_identity = Dense(256, activation='relu')(x)

            # Concatenate the audio and identity features
            concatenated_features = concatenate([encoded_audio, encoded_identity])

            # Decoder
            x = Dense(98, activation='relu')(concatenated_features)
            x = Reshape((7, 7, 2))(x)
            x = self.transposed_convolution(x, 512, 6)
            x = self.transposed_convolution(x, 256, 5)
            x = concatenate([x, x_skip3])
            x = self.transposed_convolution(x, 96, 5, 2)
            x = concatenate([x, x_skip2])
            x = self.transposed_convolution(x, 96, 5, 2)
            x = concatenate([x, x_skip1])
            x = self.transposed_convolution(x, 64, 5, 2)
            decoded = Conv2DTranspose(15, (5, 5), strides=2, activation='sigmoid', padding='same')(x)

            self.__speech2vid_net = tf.keras.models.Model(inputs=[input_audio, input_identity], outputs=[decoded])
        
        def loss(audio_inputs, visual_inputs):
            print("INSIDE LOSS")
            faces = tf.stack(tf.split(visual_inputs, num_or_size_splits=5, axis=3), axis=1)
            blw_faces = tf.image.rgb_to_grayscale(faces)
            blw_mouths = blw_faces[:, :, 112//2:,:, :]
            resized_mouths = []
            for i, mouth in enumerate(blw_mouths):
                resized_mouths.append(tf.image.resize(mouth, [224, 224]))
            visual_inputs = tf.stack(resized_mouths)
            print(visual_inputs.get_shape())
            print(audio_inputs.get_shape())
            N = audio_inputs.get_shape()[0]
            Pi_sync = self.sync_net.model([audio_inputs, visual_inputs]).numpy()
            E = (1.0/N) * tf.reduce_sum(-K.log(Pi_sync))
            print(E)
            return E
        
        self.__speech2vid_net.compile(loss=loss,
                                      optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                                      metrics=['accuracy'],
                                      run_eagerly=True)
    
    @staticmethod
    def convolution(x, filters, kernel_size=3, strides=1, padding='same'):
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
        x = BatchNormalization(momentum=.8)(x)
        x = Activation('relu')(x)
        return x

    @staticmethod
    def transposed_convolution(x, filters, kernel_size=3, strides=1, padding='same'):
        x = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
        x = BatchNormalization(momentum=.8)(x)
        x = Activation('relu')(x)
        return x

    def train(self, visual_inputs, audio_inputs, labels):
        inputs = [audio_inputs, visual_inputs]
        initial_time = time.time()
        self.__speech2vid_net.summary()
        self.__speech2vid_net.fit(inputs, labels,
                                  batch_size=batch_size,
                                  epochs=epochs
                                  )
        final_time = time.time()
        eta = (final_time - initial_time)
        time_unit = 'seconds'
        if eta >= 60:
            eta = eta / 60
            time_unit = 'minutes'
        print('Elapsed time acquired for {} epoch(s) -> {} {}'.format(epochs, eta, time_unit))

    def evaluate(self, video_inputs, audio_inputs, labels):
        self.__speech2vid_net.evaluate([audio_inputs, video_inputs], labels, batch_size=batch_size)

    def summary(self):
        self.__speech2vid_net.summary()

    def save_model(self, file_path):
        self.__speech2vid_net.save(file_path)

    def load_model(self, file_path):
        self.__speech2vid_net.load_model(file_path)

    def predict(self, input):
        return self.__speech2vid_net.predict(input)
