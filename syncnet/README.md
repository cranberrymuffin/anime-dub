# Syncnet: The Expert Discriminator

## Introduction

Wav2Lip describes an expert discriminator which determines how synced up a visual clip of a person speaking is to audio. This expert discriminator was derived from [SyncNet](https://www.robots.ox.ac.uk/~vgg/publications/2016/Chung16a/chung16a.pdf). This folder contains our implementation of SyncNet. The architechture of Syncnet contains 2 models, a visual model and an audio model. The 2 models have a shared loss function which attempts to minimize the distance between the output of the 2 networks for true pairs. This type of model with a shared loss function is called a siamese model. The shared loss function is called a contrastive loss function as it is responsible for comparing the outputs of the two models.

<img width="398" alt="origional" src="https://user-images.githubusercontent.com/70986035/114319372-3c00c500-9adf-11eb-9389-3c415fcef3a5.png">

## Data Processing

Syncnet's audio model requires an input of 0.2 second audio clips at a 100Hz sample rate converted into 13x20 [mel-frequency cepstrum (MFCC)](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum) maps. MFCC maps are commonly used as features in speech recognition systems. MFCC maps in our code are represented as 13 x 20 x 1 numpy arrays containing floats.

Syncnet's visual model requires an input of 5 frames. These 5 frames represent 0.2 seconds of a visual clip at a 25Hz sample rate (or 25 fps). These 5 frames are black and white 244 x 244 and cropped to the mouth.

Converting audio and video to this format can be found in `/preprocess.py`

After processing video and audio into their specified formats of 0.2 second chunks, we keep half the data in order and treat them as "true pairs" and shuffle the matches of the second half to unsync the audio and video and treat these as false pairs. We assign labels of 1 to true pairs and labels of 0 to false pairs.

## Training

We feed batches of 8 inputs (0.2 second audio/visual representations) with an Adam optimizer at a 1e-4 learning rate. These hyperparameters can be found in the `anime-dub/syncnet/models/hyperparameters.py` file. The visual and audio architechture is a 1:1 tensorflow conversion of the [pytorch implementation](https://github.com/joonson/syncnet_python/blob/master/SyncNetModel.py) detailed in the code released with Syncnet.
