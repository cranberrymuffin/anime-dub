# Syncnet: The Expert Discriminator

## Introduction

Wav2Lip describes an expert discriminator which determines how synced up a visual clip of a person speaking is to audio. This expert discriminator was derived from [SyncNet](https://www.robots.ox.ac.uk/~vgg/publications/2016/Chung16a/chung16a.pdf). This folder contains our implementation of SyncNet. The architechture of Syncnet contains 2 models, a visual model and an audio model. The 2 models have a shared loss function which attempts to minimize the distance between the output of the 2 networks for true pairs. This type of model with a shared loss function is called a siamese model. The shared loss function is called a contrastive loss function as it is responsible for comparing the outputs of the two models.

<img width="398" alt="origional" src="https://user-images.githubusercontent.com/70986035/114319372-3c00c500-9adf-11eb-9389-3c415fcef3a5.png">
