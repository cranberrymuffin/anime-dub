# (Modified) Speech2Vid: The Lip Generator

## Introduction

Wav2Lip describes a GAN which outputs lip movements for a given audio clip. The GAN uses an expert discriminator in its loss function during training to optimize the network to generate lip movements in sync with audio. We use Syncnet for our expert discriminator which generates a probability of how in-sync audio and video are. (See `anime-dub/syncnet` for implementation details. To create this GAN Wav2Lip cites a paper which modifies [Speech2Vid](https://arxiv.org/pdf/1705.02966.pdf).

## Speech2Vid Modifications

### Architechture

Our Architechture for Speech2Vid appears as follows:

![speech2vid modified](https://user-images.githubusercontent.com/70986035/115456220-ec15b280-a1f0-11eb-9612-d0250269b643.png)

We have modified the proposed visual encoding architechture to injest 5 frames at a 25Hz sample rate and audio architechture to injest a 12 x 20 x 1 MFCC map corresponding to 0.2 seconds at a 100Hz sample rate. The output of this network is also 5 new frames rather than a single frame.
