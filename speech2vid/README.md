# (Modified) Speech2Vid: The Lip Generator

## Introduction

Wav2Lip describes a GAN which outputs lip movements for a given audio clip. The GAN uses an expert discriminator in its loss function during training to optimize the network to generate lip movements in sync with audio. We use Syncnet for our expert discriminator which generates a probability of how in-sync audio and video are. (See `anime-dub/syncnet` for implementation details. To create this GAN Wav2Lip cites a paper which modifies [Speech2Vid](https://arxiv.org/pdf/1705.02966.pdf).

## Speech2Vid Modifications

### Architechture

Our Architechture for Speech2Vid appears as follows:

![speech2vid modified](https://user-images.githubusercontent.com/70986035/115456220-ec15b280-a1f0-11eb-9612-d0250269b643.png)

We have modified the proposed visual encoding architechture to injest 5 frames at a 25Hz sample rate and audio architechture to injest a 12 x 20 x 1 MFCC map corresponding to 0.2 seconds at a 100Hz sample rate. The output of this network is also 5 new frames rather than a single frame. The rationale for these changes is 1) Input Modification-- so the dataprocessing for the input to both Syncnet and Speech2Vid can be shared. 2) Output Modification-- 5 frame output so it can be easily fed into Syncnet's visual model.

### Loss Function

Speech2Vid uses an L1 Loss function. We have implemented a custom loss function which takes in the output video and the ground truth audio, feeds these into SyncNet to return the probability the audio/visual pair is synced and implement a BCE loss calculation on this. This implementation follows the equation below (from Wav2Lip).

<img width="334" alt="Screen Shot 2021-04-20 at 6 08 15 PM" src="https://user-images.githubusercontent.com/70986035/115469858-62231500-a203-11eb-927d-37e97413971d.png">

### Model Summary
<img width="829" alt="speech2vid_1" src="https://user-images.githubusercontent.com/25395806/115795702-6c2d4b00-a39e-11eb-9747-b2f9b3462d57.png">
<img width="829" alt="speech2vid_2" src="https://user-images.githubusercontent.com/25395806/115795707-6f283b80-a39e-11eb-975b-92eca26223e5.png">
<img width="829" alt="speech2vid_3" src="https://user-images.githubusercontent.com/25395806/115795712-718a9580-a39e-11eb-93b0-8770f3c4963d.png">
<img width="829" alt="speech2vid_4" src="https://user-images.githubusercontent.com/25395806/115795720-74858600-a39e-11eb-8dde-60c515089bfa.png">
