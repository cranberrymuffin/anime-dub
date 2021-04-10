import torch.nn as nn

class LipSyncEvalNet(nn.Module):
    def syncnet_visual():
        # conv1 3x3 96
        # pool1 3x3
        # conv2 3x3 256
        # pool2 3x3
        # conv3 3x3 512
        # conv4 3x3 512
        # conv5 3x3 512
        # pool5 3x3
        # fc6   6x6 4096
        # fc7   1x1 256

