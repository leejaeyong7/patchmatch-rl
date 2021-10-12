import torch
import torch.nn as nn
import torch.nn.functional as NF
import numpy as np


def CBR(kernel_size, input_channel, output_channel, strides, dilation=1):
    pad = ((kernel_size - 1) * dilation ) // 2
    return nn.Sequential(
        nn.Conv2d(input_channel, output_channel, kernel_size, strides, pad, dilation=dilation, bias=False),
        nn.BatchNorm2d(output_channel),
        nn.ReLU(inplace=True)
    )

class FPNFeatureExtractor(nn.Module):
    def __init__(self, hparams):
        super(FPNFeatureExtractor, self).__init__()
        self.hparams = hparams
        # input images pass through
        ######################
        # feature extraction
        # input : N x 3 x W x H channel images
        # output: N x 32 x W/4 x H/4 channel
        # features
        input_channel = 3
        base_channel = self.hparams.feature_extractor_channel_scale

        # aggregation
        # input x=>
        self.conv0_0 = CBR(7, input_channel, base_channel * 1, 1, dilation=2)
        self.conv0_1 = CBR(3, base_channel * 1, base_channel * 1, 1)

        self.conv1_0 = CBR(5, base_channel * 1, base_channel * 2, 2)
        self.conv1_1 = CBR(3, base_channel * 2, base_channel * 2, 1)
        self.conv1_2 = CBR(3, base_channel * 2, base_channel * 2, 1)

        self.conv2_0 = CBR(5, base_channel * 2, base_channel * 4, 2)
        self.conv2_1 = CBR(3, base_channel * 4, base_channel * 4, 1)

        self.conv3_0 = CBR(5, base_channel * 4, base_channel * 8, 2)
        self.conv3_1 = CBR(3, base_channel * 8, base_channel * 8, 1)

        self.conv_l3 = nn.Conv2d(base_channel*8, base_channel*8, kernel_size=1)
        self.conv_l3_2 = nn.Conv2d(base_channel*8, base_channel*4, kernel_size=1)
        self.conv_l2 = nn.Conv2d(base_channel*4, base_channel*4, kernel_size=1)
        self.conv_l2_1 = nn.Conv2d(base_channel*4, base_channel*2, kernel_size=1)
        self.conv_l1 = nn.Conv2d(base_channel*2, base_channel*2, kernel_size=1)
        self.hparams.feature_extractor_output_channel = self.output_channel()

    def output_channel(self):
        return {
            3: self.hparams.feature_extractor_channel_scale * 8,
            2: self.hparams.feature_extractor_channel_scale * 4,
            1: self.hparams.feature_extractor_channel_scale * 2,
        }

    def group_corr(self):
        return self.hparams.feature_extractor_channel_scale

    def forward_layer(self, x):
        x = self.conv0_0(x)
        x = self.conv0_1(x)
        x = self.conv1_0(x)
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        l1 = x

        x = self.conv2_0(x)
        x = self.conv2_1(x)
        l2 = x

        x = self.conv3_0(x)
        x = self.conv3_1(x)
        l3 = x

        o3 = self.conv_l3(l3)
        o2 = self.conv_l3_2(NF.interpolate(o3, size=l2.shape[2:])) + self.conv_l2(l2)
        o1 = self.conv_l2_1(NF.interpolate(o2, size=l1.shape[2:])) + self.conv_l1(l1)

        return o3, o2, o1

        

    def forward(self, x):
        o3, o2, o1 = self.forward_layer(x)
        return {
            3: o3,
            2: o2,
            1: o1
        }