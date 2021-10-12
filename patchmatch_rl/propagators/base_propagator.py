import torch.nn as nn
import torch.nn.functional as NF
import torch
from utils.geometry import to_bchw, to_bhwc

class BasePropagator(nn.Module):
    def __init__(self, hparams=None):
        super(BasePropagator, self).__init__()
        self.hparams = hparams

        prop_shape = self.define_shape()

        # register buffers
        self.pad = self.create_pad(prop_shape)
        self.register_buffer('kernel', self.create_kernel(prop_shape, 4))

    def define_shape(self):
        raise NotImplementedError

    def create_pad(self, prop_shape):
        '''
        Given prop shape of KxK pixels, creates padding that makes the propagated
        shapes to be identical to original size.
        '''
        prop_size = prop_shape.shape[-1]
        return nn.ReflectionPad2d(prop_size // 2)

    def create_kernel(self, prop_shape, C=4):
        '''
        Given prop shape of KxK pixels, 
        creates kernel of shape Nx1xKxK that is used as convolutional filter
        '''
        KH, KW = prop_shape.shape
        valid_cells = prop_shape.nonzero()
        S = len(valid_cells)
        kernel = torch.zeros((S*C, C, KH, KW)).float()
        for i, valid_cell in enumerate(valid_cells):
            for d in range(C):
                kernel[C*i + d, d, valid_cell[1], valid_cell[0]] = 1
        return kernel


    def forward(self, propagatable):
        '''
        Given 1xHxWxK propagatable geometry, 
        propagate them to SxHxWxK neighbors
        '''
        p = to_bchw(propagatable)
        H, W = p.shape[-2:]
        # padded plane => 1x4xWxH => 1x1x4xHxW
        padded = self.pad(p)
        C = 4

        # returns 1xS4xHxW => Sx4xHxW
        propagated = to_bhwc(NF.conv2d(padded, self.kernel).view(-1, C, H, W))

        # ensure good views for all propagated values
        return propagated 
