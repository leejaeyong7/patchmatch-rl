import torch
import torch.nn.functional as NF
from .base_propagator import BasePropagator
from utils.geometry import to_bchw, to_bhwc

class NeighPropagator(BasePropagator):
    def __init__(self, hparams=None):
        super(NeighPropagator, self).__init__(hparams)
        self.hparams = hparams
        self.dilation = hparams.patch_dilation
        prop_shape = self.define_shape()
        patch_shape = self.define_patch_shape()
        prop_shape_all = self.define_shape_all()
        self.register_buffer('point_kernel', self.create_kernel(prop_shape, 3))
        self.register_buffer('patch_kernel', self.create_kernel(patch_shape, 3))
        self.register_buffer('normal_kernel', self.create_kernel(prop_shape_all, 3))

    def define_patch_shape(self):
        D = self.dilation
        P = D * 2 + 1
        kernel = torch.zeros((P, P))
        kernel[0, 0] = 1
        kernel[0, D] = 1
        kernel[0, -1] = 1
        kernel[D, 0] = 1
        kernel[D, D] = 1
        kernel[D, -1] = 1
        kernel[-1, 0] = 1
        kernel[-1, D] = 1
        kernel[-1, -1] = 1
        return kernel

    def define_shape(self):
        return torch.FloatTensor([
            0, 1, 0,
            1, 0, 1,
            0, 1, 0,
        ]).view(3, 3)

    def define_shape_all(self):
        return torch.FloatTensor([
            1, 1, 1,
            1, 1, 1,
            1, 1, 1,
        ]).view(3, 3)


    def create_kernel(self, prop_shape, C = 1):
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

    def propagate_points(self, propagatable):
        '''
        Given 1xHxWxK propagatable geometry, 
        propagate them to SxHxWxK neighbors
        '''
        p = to_bchw(propagatable)
        H, W = p.shape[-2:]
        # padded plane => 1x4xWxH => 1x1x4xHxW
        padded = self.pad(p)
        C = 3

        # returns 1xS4xHxW => Sx4xHxW
        propagated = to_bhwc(NF.conv2d(padded, self.point_kernel).view(-1, C, H, W))

        # ensure good views for all propagated values
        return propagated 
    
    def propagate_planes(self, propagatable):
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
    
    def propagate_rays(self, propagatable):
        '''
        Given 1xHxWxK propagatable geometry, 
        propagate them to SxHxWxK neighbors
        '''
        p = to_bchw(propagatable)
        H, W = p.shape[-2:]
        # padded plane => 1x4xWxH => 1x1x4xHxW
        padded = self.pad(p)
        C = 3
        # returns 1xS4xHxW => Sx4xHxW
        propagated = to_bhwc(NF.conv2d(padded, self.normal_kernel).view(-1, C, H, W))

        # ensure good views for all propagated values
        return propagated 