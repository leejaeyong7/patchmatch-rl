import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as NF
from torch.nn import Module
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _quadruple
from torch.autograd import Variable
from torch.nn import Conv2d
import math

def conv4d(data,filters,bias,padding):
    b, c, h, w, d, t = data.shape
    oc, c, fh, fw, fd, ft = filters.shape
    ph, pw, pd, pt = padding

    oh = h - fh + 1 + ph * 2
    ow = w - fw + 1 + pw * 2
    od = d - fd + 1 + pd * 2
    ot = t - ft + 1 + pt * 2
    dev = data.device

    output = Variable(torch.zeros(b, oc, oh, ow, od, ot, device=dev))
    if((ph == 0) and (pw == 0) and (pd == 0) and (pt == 0)):
        padded = data
    else:
        padded = NF.pad(data, (ph, ph, pw, pw, pd, pd, pt, pt))
    '''
    O[OC, X, Y, Z]
    d[IC, X, Y, Z]
    for x, y, z:
        o[OC, A, B, C]
        # f = OC x IC x A x B x C
        for a, b, c:
            o[OC, a, b] += f[OC, IC, a, b, c] @ d[IC, 1, x+a, y+b, z+c]
        
        O[OC, x, y, z] = o[OC].sum(1, 2, 3)

    O[OC, X, Y, I, J]
    for j:
        for x, y, i, j:
            o[OC, A, B, C, D]
            for d:
                for a, b, c:
                    o[a, b, c, d] += d[x+a, y+b, i+c, j+d] * f[a, b, c, d]
        
    O[x, y, i, j] = o[A, B, C, D]
    '''

    for x in range(oh):
        for fx in range(fh):
            b = bias if fx == fh // 2 else None
            output[:, :, x] += NF.conv3d(padded[:, :, x + fx], filters[:, :, fx], bias=b, stride=1)

    return output

class Conv4d(_ConvNd):
    """Applies a 4D convolution over an input signal composed of several input
    planes.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, bias=True): 
        # stride, dilation and groups !=1 functionality not tested 
        stride=1
        dilation=1
        groups=1
        # zero padding is added automatically in conv4d function to preserve tensor size
        padding = _quadruple(padding)
        kernel_size = _quadruple(kernel_size)
        stride = _quadruple(stride)
        padding = _quadruple(padding)
        dilation = _quadruple(dilation)
        padding_mode = 'zeros'
        super(Conv4d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _quadruple(0), groups, bias, padding_mode)  

    def forward(self, input):
        return conv4d(input, self.weight, bias=self.bias, padding=self.padding)
