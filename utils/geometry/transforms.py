import torch
import torch.nn.functional as NF

def from_homogeneous(points):
    return points[:, :, :, :-1] / points[:, :, :, -1:]

def to_homogeneous(points):
    dims = list(points.shape)
    dims[3] = 1
    ones = torch.ones(tuple(dims), dtype=points.dtype, device=points.device)
    return torch.cat((points, ones), 3)

def to_vector(points):
    return points.squeeze(-1)

def from_vector(points):
    return points.unsqueeze(-1)

def to_bchw(bhwc):
    return bhwc.permute(0, 3, 1, 2)

def to_bhwc(bchw):
    return bchw.permute(0, 2, 3, 1)

def resize_bhwc(tensor, size, mode='nearest'):
    return to_bhwc(NF.interpolate(to_bchw(tensor), size=size, mode=mode))