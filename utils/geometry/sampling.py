import torch
import torch.nn.functional as NF
from .transforms import *

def sample_points(world_p: torch.Tensor, 
                  points: torch.Tensor)->torch.Tensor:
    '''
        World P: NxHxWx3 shaped tensor
        Points: NxHxWxPx2 shaped tensor

    Returns:
        NxHxWxCxA shaped tensor

    '''
    N, H, W, _ = world_p.shape
    # NxHWxAx2
    flat_points = points.view(N, H*W, -1, 2)

    sampled = NF.grid_sample(to_bchw(world_p), flat_points, align_corners=False).view(N, H, W, 3, -1)
    return sampled

def sample_features(features: torch.Tensor, 
                    points: torch.Tensor)->torch.Tensor:
    '''
        Features: NxCxHxW shaped tensor
        Points: NxHxWxPx2 shaped tensor
    Returns:
        NxHxWxCxA shaped tensor
    '''
    N, C, H, W = features.shape

    # NxCxHxW
    sampled = NF.grid_sample(features, points, align_corners=True)
    return sampled

def sample_patches(features: torch.Tensor, 
                   patches: torch.Tensor)->torch.Tensor:
    '''
        Features: NxCxHxW shaped tensor
        Points: NxHxWxPx2 shaped tensor
    Returns:
        NxHxWxCxA shaped tensor
    '''
    N, H, W, P, _ = patches.shape
    C = features.shape[1]

    # NxCxHxW
    sampled = NF.grid_sample(features, patches.reshape(N, H, W*P, 2).clamp(-2, 2), align_corners=True).reshape(N, C, H, W, P)
    return sampled