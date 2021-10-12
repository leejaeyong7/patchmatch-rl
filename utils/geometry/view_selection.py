import torch
from .camera import MVSCamera
from .transforms import *
from .sampling import *
import torch.nn.functional as NF


def compute_visibility_map_for_ref(cameras: MVSCamera, depths:torch.Tensor):
    '''
    Depth map
        depths: NxHxWx1

    Visibility map
        Nx1xHxW
    '''
    # we consider it to be visible if depths are consistent
    d = to_bchw(depths)
    # project ref depth
    world_p = cameras.back_project(depths, None)

    # get depth values at the projected coordinates
    projected = cameras.project(world_p[:1], None)

    # NxHxWx2x1
    h_projected = from_vector(to_homogeneous(projected))
    n_projected = to_vector(cameras.normalize(h_projected))

    # NxHxWx1 depth values at src images, projected from ref image
    repr_p = to_bhwc(NF.grid_sample(to_bchw(world_p), n_projected, align_corners=False, mode='nearest'))

    # NxHxW
    visible = ((world_p[:1] - repr_p).sum(-1, keepdim=True).abs() < (depths[:1] * 0.01))
    return visible

def compute_warps(cameras, depth_maps):
    world_p = cameras.back_project(depth_maps[:1], 1)
    projected = cameras.project(world_p, None)
    h_projected = to_homogeneous(projected)
    # NxHxWx2
    warped = to_vector(cameras.normalize(from_vector(h_projected)))
    return warped

def compute_visibility_map(cameras: MVSCamera, depths):
    '''
    Depth map
        depths: NxHxWx1

    Visibility map
        Nx1xHxW
    '''
    world_p = cameras.back_project(depths, None)
    w = compute_warps(cameras, depths)

    # depths from source view based on homography
    #NxHxWx3
    # contains world point, obtained by backprojected ref projected points
    repr_p = to_bhwc(NF.grid_sample(to_bchw(world_p), w, align_corners=False, mode='nearest'))

    # contains true, if backprojected ref_depth-projected-point has depth
    valid_p = to_bhwc(NF.grid_sample(to_bchw(depths), w, align_corners=False, mode='nearest')) > 0

    # NxHxWx1
    proj_p= cameras.project(repr_p)
    proj_dist = (proj_p[:1] - proj_p[1:]).norm(p=2, dim=-1, keepdim=True)
    # p_dist = (repr_p[:1] - repr_p[1:]).norm(p=2, dim=-1, keepdim=True)

    # visible = p_dist < (depths[:1] * 0.02)
    valids = (valid_p[:1] & valid_p[1:])
    visible = (proj_dist < 2) & valids
    invisible = (proj_dist > 5) & valids

    return visible, invisible, valids
