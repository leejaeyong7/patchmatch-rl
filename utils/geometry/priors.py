from .plane import *
from .transforms import *

def compute_priors(cameras, depth_maps):
    H, W = depth_maps.shape[1:3]
    world_p = cameras.back_project(depth_maps)
    world_u = cameras.up()
    projected = cameras.project(world_p, None)
    h_projected = from_vector(to_homogeneous(projected))
    # NxHxWx2
    warped = to_vector(cameras.normalize(h_projected))
    # obtain world points to camera vectors
    # NxHxWx3
    world_p_to_cam = to_vector(cameras.C) - world_p
    r = cameras.camera_rays()

    dists = to_vector(cameras.R @ from_vector(world_p) + cameras.t)[..., 2:]

    # NxHxWx3
    dirs =  NF.normalize(world_p_to_cam, dim=-1, p=2)

    scales = dists / dists[:1]
    tri_angles = (dirs * dirs[:1]).sum(-1, keepdim=True).clamp(-1, 1)
    up_diffs = (world_u * world_u[:1]).sum(-1).view(-1, 1, 1, 1).repeat(1, H, W, 1).clamp(-1, 1)

    # return stacked tensors
    priors = torch.cat((scales, tri_angles, up_diffs), dim=-1)
    return warped, to_bchw(priors)