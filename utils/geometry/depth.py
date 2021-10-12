from .transforms import *
from .sampling import sample_patches
import torch.nn.functional as NF

def resize_depth(depth, size):
    return NF.interpolate(depth, size, mode='nearest')

def resize_depth_by_fill(depth, size):
    _, _, OH, OW = depth.shape
    TH, TW = size
    kernel = (OH // TH, OW // TW)

    return NF.interpolate(NF.max_pool2d(depth, kernel), size, mode='nearest')

def compute_finite_diff_gradients(depth_map):
    '''
    Given Depth, compute dD/du, dD/dv
    '''
    # Nx3xHxW
    depth = to_bchw(depth_map)
    mask = depth != 0
    pxd = to_bhwc(NF.pad(depth, [1, 1, 0, 0]))
    pyd = to_bhwc(NF.pad(depth, [0, 0, 1, 1]))
    pxm = to_bhwc(NF.pad(mask, [1, 1, 0, 0]))
    pym = to_bhwc(NF.pad(mask, [0, 0, 1, 1]))

    # compute valid mask for normal
    mx1 = (pxm[:, :, 2:] & pxm[:, :, 1:-1]) 
    mx2 = (pxm[:, :, :-2] & pxm[:, :, 1:-1])
    my1 = (pym[:, 2:] & pym[:, 1:-1]) 
    my2 = (pym[:, :-2] & pym[:, 1:-1])

    # filter mask by limited depth diff
    ddx1 = (pxd[:, :, 2:] - pxd[:, :, 1:-1]) 
    ddx2 = (pxd[:, :, 1:-1] - pxd[:, :, :-2]) 
    ddy1 = (pyd[:, 2:] - pyd[:, 1:-1]) 
    ddy2 = (pyd[:, 1:-1] - pyd[:, :-2]) 
    mx = mx1 | mx2
    my = my1 | my2
    m = mx & my
    dxs = (ddx1 * mx1 + ddx2 * mx2) / (mx1.float() + mx2.float())
    dys = (ddy1 * my1 + ddy2 * my2) / (my1.float() + my2.float())
    return dxs, dys


def compute_normal_from_depth(camera, depth_map, apply_mask=True):
    # Nx3xHxW
    depth = to_bchw(depth_map)
    mask = depth != 0
    cam_r = to_bchw(camera.camera_rays(1))
    cam_p = cam_r * depth

    pxd = NF.pad(depth, [1, 1, 0, 0])
    pyd = NF.pad(depth, [0, 0, 1, 1])
    px = NF.pad(cam_p, [1, 1, 0, 0])
    py = NF.pad(cam_p, [0, 0, 1, 1])
    pxm = NF.pad(mask, [1, 1, 0, 0])
    pym = NF.pad(mask, [0, 0, 1, 1])

    # compute valid mask for normal
    mx1 = (pxm[:, :, :, 2:] & pxm[:, :, :, 1:-1]) 
    mx2 = (pxm[:, :, :, :-2] & pxm[:, :, :, 1:-1])
    my1 = (pym[:, :, 2:] & pym[:, :, 1:-1]) 
    my2 = (pym[:, :, :-2] & pym[:, :, 1:-1])

    # filter mask by limited depth diff
    ddx1 = (pxd[:, :, :, 2:] - pxd[:, :, :, 1:-1]) 
    ddx2 = (pxd[:, :, :, 1:-1] - pxd[:, :, :, :-2]) 
    ddy1 = (pyd[:, :, 2:] - pyd[:, :, 1:-1]) 
    ddy2 = (pyd[:, :, 1:-1] - pyd[:, :, :-2]) 
    if(apply_mask):
        mx1 = mx1 & (ddx1.abs() < pxd[:, :, :, 1:-1] * 0.05)
        mx2 = mx2 & (ddx2.abs() < pxd[:, :, :, 1:-1] * 0.05)
        my1 = my1 & (ddy1.abs() < pyd[:, :, 1:-1] * 0.05)
        my2 = my2 & (ddy2.abs() < pyd[:, :, 1:-1] * 0.05)
    mx = mx1 | mx2
    my = my1 | my2
    m = mx & my

    # compute finite diff gradients
    dx1 = (px[:, :, :, 2:] - px[:, :, :, 1:-1]) 
    dx2 = (px[:, :, :, 1:-1] - px[:, :, :, :-2]) 
    dy1 = (py[:, :, 2:] - py[:, :, 1:-1]) 
    dy2 = (py[:, :, 1:-1] - py[:, :, :-2]) 
    # dxs = dx1 * mx1 + dx2 * (mx2 & ~mx1)
    # dys = dy1 * my1 + dy2 * (my2 & ~my1)
    dxs = (dx1 * mx1 + dx2 * mx2) / (mx1.float() + mx2.float() + (~mx).float())
    dys = (dy1 * my1 + dy2 * my2) / (my1.float() + my2.float() + (~my).float())
    dx = NF.normalize(dxs, dim=1, p=2)
    dy = NF.normalize(dys, dim=1, p=2)

    # compute normal direction from cross products
    normal = torch.cross(dy, dx, dim=1)

    # flip normal based on camera view
    normal = NF.normalize(normal, p=2, dim=1)
    cam_dir = NF.normalize(cam_r, p=1, dim=1)

    dot = (cam_dir * normal).sum(1, keepdim=True)
    normal *= -dot.sign()

    if(apply_mask):
        nm = m.repeat(1, 3, 1, 1)
        normal[~nm] = 0
    return to_bhwc(normal)

def generate_random_depths(cameras, num_depths, min_d, max_d):
    H = cameras.H
    W = cameras.W
    dev = cameras.K.device
    # uniformly sample depth from uniformly distributed range
    depths = torch.linspace(1 / max_d, 1 / min_d, num_depths + 1, device=dev).view(-1, 1, 1, 1)
    # depths = torch.linspace(min_d, max_d, num_depths + 1, device=dev).view(-1, 1, 1, 1)
    min_ds = depths[:-1]
    max_ds = depths[1:]

    return 1 / (torch.rand((num_depths, H, W, 1), dtype=torch.float, device=dev) * (max_ds - min_ds) + min_ds)

def perturb(num_d, d_level, cameras, depth_map, ranges):
    # given current plane map, perform refinement
    min_d, max_d = ranges
    _, H, W, _ = depth_map.shape

    max_d = (depth_map * (1 + d_level)).clamp_max(max_d)
    min_d = (depth_map * (1 - d_level)).clamp_min(min_d)
    rand_pert = torch.rand((num_d, H, W, 1), device=depth_map.device)
    return (rand_pert* (max_d - min_d) + min_d)

def get_gt_depth(cameras, depths):
    gt_d = to_bhwc(resize_depth(to_bchw(depths), (cameras.H, cameras.W)))
    gt_d[gt_d.isnan()] = 0
    return gt_d

def warp_by_depth(cameras, depth_maps, features):
    '''
    Arguments:
        - cameras
    '''
    world_p = cameras.back_project(depth_maps)
    projected = cameras.project(world_p, None)
    h_projected = from_vector(to_homogeneous(projected))
    # NxHxWx2
    warped = to_vector(cameras.normalize(h_projected))[1:]
    fs = NF.grid_sample(features, warped, align_corners=True, mode='bilinear')
    mask = ((warped > 1) | (warped < -1)).any(-1).unsqueeze(1)
    return fs, ~mask

def compute_samples(cameras, depth_maps, features, view_w):
    '''
    Arguments:
        - cameras
    '''
    world_p = cameras.back_project(depth_maps)
    projected = cameras.project(world_p, None)
    h_projected = from_vector(to_homogeneous(projected))
    # NxHxWx2
    warped = to_vector(cameras.normalize(h_projected))[1:]

    view_coords = warped.gather(0, view_w.repeat(1, 1, 1, 2))
    _, C, N = features.shape[:3]
    batch_coords = (view_w / (N - 1) - 0.5) * 2
    coords = torch.cat((view_coords, batch_coords), dim=-1).unsqueeze(0)
    mask = ((coords > 1) | (coords < -1)).any(-1).repeat(C, 1, 1, 1).permute(1, 0, 2, 3)

    sampled = NF.grid_sample(features, coords, align_corners=True, mode='bilinear')[0].permute(1, 0, 2, 3)
    sampled[mask] = 0
    return sampled



def compute_samples_(cameras, depth_maps, features, view_w):
    '''
    Arguments:
        - cameras
    '''
    world_p = cameras.back_project(depth_maps)
    projected = cameras.project(world_p, None)
    h_projected = from_vector(to_homogeneous(projected))
    # NxHxWx2
    warped = to_vector(cameras.normalize(h_projected))[1:]

    view_coords = warped.gather(0, view_w.repeat(1, 1, 1, 2))
    _, C, N = features.shape[:3]
    
    ff = features[0].permute(1, 0, 2, 3)
    sampled = NF.grid_sample(ff, warped, align_corners=True, mode='bilinear')

    mask = ((warped > 1) | (warped < -1)).any(-1, keepdim=True).permute(0, 3, 1, 2).repeat(1, C, 1, 1)
    sampled[mask] = 0

    return sampled.gather(0, view_w.permute(0, 3, 1, 2).repeat(1, C, 1, 1))


def compute_samples_(cameras, depth_maps, features, view_w):
    '''
    Arguments:
        - cameras
    '''
    world_p = cameras.back_project(depth_maps)
    projected = cameras.project(world_p, None)
    h_projected = from_vector(to_homogeneous(projected))
    # NxHxWx2
    warped = to_vector(cameras.normalize(h_projected))[1:]

    view_coords = warped.gather(0, view_w.repeat(1, 1, 1, 2))
    _, C, N = features.shape[:3]
    batch_coords = (view_w / (N - 1) - 0.5) * 2
    coords = torch.cat((view_coords, batch_coords), dim=-1).unsqueeze(1)
    mask = ((coords > 1) | (coords < -1)).any(-1).repeat(1, C, 1, 1)
    mask2 = ((warped > 1) | (warped < -1)).any(-1, keepdim=True).permute(0, 3, 1, 2).repeat(1, C, 1, 1)



    
    ff = features[0].permute(1, 0, 2, 3)
    orig = NF.grid_sample(ff, warped, align_corners=True, mode='bilinear')
    newt = NF.grid_sample(features, coords, align_corners=True, mode='bilinear')[0].permute(1, 0, 2, 3)
    newo = orig.gather(0, view_w.permute(0, 3, 1, 2).repeat(1, 64, 1, 1))

    newt[mask] = 0
    newo[mask] = 0

    return newt
