import torch
import torch.nn.functional as NF
from .camera import MVSCamera
from .transforms import *
from .depth import compute_normal_from_depth, generate_random_depths, compute_finite_diff_gradients

def normal_depth_to_plane(camera, normal_maps, depth_maps):
    '''

    The point is on the line iff 
        N . p + d = 0
        (Na . cx + Nb. cy + Nc . cz)d + disp  = 0
    Hence, the disp is:
        disp = -(Na . cx + Nb. cy + Nc . cz)d 
    '''
    disp_maps = normal_depth_to_disp(camera, normal_maps, depth_maps)
    return torch.cat((normal_maps, disp_maps), dim=-1)


def normal_depth_to_disp(camera, normal_maps, depth_maps):
    '''

    The point is on the line iff 
        point on scene = (cx . d , cy . d , cz . d)
        cz . d = -1/nz (disp + nx . cx . d + ny . cy . d)
        -disp = (nx . cx + ny . cy + nz . cz) d
        (Na . cx + Nb. cy + Nc . cz)d + disp  = 0
    Hence, the disp is:
        disp = -(Na . cx + Nb. cy + Nc . cz)d 
    '''
    cam_r = camera.camera_rays()
    cam_p = cam_r * depth_maps
    return -(cam_p * normal_maps).sum(-1, keepdim=True)

def plane_to_depth(camera, plane_maps):
    '''
    The point is intersecting on the plane
        (Na . cx + Nb . cy + Nc . cz)d + disp = 0

    Hence, the depth is:
        d = -disp / (Na . cx + Nb . cy + Nc . cz)
    '''
    normal_maps = plane_maps[..., :3]
    disp_maps = plane_maps[..., 3:]
    cam_dir = camera.camera_rays()
    dot = (cam_dir * normal_maps).sum(-1, keepdim=True).clamp_max(-1e-5)
    depths = (-disp_maps / dot).clamp(*camera.ranges)
    return depths

def planes_to_depths(cameras, plane_maps, patch_radius=1, dilation=3):
    # 1xHxWx3xA
    patches = generate_patches(cameras, patch_radius, dilation)
    points = cameras.back_project_patches(patches, 1)
    h_points = to_homogeneous(points)
    # NxHxWx4x1 * 1xHxWx4xA => NxHxWxA
    return (from_vector(plane_maps).transpose(-1, -2) @ h_points)

def generate_random_normal_maps(camera, num_planes):
    S = num_planes
    H = camera.H
    W = camera.W
    dev = camera.K.device
    cam_r = camera.camera_rays()

    # generate random point in hemisphere by 3 normal distributions
    normal_map = torch.randn(S, H, W, 3, device=dev)
    normal_map = NF.normalize(normal_map, dim=-1)

    # in case where the camera ray is looking at the same direction, flip the normal value
    ones = torch.ones_like(normal_map)
    dot = (normal_map * cam_r).sum(-1, keepdim=True)
    normal_map *= -dot.sign()

    # in case where the camera ray is in orthogonal to normals, make the normal value 
    # as 0, 0, 1
    default_normal_map = torch.zeros_like(normal_map)
    default_normal_map[:, :, :, 2] = -1
    orthogonal = dot.abs() < 1e-2
    normal_map = torch.where(orthogonal, default_normal_map, normal_map)
    return normal_map

def resize_plane_maps(caemras, plane_maps, shape):
    # depth_map = plane_to_depth(camera, plane_maps)
    # normal_map = plane_maps[..., :3]
    plane_maps = NF.interpolate(plane_maps.permute(0, 3, 1, 2), size=shape, mode='nearest').permute(0, 2, 3, 1)
    return plane_maps

def generate_random_plane_maps(camera, num_planes, min_d, max_d):
    '''
    Generates Random plane maps given range of depths
    Arguments
        - camera: MVSCamera Module
        - num_planes: number of planes
        - min_d: minimum depth value
        - max_d: maximum depth value

    Returns
        - plane_map: NxHxWx4 tensor representing oriented points per pixel 
    '''
    normal_map = generate_random_normal_maps(camera, num_planes)

    # generate uniform random values
    depth_map = generate_random_depths(camera, num_planes, min_d, max_d)

    disp_map = normal_depth_to_disp(camera, normal_map, depth_map)
    plane_map = torch.cat((normal_map, disp_map), dim=-1)

    return plane_map

def depth_normal_consistency(cameras: MVSCamera, plane_map: torch.Tensor):
    depth_map = plane_to_depth(cameras, plane_map)
    normal_map = plane_map[..., :3]

    d_dz_du, d_dz_dv = compute_finite_diff_gradients(depth_map)
    d_dz = torch.cat((d_dz_du, d_dz_dv), -1)

    nx = normal_map[..., 0]
    ny = normal_map[..., 1]
    nz = normal_map[..., 2]
    rays = cameras.camera_rays()
    denom = (rays * normal_map).sum(-1)

    n_dz_du = (-nx / (nz * cameras.fx()) * depth_map[..., 0] / denom).unsqueeze(-1)
    n_dz_dv = (-ny / (nz * cameras.fy()) * depth_map[..., 0] / denom).unsqueeze(-1)
    n_dz = torch.cat((n_dz_du, n_dz_dv), -1)


    return NF.smooth_l1_loss((d_dz, n_dz), reduce='mean')

def warp_plane_maps(cameras: MVSCamera, plane_map: torch.Tensor, patch_radius: int, dilation=3, stride=1):
    # obtain homographies from planes
    h = cameras.get_homographies(plane_map)
    patches = generate_patches(cameras, patch_radius, dilation, stride)

    # NxHxWx3x3 @ NxHxWx3xA => NxHxWx3xA => NxHxWxAx2
    coords = cameras.normalize(h @ patches).transpose(-1, -2)
    coords[coords.isnan()] = -2
    return coords

def compute_ref_patches(cameras, ref_f, P, dilation=3, stride=1, mode='bilinear'):
    N, C, H, W = ref_f.shape
    ref_coords = generate_patches(cameras, P, dilation, stride)
    coords = cameras.normalize(ref_coords)
    _, H, W, _, S = coords.shape
    g_coords = coords.permute(0, 1, 2, 4, 3).reshape(1, H, -1, 2).repeat(N, 1, 1, 1)
    return NF.grid_sample(ref_f, g_coords, mode=mode, align_corners=True).view(N, -1, H, W, S)

def compute_ref_patches_by_select(cameras, ref_f, P, dilation=3, stride=1, mode='bilinear'):
    '''
    P = 1, D=3, S = 1 =>  0 3
    '''
    N, C, H, W = ref_f.shape
    PW = P * dilation * stride
    ref_f_p = NF.pad(ref_f, (PW, PW, PW, PW))
    ref_coords = (generate_patches(cameras, P, dilation, stride) - 0.5 + PW).long()[..., :2, :]
    ref_xc = ref_coords[..., 0, :]
    ref_yc = ref_coords[..., 1, :]
    ref_i = (ref_yc * (W + 2 * PW) + ref_xc).view(1, 1, -1).repeat(N, C, 1)
    vals = ref_f_p.view(N, C, -1).gather(2, ref_i).view(N, C, H, W, -1)
    return vals


def generate_patches(cameras, patch_radius, dilation=3, stride=1):
    '''

    Returns:
        tensor of shape : 1xPxPxHxWx3x1
    '''
    R = patch_radius
    P = 2*R + 1
    A = P ** 2
    # NxHxWx2x1
    img_p = from_vector(cameras.pixel_points()) * stride + stride // 2

    # 2x(P^2) perturbations
    pert = torch.arange(-R, R+1).float().type_as(cameras.K)
    perts = pert.view(-1, 1).repeat(1, P)
    xy_perts = torch.stack((perts.t(), perts), dim=0).view(1, 1, 1, 2, A) * dilation

    # perturbed = NxHxWx2xA => NxHxWx3xA
    return to_homogeneous((img_p + xy_perts) / stride)

def perturb_planes(n_level, d_level, cameras, plane_maps, ranges):
    # given current plane map, perform refinement
    min_d, max_d = ranges
    _, H, W, _ = plane_maps.shape
    orig_d = plane_to_depth(cameras, plane_maps)

    max_d = (orig_d * (1 + d_level)).clamp_max(max_d)
    min_d = (orig_d * (1 - d_level)).clamp_min(min_d)
    rand_pert = torch.rand((1, H, W, 1), device=plane_maps.device)
    pert_d = (rand_pert * (max_d - min_d) + min_d)

    orig_n = plane_maps[..., :3]
    curr_n = orig_n
    for i in range(4):
        perturbation = (torch.rand_like(curr_n) - 0.5) * n_level
        ref_n = NF.normalize(orig_n + perturbation, p=2, dim=-1)

        cam_r = cameras.camera_rays()
        dot = (cam_r * ref_n).sum(-1, keepdim=True)
        curr_n = torch.where(dot > -1e-2, curr_n, ref_n)
    pert_n = curr_n

    orig_pert_disp = normal_depth_to_disp(cameras, orig_n, pert_d)
    pert_orig_disp = normal_depth_to_disp(cameras, pert_n, orig_d)
    pert_pert_disp = normal_depth_to_disp(cameras, pert_n, pert_d)

    orig_pert = torch.cat((orig_n, orig_pert_disp), -1)
    pert_orig = torch.cat((pert_n, pert_orig_disp), -1)
    pert_pert = torch.cat((pert_n, pert_pert_disp), -1)

    return torch.cat((orig_pert, pert_orig, pert_pert))

def compute_samples_from_patch(cameras, plane_map, features, view_w=None, P=1, dilation=1, stride=1):
    '''
    Arguments:
        - cameras
    '''
    C = features.shape[1]
    dev = features.device
    # obtain homographies from planes
    h = cameras.get_homographies(plane_map)
    patches = generate_patches(cameras, P, dilation, stride)


    # NxHxWx3x3 @ NxHxWx3xA => NxHxWx3xA => NxHxWxAx2
    warped = cameras.normalize(h @ patches).transpose(-1, -2)[1:]
    warped[warped.isnan()] = -2
    N, H, W, S, _ = warped.shape

    if(view_w is None):
        view_p_w = torch.arange(N, device=dev).view(N, 1, 1, 1, 1).repeat(1, H, W, S, 1)
    else:
        view_p_w = view_w.unsqueeze(-1).repeat(1, 1, 1, S, 1)

    view_coords = warped.gather(0, view_p_w.repeat(1, 1, 1, 1, 2))
    batch_coords = (view_p_w / (N - 1) - 0.5) * 2
    coords = torch.cat((view_coords, batch_coords), dim=-1).unsqueeze(0).view(1, -1, H, W*S, 3).clamp(-2, 2)
    sampled = NF.grid_sample(features, coords, align_corners=True, mode='bilinear')[0].view(C, -1, H, W, S).permute(1, 0, 2, 3, 4)
    return sampled

def compute_priors(cameras, plane_maps):
    '''
    Given Plane Map of shape 4xHxW, obtain prior values

    The prior values should be:
      - Scale: contains ratio between ref to point vs src to point
      - Incident Angle: incident angle from ref to plane
      - Triangulation Angle: triangulation angle between ref to poin to src

    Arguments:
        cameras: Camera class object
        plane_maps: HxWx4 plane tensor

    Returns:
        Nx3xHxW tensor containing scale, angle and triangulation angle
    '''
    H, W = plane_maps.shape[1:3]
    normal_maps = plane_maps[..., :3]
    depth_maps = plane_to_depth(cameras, plane_maps)

    world_u = cameras.up()
    world_p = cameras.back_project(depth_maps)

    # obtain world points to camera vectors
    # NxHxWx3
    world_p_to_cam = world_p - to_vector(cameras.C)

    # NxHxWx3
    dists = to_vector(cameras.R @ from_vector(world_p) + cameras.t)[..., 2:]
    dirs =  NF.normalize(world_p_to_cam, dim=-1, p=2)

    scales = dists / dists[:1]
    tri_angles = (dirs * dirs[:1]).sum(-1, keepdim=True)
    inc_angles = (dirs * normal_maps).sum(-1, keepdim=True)
    up_diffs = (world_u * world_u[:1]).sum(-1).view(-1, 1, 1, 1).repeat(1, H, W, 1).clamp(-1, 1)

    # return stacked tensors
    priors = torch.cat((scales, tri_angles, inc_angles, up_diffs), dim=-1)
    return to_bchw(priors)