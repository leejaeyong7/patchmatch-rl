import torch
import torch.nn.functional as NF

def generate_pixel_grids(H, W):
    '''returns W x H grid pixels

    Given width and height, creates a mesh grid, and returns homogeneous 
    coordinates
    of image in a 3 x W*H Tensor

    Arguments:
        width {Number} -- Number representing width of pixel grid image
        height {Number} -- Number representing height of pixel grid image

    Returns:
        torch.Tensor -- 1x2xHxW, oriented in x, y order
    '''
    # from 0.5 to w-0.5
    x_coords = torch.linspace(0.5, W - 0.5, W)
    # from 0.5 to h-0.5
    y_coords = torch.linspace(0.5, H - 0.5, H)
    y_grid_coords, x_grid_coords = torch.meshgrid([y_coords, x_coords])
    # x_grid_coords = x_grid_coords.contiguous().view(-1)
    # y_grid_coords = y_grid_coords.contiguous().view(-1)
    return torch.stack([
        x_grid_coords,
        y_grid_coords
    ], 0).unsqueeze(0)

def generate_patches(P, H, W):
    '''

    Returns:
        tensor of shape : 1xPxPxHxWx3x1
    '''
    # Points = 1x2xHxW
    points = generate_pixel_grids(H, W)
    A = 2*P + 1

    # Homogenous Points = 1x3xHxW

    # generate patch-size neighbor pert
    # Px1 tensor
    pert = torch.arange(-P, P+1).float()

    # PxP tensor and stack
    perts = pert.view(-1, 1).repeat(1, A)

    # 2xPxP perturbations
    xy_perts = torch.stack((perts, perts.t()))

    # perturbed = 1x2xPxPxHxW
    perturbed = points.view(1, 2, 1, 1, H, W) + xy_perts.view(1, 2, A, A, 1, 1)

    # return 1xPxPxHxWx3x1
    return to_homogeneous(perturbed).permute(0, 2, 3, 4, 5, 1).unsqueeze(6)

def resize_depth(depth: torch.FloatTensor, size: tuple)->torch.FloatTensor:
    return NF.interpolate(depth, size, mode='nearest')

def compute_normal_from_depth(depth: torch.FloatTensor, K):
    '''
    dzdx=(z(x+1,y)-z(x-1,y))/2.0;
    dzdy=(z(x,y+1)-z(x,y-1))/2.0;
    direction=(-dzdx,-dzdy,1.0)
    magnitude=sqrt(direction.x**2 + direction.y**2 + direction.z**2)
    normal=direction/magnitude
    '''
    # dx = (x + 1) - (x)
    H, W = depth.shape[-2:]
    mask = depth != 0

    # 1x2xHxW pixels
    grid = generate_pixel_grids(H, W).type_as(depth)
    h_grid = to_homogeneous(grid).view(1, 3, -1)

    # obtain 3d points from depth + intrinsics
    p = ((K.inverse() @ h_grid) * depth.view(1, 1, -1))

    # estimate surface normal using finite difference
    p = p.view(1, 3, H, W)

    px = NF.pad(p, [1, 1, 0, 0])
    py = NF.pad(p, [0, 0, 1, 1])
    pxm = NF.pad(mask, [1, 1, 0, 0])
    pym = NF.pad(mask, [0, 0, 1, 1])

    mx = pxm[:, :, :, 2:] & pxm[:, :, :, 1:-1] & pxm[:, :, :, :-2] 
    my = pym[:, :, 2:] & pym[:, :, 1:-1:] & pym[:, :, :-2] 
    m = mx & my

    # 
    dxs = (px[:, :, :, 2:] - px[:, :, :, 1:-1]) + (px[:, :, :, 1:-1] - px[:, :, :, :-2]) 
    dys = (py[:, :, 2:] - py[:, :, 1:-1]) + (py[:, :, 1:-1] - py[:, :, :-2]) 

    dx = NF.normalize(dxs / 2, dim=1, p=2)
    dy = NF.normalize(dys / 2, dim=1, p=2)

    normal = torch.cross(dx, dy, dim=1)

    ones = torch.ones_like(normal)
    normal *= torch.where(normal[:, 2:] > 0, -ones, ones)
    normal = NF.normalize(normal, p=2, dim=1)

    nm = m.repeat(1, 3, 1, 1)
    normal[~nm] = 0
    return normal

def to_flat_map(P: torch.Tensor)->torch.Tensor:
    N, C, H, W = P.shape
    return P.view(N, C, H*W)

def to_map(P: torch.Tensor, shape:tuple)->torch.Tensor:
    N, C, HW = P.shape
    return P.view(N, C, *shape)

def to_homogeneous(P: torch.Tensor)->torch.Tensor:
    '''
    Converts points from non-homogeneous coordinate to homogeneous coordinate

    Arguments:
        - P: 

    Returns:
        - P: 
    '''
    if(len(P.shape) == 2):
        D, NP = P.shape
        hP = torch.ones((D+1, NP), device=P.device)
        hP[:D] = P
        return hP
    elif(len(P.shape) >= 3):
        B, D = P.shape[:2]
        R = P.shape[2:]
        hP = torch.ones((B, D+1, *R), device=P.device)
        hP[:, :D] = P
        return hP
    else:
        raise NotImplementedError

def from_homogeneous(hP):
    if(len(hP.shape) == 2):
        D, NP = hP.shape
        return hP[:D-1] / hP[D-1:]
    elif(len(hP.shape) == 3):
        B, D, NP = hP.shape
        return hP[:, :D-1] / hP[:, D-1:]
    elif(len(P.shape) == 4):
        B, D, H, W = P.shape
        return hP[:, :D-1] / hP[:, D-1:]
    else:
        raise NotImplementedError

def compute_priors(plane_map: torch.Tensor, 
                   intrinsics: torch.Tensor, 
                   extrinsics: torch.Tensor)->torch.Tensor:
    '''
    Given Plane Map of shape 4xHxW, obtain prior values

    The prior values should be:
      - Scale: contains ratio between ref to point vs src to point
      - Incident Angle: incident angle from ref to plane
      - Triangulation Angle: triangulation angle between ref to poin to src

    Arguments:
        plane_map: 4xHxW map in plane-normal form
        intrinsics: Nx3x3 intrinsic matrices
        extrinsics: Nx3x3 extrinsic matrices

    Returns:
        Nx3xHxW tensor containing scale, angle and triangulation angle
    '''
    _, H, W = plane_map.shape
    # 4xHW
    planes = plane_map.view(-1, H*W)
    normals = planes[:3]
    depths = depth_from_plane(plane_map.unsqueeze(0), intrinsics).view(1, -1)

    # intrinsic / inverse
    K = intrinsics
    Ki = K.inverse()

    # extrinsic / rotation / translation / inverse
    E = extrinsics
    R = E[:, :3, :3]
    t = E[:, :3, 3:]
    Ri = R.inverse()

    # center of camera = Nx3x1
    C = -Ri @ t

    # generate pixel grids of shape 1x2xHxW => 1x3xHW
    P = generate_pixel_grids(H, W).type_as(depths)
    HP = to_homogeneous(P).view(1, 3, -1)
    D = depths.unsqueeze(0)
    N = normals.unsqueeze(0)

    # Nx3x3 @ Nx3x3 @ (1x3xP @ 1x1xP - Nx3x1) => Nx3x3 @ Nx3xP => Nx3xP
    world_points = Ri @ ((Ki @ HP) * D - t)

    # compute vectors from world points to camera centers
    p_dir = world_points - C

    # compute distances and direction vectors
    # Nx3xP => Nx1xP
    dists = p_dir.norm(p=2, dim=1, keepdim=True)
    n_p_dir = p_dir / dists

    # compute scalar diff / angular diffs
    ref_d = dists[:1]
    scales = ref_d / dists
    tri_angles = (n_p_dir[:1] * n_p_dir).sum(dim=1, keepdim=True)
    inc_angles = (N * n_p_dir).sum(dim=1, keepdim=True)

    # return stacked tensors
    return torch.cat((scales, tri_angles, inc_angles), dim=1).view(-1, 3, H, W)

def to_camera_coord(K, H, W):
    p = to_homogeneous(generate_pixel_grids(H, W)).type_as(K).view(3, -1).t().view(1, -1, 3, 1)
    Ki = K.inverse().unsqueeze(1)
    return (Ki @ p).view(-1, H, W, 3).permute(0, 3, 1, 2)

def dist_from_normal_depth(N, D, K):
    '''
    D: Nx1xHxW
    N: Nx3xHxW
    '''
    H, W = D.shape[-2:]
    p = to_camera_coord(K[:1], H, W)
    # obtain distance from plane
    return (D * p * N).sum(1, keepdim=True)

def depth_from_plane(P, K):
    '''
    P: Nx4xHxW plane
    K: Nx3x3 intrisic matrix
    '''
    H, W = P.shape[-2:]
    N = P[:, :3]
    dist = P[:, 3:]
    cp = to_camera_coord(K[:1], H, W)
    return dist / (N * cp).sum(1, keepdim=True)

def depths_from_plane(P, K, R):
    '''
    P: Sx4xHxW plane
    K: Nx3x3 intrisic matrix
    '''
    H, W = P.shape[-2:]
    N = P[:, :3]
    dist = P[:, 3:]

    cp = to_camera_coord(K[:1], H, W)

    # 1xPxPxHxWx3x1
    Ps = generate_patches(R, H, W).type_as(P)
    cPs = K[:1].inverse().view(1, 1, 1, 1, 1, 3, 3) @ Ps
    Ns = N.permute(0, 2, 3, 1).view(-1, 1, 1, H, W, 3, 1)
    dot = (Ns * cPs).sum(-2, keepdim=True).clamp_min(1e-6)

    # NxPxPxHxWx1
    A = R * 2 + 1
    return (dist.view(-1, 1, 1, H, W, 1, 1) / dot).view(-1, A, A, H, W)

def warp_plane_map(plane_map: torch.Tensor, 
                   intrinsics: torch.Tensor, 
                   extrinsics: torch.Tensor,
                   P: int)->torch.Tensor:
    '''
    Given Plane Map of shape 4xHxW, obtain warping pixels from the reference images

    Arguments:
        plane_map: 4xHxW map in plane-normal form
        intrinsics: Nx3x3 intrinsic matrices
        extrinsics: Nx3x3 extrinsic matrices
        P: patch size
    Returns:
        NxPxPxHxWx2 coordinates representing sample coordinates for each patch
    '''
    _, H, W = plane_map.shape
    planes = plane_map.view(4, -1)
    N = intrinsics.shape[0]
    A = P * 2 + 1

    # define variables used in warping
    # intrinsic / inverse
    K = intrinsics.unsqueeze(1)
    Ki = K.inverse()

    # extrinsic / rotation / translation / inverse
    E = extrinsics.unsqueeze(1)
    R = E[:, :, :3, :3]
    t = E[:, :, :3, 3:]
    Ri = R.inverse()

    # convert plane into plane normal form
    # 3xHW => 1xHWx1x3
    normal = planes[:3]
    dists = planes[3:]

    # obtain distance from plane
    pnf_planes = (normal / -dists).t().view(1, H*W, 1, 3)

    # compute homography in reference camera space to source camera space
    # Nx1x3x3
    # Nx1x3x1
    rel_R = R @ Ri[:1]
    rel_t = (-rel_R @ t[:1]) + t

    # Nx1x3x1 @ 1xHWx1x3 => NxHWx3x3
    # Nx1x3x3 - NxHWx3x3 => NxHWx3x3
    camera_homography = rel_R - (rel_t @ pnf_planes)

    # Nx1x3x3 @ NxHWx3x3 @ Nx1x3x3 => NxHWx3x3
    homographies = K @ camera_homography @ Ki[:1]

    # Nx1x1xHWx3x3 @ 1xPxPxHxWx3x1 => NxPxPxHxWx3x1
    Hs = homographies.view(N, 1, 1, H, W, 3, 3)
    Ps = generate_patches(P, H, W).type_as(Hs)
    warped = Hs @ Ps

    # NxPxHWx3x1 => NxPxHWx2 => NxPxHxWx2
    norm_warped = warped[:, :, :, :, :, :2] / warped[:, :, :, :, :, 2:]
    # coordinates are oriented as:
    #  -0.5, -0.5
    #  -0.5, 0.5
    #  -0.5, 1.5 ...
    return norm_warped.view(N, A, A, H, W, 2)

