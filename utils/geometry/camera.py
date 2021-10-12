import math
import torch
import torch.nn.functional as NF
from .transforms import *
class MVSCamera:
    def __init__(self, K, E, P, shape, ranges):
        '''
        Creates commonly used variables from intrinsics, extrinsics

        Args:
            K: Bx3x3 intrinsic matrix
            E: Bx4x4 extrinsic matrix
            shape: (H, W) image size
        '''
        self.H, self.W = shape
        self.P = P
        self.K = K.view(-1, 1, 1, 3, 3)
        self.E = E.view(-1, 1, 1, 4, 4)
        self.R = self.E[:, :, :, :3, :3]
        self.t = self.E[:, :, :, :3, 3:]
        self.Ki = self.inverse_intrinsic(self.K)
        self.Rt = self.R.transpose(-1, -2)
        self.C = -self.Rt @ self.t
        self.grid_K = self.generate_grid_intrinsics()
        self.ranges = ranges

    def fx(self):
        return self.K[0, 0, 0, 0, 0]

    def fy(self):
        return self.K[0, 0, 0, 1, 1]

    def cx(self):
        return self.K[0, 0, 0, 0, -1]

    def cy(self):
        return self.K[0, 0, 0, 1, -1]

    def up(self):
        return NF.normalize(self.Rt[..., 1], p=2, dim=-1).view(-1, 3)

    def scale(self):
        fx = self.fx()
        fy = self.fx()
        f = (fx + fy) / 2.
        cs = self.C[:, 0, 0, :, 0]
        b = (cs[:1] - cs[1:]).norm(dim=-1, p=2)
        return f * b.mean() / math.sqrt(2)


    def inverse_intrinsic(self, K):
        Ki = K.clone()
        fx = K[..., 0:1, 0:1]
        fy = K[..., 1:2, 1:2]
        Ki[..., 0:1, :] /= fx
        Ki[..., 1:2, :] /= fy
        Ki[..., 0:1, 0:1] /= fx
        Ki[..., 1:2, 1:2] /= fy
        Ki[..., :2, -1] *= -1
        return Ki


    def generate_grid_intrinsics(self):
        dev = self.K.device
        grid_K = torch.zeros(2, 3, device=dev)
        grid_K[0, 0] = 2 / float(self.W - 1)
        grid_K[0, 2] = -float(self.W) / float(self.W - 1)
        grid_K[1, 1] = 2 / float(self.H - 1)
        grid_K[1, 2] = -float(self.H) / float(self.H - 1)
        return grid_K.view(1, 1, 1, 2, 3)

    def resize(self, new_shape):
        NH, NW = new_shape
        self.K = self.K.clone()
        self.K[:, :, :, 0] *= NW / float(self.W)
        self.K[:, :, :, 1] *= NH / float(self.H)
        self.Ki = self.inverse_intrinsic(self.K)
        self.H, self.W = new_shape
        self.grid_K = self.generate_grid_intrinsics()

    def resize_by_scale(self, scale: float):
        # clone buffer to ensuer inplace operation
        self.K = self.K.clone()
        self.K[:, :, :, 0] *= scale
        self.K[:, :, :, 1] *= scale
        self.Ki = self.inverse_intrinsic(self.K)
        self.H = int(scale * self.H)
        self.W = int(scale * self.W)
        self.grid_K = self.generate_grid_intrinsics()

    ###############################
    # inverse projection related
    # all projection related takes
    # NxHxWx... format shapes and outputs NxHxWx... format shapes
    def pixel_points(self, offset=0.5):
        '''
        Given width and height, creates a mesh grid, and returns homogeneous 
        coordinates
        of image in a 3 x W*H Tensor

        Arguments:
            width {Number} -- Number representing width of pixel grid image
            height {Number} -- Number representing height of pixel grid image

        Returns:
            torch.Tensor -- 1x2xHxW, oriented in x, y order
        '''
        dev = self.K.device
        W = self.W
        H = self.H
        O = offset
        x_coords = torch.linspace(O, W - 1 + O, W, device=dev)
        y_coords = torch.linspace(O, H - 1 + O, H, device=dev)

        # HxW grids
        y_grid_coords, x_grid_coords = torch.meshgrid([y_coords, x_coords])

        # HxWx2 grids => 1xHxWx2 grids
        return torch.stack([ x_grid_coords, y_grid_coords ], 2).unsqueeze(0)

    def camera_rays(self, num_views=1):
        x = from_vector(to_homogeneous(self.pixel_points()))
        Ki = self.Ki[:num_views]
        return to_vector(Ki @ x)

    def back_project(self, depth_maps, num_views=1):
        '''
        Given depth map, back project its depth to obtain world coordinates

        Args:
            depth_map: NxHxWx1 depths
        Returns:
            NxHxWx3 points in world coordinates
        '''
        Rt = self.Rt[:num_views]
        t = self.t[:num_views]

        # 1xHxWx3 * NxHxWx1
        r = self.camera_rays(1)
        p = from_vector(r * depth_maps)

        return to_vector(Rt @ (p - t))

    def to_world_normals(self, normal_maps, num_views=1):
        '''
        Given depth map, back project its depth to obtain world coordinates

        Args:
            depth_map: NxHxWx1 depths
        Returns:
            NxHxWx3 points in world coordinates
        '''
        Rt = self.Rt[:num_views]
        t = self.t[:num_views]
        return to_vector(Rt @ from_vector(normal_maps))

    def back_project_patches(self, patches, num_views=1):
        '''
        Given depth map, back project its depth to obtain world coordinates

        Args:
            patches: NxHxWx3xA patches
        Returns:
            NxHxWx3xA points in world coordinates
        '''
        Rt = self.Rt[:num_views]
        t = self.t[:num_views]
        Ki = self.Ki[:num_views]
        # NxHxWx3xA => 3xA => 3xA => 3xA
        return Rt @ ((Ki @ patches) - t)

    def project(self, world_p, num_views=1):
        p = from_vector(world_p)
        K = self.K[:num_views]
        R = self.R[:num_views]
        t = self.t[:num_views]

        return to_vector(from_homogeneous(K @ (R @ p + t)))


    def get_homographies(self, plane_maps):
        '''
        Given per-pixel planes in reference camera, obtain homographies for each pixel
        '''
        K = self.K
        Ki = self.Ki
        R = self.R
        Rt = self.Rt
        t = self.t

        normal = from_vector(plane_maps[:, :, :, :3])
        dists = from_vector(plane_maps[:, :, :, 3:])

        rel_R = R @ Rt[:1]
        rel_t = (-rel_R @ t[:1]) + t

        # obtain distance from plane
        pnf_planes = (normal / dists).transpose(3, 4)
        cam_h = rel_R - (rel_t @ pnf_planes)

        # NxHxWx3x3
        return K @ cam_h @ Ki[:1]

    def normalize(self, points):
        # NxHxWx3x1
        projected = to_homogeneous(from_homogeneous(points))
        return self.grid_K @ projected