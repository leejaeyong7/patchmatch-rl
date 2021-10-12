import sys
sys.path.append('.')
sys.path.append('..')

import math
import torch.nn.functional as NF    
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
import open3d as o3d
from pyntcloud import PyntCloud
from argparse import ArgumentParser, Namespace
from pathlib import Path
from utils.geometry import *
import data_modules as DataModules
from utils.geometry.camera import MVSCamera
from PIL import Image
import torchvision.transforms.functional as F
'''
Given output path of following format:
    output_path/ experiment_name / last.ckpt / maps /
        000000_image.pth => 3xHxW tensor
        000000_depth_map.pth => 1xHxW tensor
        000000_normal_map.pth => 3xHxW tensor
        000000_camera.pth => (1x3x3, 1x4x4) tuple

creates a point cloud by iterating for all N views, and checking for depth / normal consistency

The output would be a PLY file that is fused from all views and is saved in:
    output_path/ experiment_name / last.ckpt.ply
'''

def gather(cameras, projected, target):
    h_projected = from_vector(to_homogeneous(projected))
    warped = to_vector(cameras.normalize(h_projected))
    return to_bhwc(NF.grid_sample(to_bchw(target), warped, mode='nearest', align_corners=False))


def main(args):
    output_path = Path(args.output_path)
    experiment_name = args.experiment_name
    checkpoint_name = 'last.ckpt' if (args.epoch_id is None) else 'epoch={}.ckpt'.format(args.epoch_id)

    dev = torch.device(args.device)
    output_exp_path = output_path / experiment_name /  checkpoint_name 
    output_map_path = output_exp_path / args.dataset / 'maps'
    output_ply_path = output_exp_path / args.dataset / 'pointclouds'

    # obtain output experiment path
    if (not output_exp_path.exists()) or (not output_map_path.exists()):
        raise FileNotFoundError("Output path not found!")
    output_ply_path.mkdir(exist_ok=True, parents=True)

    # setup parameters
    num_con = args.num_consistent
    dist_scale = args.dist_th
    proj_th = args.proj_th
    angle_th = args.angle_th / 180.0 * math.pi

    image_filename = '{:06d}_image.jpg'
    depth_filename = '{:06d}_depth_map.pth'
    vis_filename = '{:06d}_vis_map.pth'
    fus_d_filename = '{:06d}_fused.pth'
    fus_n_filename = '{:06d}_fusen.jpg'
    fus_c_filename = '{:06d}_fusec.jpg'
    fus_m_filename = '{:06d}_fusem.pth'
    mask_filename = '{:06d}_mask.pth'
    normal_filename = '{:06d}_normal_map.pth'
    camera_filename = '{:06d}_camera.pth'
    meta_filename = '{:06d}_meta.pth'

    use_full = False
    if(args.target_width is not None) and (args.target_height is not None):
        use_full = True
        TW = args.target_width
        TH = args.target_height

    set_ids = [filepath.stem for filepath in list(output_map_path.iterdir()) if filepath.is_dir() ]

    # detect num images
    for set_id in tqdm(set_ids, desc="processing set"):
        # num_con =  set_id_num_cons[set_id] if set_id in set_id_num_cons else num_con
        output_set_path = output_map_path / set_id
        output_ply_file_path = output_ply_path / (set_id + '.ply')

        image_ids = [int(filepath.stem[:6]) for filepath in list(output_set_path.iterdir()) if filepath.suffix.endswith('.pth')]
        num_images = max(image_ids) + 1

        # initialize mask
        for ref_id in tqdm(range(num_images), desc="mask assignment", leave=False):
            ref_maskfile = output_set_path / mask_filename.format(ref_id)
            # if(ref_maskfile.exists()):
                # continue
            if(use_full):
                ref_mask = torch.zeros((1, TH, TW, 1), dtype=torch.bool, device=dev)
            else:
                ref_depth = torch.load(output_set_path / depth_filename.format(ref_id), map_location=dev)
                H, W= ref_depth.shape[1:3]
                ref_mask = torch.zeros((1, H, W, 1), dtype=torch.bool, device=dev)
            torch.save(ref_mask, ref_maskfile)

        # map visibility
        for ref_id in tqdm(range(num_images), desc="visibility mapping", leave=False):
            # load in ref maps / src maps
            ref_visfile = output_set_path / vis_filename.format(ref_id)
            ref_fus_d_file = output_set_path / fus_d_filename.format(ref_id)
            ref_fus_n_file = output_set_path / fus_n_filename.format(ref_id)
            ref_fus_c_file = output_set_path / fus_c_filename.format(ref_id)
            ref_fus_m_file = output_set_path / fus_m_filename.format(ref_id)
            ref_maskfile = output_set_path / mask_filename.format(ref_id)
            # if(ref_visfile.exists()):
                # continue
            if not ((output_set_path / depth_filename.format(ref_id)).exists()):
                continue

            with Image.open(output_set_path / image_filename.format(ref_id)) as img:
                ref_image = F.to_tensor(img).unsqueeze(0).to(dev)

            ref_depth = torch.load(output_set_path / depth_filename.format(ref_id), map_location=dev)
            ref_normal = torch.load(output_set_path / normal_filename.format(ref_id), map_location=dev)
            ref_meta = torch.load(output_set_path / meta_filename.format(ref_id), map_location=dev)
            ref_mask = torch.load(output_set_path / mask_filename.format(ref_id), map_location=dev)
            ref_K, ref_E = torch.load(output_set_path / camera_filename.format(ref_id), map_location=dev)
            H, W = ref_depth.shape[1:3]
            ranges = ref_depth.min(), ref_depth.max()
            ref_camera = MVSCamera(ref_K, ref_E, P, (H, W), ranges)
            if(use_full):
                ref_plane = normal_depth_to_plane(ref_camera, ref_normal, ref_depth)
                ref_plane = to_bhwc(NF.interpolate(to_bchw(ref_plane), size=(TH, TW), mode='nearest'))
                ref_camera.resize((TH, TW))
                ref_depth = plane_to_depth(ref_camera, ref_plane)
                ref_normal = ref_plane[..., :3]
            ref_image = NF.interpolate(ref_image, size=ref_depth.shape[1:3], mode='bilinear')
            ref_color = to_bhwc(ref_image)
            H, W = ref_depth.shape[1:3]
            ref_vis = torch.zeros((1, H, W, 1), dtype=torch.int, device=dev)

            ref_fused = torch.zeros((1, H, W, 1), dtype=torch.float, device=dev) 
            ref_fusen = torch.zeros((1, H, W, 3), dtype=torch.float, device=dev) 
            ref_fusec = torch.zeros((1, H, W, 3), dtype=torch.float, device=dev) 

            ref_ps = ref_camera.pixel_points()
            ref_wp = ref_camera.back_project(ref_depth)
            dist_th = ref_depth[..., 0] * dist_scale

            # for src_id in tqdm(range(ref_id + 1, num_images), leave=False):
            src_ids = ref_meta['src_ids']
            for src_id in tqdm(src_ids, leave=False):
                if(ref_id == src_id):
                    continue
                if not (output_set_path / depth_filename.format(src_id)).exists():
                    continue
                with Image.open(output_set_path / image_filename.format(src_id)) as img:
                    src_image = F.to_tensor(img).unsqueeze(0).to(dev)
                src_maskfile = output_set_path / mask_filename.format(src_id)
                src_depth = torch.load(output_set_path / depth_filename.format(src_id), map_location=dev)
                src_normal = torch.load(output_set_path / normal_filename.format(src_id), map_location=dev)
                src_mask = torch.load(src_maskfile, map_location=dev)
                src_K, src_E = torch.load(output_set_path / camera_filename.format(src_id), map_location=dev)
                H, W = src_depth.shape[1:3]
                src_camera = MVSCamera(src_K, src_E, P, src_depth.shape[1:3], ranges)
                Ks = torch.cat((ref_K, src_K))
                Es = torch.cat((ref_E, src_E))
                cameras = MVSCamera(Ks, Es, P, (H, W), ranges)
                if(use_full):
                    # # 1HxWx3
                    src_plane = normal_depth_to_plane(src_camera, src_normal, src_depth)
                    src_plane = to_bhwc(NF.interpolate(to_bchw(src_plane), size=(TH, TW), mode='nearest'))
                    src_camera.resize((TH, TW))
                    src_depth = plane_to_depth(src_camera, src_plane)
                    src_normal = src_plane[..., :3]
                src_image = NF.interpolate(src_image, size=src_depth.shape[1:3], mode='bilinear')
                src_color = to_bhwc(src_image)
                src_wp = src_camera.back_project(src_depth)
                src_n = to_vector(ref_camera.R @ src_camera.Rt @ from_vector(src_normal))

                # 1xHxWx2
                src__ref_p = src_camera.project(ref_wp)

                # use projected points to get depths, world normals
                src__ref_wp = gather(src_camera, src__ref_p, src_wp)
                src__ref_n = gather(src_camera, src__ref_p, src_n)
                src__ref_c = gather(src_camera, src__ref_p, src_color)

                ref__src__ref_p = ref_camera.project(src__ref_wp)
                ref__src__ref_cam_p = (ref_camera.R @ src__ref_wp.unsqueeze(-1) + ref_camera.t).squeeze(-1)
                src_d = ref__src__ref_cam_p[..., 2:]

                # create a visibility map
                point_dists = (src__ref_wp - ref_wp).norm(p=2, dim=-1)
                proj_dists = (ref__src__ref_p - ref_ps).norm(p=2, dim=-1)
                ang_dists = (ref_normal * src__ref_n).sum(-1).clamp(-1, 1).acos()

                # consistent = (point_dists < dist_th) & (angle_diffs < angle_th)
                # s = cameras.scale()
                # d_std = (0.5 * ref_depth ** 2) / s
                # dist_th = (d_std)[..., 0] * dist_scale
                # point_consistent = (point_dists < dist_th).view(ref_vis.shape)
                # proj_th = 0.75
                point_consistent = (point_dists < dist_th).view(ref_vis.shape)
                proj_consistent = (proj_dists < proj_th).view(ref_vis.shape)
                ang_consistent = (ang_dists < angle_th).view(ref_vis.shape)
                consistent = (proj_consistent & point_consistent & ang_consistent)
                consistent_f = consistent.float()

                cur_con = ref_vis
                total_con = cur_con + consistent_f
                total_cons = total_con.repeat(1, 1, 1, 3)

                rm_d = (ref_fused * cur_con.float() + src_d * consistent_f) / total_con
                rm_n = (ref_fusen * cur_con.float() + src__ref_n* consistent_f) / total_con
                rm_c = (ref_fusec * cur_con.float() + src__ref_c* consistent_f) / total_con
                rm_d[total_con == 0] = 0
                rm_n[total_cons == 0] = 0
                rm_c[total_cons == 0] = 0
                ref_fused = rm_d 
                ref_fusen = rm_n 
                ref_fusec = rm_c 

                ref_vis = total_con
                src_coords = src__ref_p.view(-1, 2)
                src_y = (src_coords[:, 1] + 0.5).long()
                src_x = (src_coords[:, 0] + 0.5).long()
                ref_c = consistent.view(-1)
                src_v = (src_y >= 0) & (src_y < H) & (src_x >= 0) & (src_x < W) & ref_c
                src_vy = src_y[src_v]
                src_vx = src_x[src_v]
                src_mask[0, src_vy, src_vx, 0] = True
                torch.save(src_mask, src_maskfile)

            # add ref
            ref_fused = (ref_fused * ref_vis.float() + ref_depth) / (ref_vis.float() + 1)
            ref_fusen = (ref_fusen * ref_vis.float() + ref_normal) / (ref_vis.float() + 1)
            ref_fusec = (ref_fusec * ref_vis.float() + ref_color) / (ref_vis.float() + 1)
            ref_fusem = ref_mask
            # ref_fused[ref_mask] = 0

            # write ref visibility map
            torch.save(ref_vis, ref_visfile)
            torch.save(ref_fused, ref_fus_d_file)
            torch.save(ref_fusem, ref_fus_m_file)
            F.to_pil_image(to_bchw(ref_fusec)[0]).save(ref_fus_c_file)
            F.to_pil_image(to_bchw(ref_fusen * 0.5 + 0.5)[0]).save(ref_fus_n_file)

        # iterate visibility map and write to output
        world_ps = []
        world_ns = []
        colors = []
        for i in tqdm(range(num_images),desc='Fusing', leave=False):
            if not (output_set_path / depth_filename.format(i)).exists():
                continue
            depth = torch.load(output_set_path / depth_filename.format(i), map_location=dev)
            vis = torch.load(output_set_path / vis_filename.format(i), map_location=dev)
            K, E = torch.load(output_set_path / camera_filename.format(i), map_location=dev)

            fusd = torch.load(output_set_path / fus_d_filename.format(i), map_location=dev)
            with Image.open(output_set_path / fus_n_filename.format(i)) as img:
                fusn = (to_bhwc(F.to_tensor(img)[None]).to(dev) - 0.5) * 2.0
            with Image.open(output_set_path / fus_c_filename.format(i)) as img:
                fusc = to_bhwc(F.to_tensor(img)[None]).to(dev)

            ranges = depth.min(), depth.max()
            P = 1
            camera = MVSCamera(K, E, P, depth.shape[1:3], ranges)
            if(use_full):
                camera.resize((TH, TW))
            depth = fusd
            normal = fusn
            color = fusc
            V = (vis.view(-1) >= num_con) & (depth > 0).view(-1)
            world_p = camera.back_project(depth).view(-1, 3)
            world_n = to_vector(camera.Rt @ from_vector(normal)).view(-1, 3)
            world_c = color.view(-1, 3)

            fil_world_p = world_p[V].view(-1, 3)
            fil_world_n = world_n[V].view(-1, 3)
            fil_color = world_c[V].view(-1, 3)

            world_ps.append(fil_world_p.cpu())
            world_ns.append(fil_world_n.cpu())
            colors.append(fil_color.cpu())

        all_p = torch.cat(world_ps).numpy()
        all_n = torch.cat(world_ns).numpy()
        all_c = (torch.cat(colors).numpy().astype(np.float32) * 255).astype(np.uint8)

        pointdata = pd.DataFrame({
            'x': all_p[:, 0],
            'y': all_p[:, 1],
            'z': all_p[:, 2],
            'nx': all_n[:, 0],
            'ny': all_n[:, 1],
            'nz': all_n[:, 2],
            'red': all_c[:, 0],
            'green': all_c[:, 1],
            'blue': all_c[:, 2],
        })
        pointcloud = PyntCloud(pointdata)
        pointcloud.to_file(str(output_ply_file_path))




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--output_path', type=str, default='./outputs')
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--dataset', type=str, choices=DataModules.__all__, required=True)
    parser.add_argument('--epoch_id', type=int, default=None)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--proj_th', type=float, default=2)
    parser.add_argument('--dist_th', type=float, default=0.01)
    parser.add_argument('--angle_th', type=float, default=180.0)
    parser.add_argument('--target_width', type=int, default=None)
    parser.add_argument('--target_height', type=int, default=None)
    parser.add_argument('--num_consistent', type=int, default=3)

    args = parser.parse_args()

    main(args)