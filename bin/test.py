import sys
sys.path.append('.')
sys.path.append('..')

from argparse import ArgumentParser, Namespace
from pathlib import Path

# setup locaml
from patch_match_mvs_net import PatchMatchMVSNet
import data_modules as DataModules

import torch
torch.backends.cudnn.enabled=True
torch.backends.cudnn.benchmark = False
from tqdm import tqdm
from utils.geometry.camera import MVSCamera
from utils.geometry import compute_normal_from_depth, to_bhwc
from utils.geometry.plane import plane_to_depth
from utils.refinement import refine_offline
import torchvision.transforms.functional as F
import torch.nn.functional as NF
import matplotlib.pyplot as plt


def main(args):
    # setup checkpoint loading
    checkpoint_path = Path(args.checkpoint_path)
    output_path = Path(args.output_path)
    experiment_name = args.experiment_name
    checkpoint_name = 'last.ckpt' if (args.epoch_id is None) else 'epoch={}.ckpt'.format(args.epoch_id)
    checkpoint_file = checkpoint_path / experiment_name / checkpoint_name
    dev = torch.device(args.device)
    if not(checkpoint_file.exists()):
        raise FileNotFoundError("CheckPoint Not Found!")

    data_path = Path(args.dataset_path)

    # setup model / trainer
    model = PatchMatchMVSNet.load_from_checkpoint(str(checkpoint_file))
    model = model.to(dev).eval()
    model.eval()

    # setup data module
    if(args.width != None) and (args.height != None):
        size = (args.height, args.width)
    else:
        size = None
    
    data_module = getattr(DataModules, args.dataset)(data_path, num_views=args.num_views, options={'return_set_id': True, 'resize': size})
    data_module.prepare_data()
    data_module.setup('test')

    # iterate test data and write depth map

    output_exp_path = output_path / experiment_name /  checkpoint_name / args.dataset / 'maps'
    output_exp_path.mkdir(parents=True, exist_ok=True)

    for _, batch in tqdm(enumerate(data_module.test_dataloader()), total=len(data_module.test_dataset)):
        set_name = batch['set_name']
        i = batch['ref_id']
        srcs = batch['src_ids']

        # create set dirs for each output sets
        output_set_path = output_exp_path / set_name
        output_set_path.mkdir(parents=True, exist_ok=True)

        image_filename = '{:06d}_image.jpg'.format(i)
        depth_filename = '{:06d}_depth_map.pth'.format(i)
        # raw_depth_filename = '{:06d}_raw_depth_map.pth'.format(i)
        normal_filename = '{:06d}_normal_map.pth'.format(i)
        conf_filename = '{:06d}_conf_map.pth'.format(i)
        camera_filename = '{:06d}_camera.pth'.format(i)
        meta_filename = '{:06d}_meta.pth'.format(i)

        image_file = output_set_path / image_filename
        depth_file = output_set_path / depth_filename
        # raw_depth_file = output_set_path / raw_depth_filename
        normal_file = output_set_path / normal_filename
        camera_file = output_set_path / camera_filename
        conf_file = output_set_path / conf_filename
        meta_file = output_set_path / meta_filename
        # if(meta_file.exists()):
        #     continue

        # setup data
        hparams = model.hparams
        images = batch['images'].to(dev)
        K = batch['intrinsics'].to(dev)
        E = batch['extrinsics'].to(dev)

        P = hparams['patch_size']
        DIL = 4#hparams['patch_dilation']
        NV = hparams['num_view_selection']
        P = hparams['patch_size'] 

        ranges = batch['ranges']
        cameras = MVSCamera(K, E, P, images.shape[-2:], ranges)

        torch.cuda.empty_cache()
        with torch.no_grad():
            inf_p, inf_c = model(images, K, E, P, DIL, ranges, num_iterations=8, num_refine=2, num_views=NV, soft=False, details=False, train=False)
            out_size = inf_p.shape[1:3]
            # resize intrinsics / images
            cameras.resize(out_size)
            inf_n = inf_p[..., :3]
            inf_d = plane_to_depth(cameras, inf_p)

        resized = NF.interpolate(images[:1], size=out_size)

        # clear cache
        torch.cuda.empty_cache()

        metadata = {
            'ref_id': i,
            'src_ids': srcs,
            'ranges': ranges
        }

        F.to_pil_image(images[0]).save(image_file)
        torch.save(inf_d, depth_file)
        torch.save(inf_n, normal_file)
        torch.save(inf_c, conf_file)
        torch.save((cameras.K[:1].view(1, 3, 3), E[:1]), camera_file)
        torch.save(metadata, meta_file)

        torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = ArgumentParser()

    # add PROGRAM level args
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints')
    parser.add_argument('--dataset', type=str, choices=DataModules.__all__, required=True)
    parser.add_argument('--set_id', type=str, required=True)
    parser.add_argument('--num_views', type=int, default=10)
    parser.add_argument('--epoch_id', type=int, default=None)
    parser.add_argument('--dataset_path', type=str, default='./data')
    parser.add_argument('--output_path', type=str, default='./outputs')
    parser.add_argument('--width', type=int, default=None)
    parser.add_argument('--height', type=int, default=None)
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()

    main(args)
