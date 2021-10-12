from utils.io import *
from utils.io import read_pfm, read_camera
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as NF
from torchvision.transforms.functional import to_tensor

class SampleLoader(Dataset):
    def __init__(self, samples, options):
        super(SampleLoader, self).__init__()
        self.samples = samples
        self.options = options

    def __len__(self) -> int:
        '''
        Returns number of samples in the dataset.
        '''
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple:
        '''
        Actually fetches dataset.
        '''
        sample = self.samples[index]
        set_name = sample['set_name']
        ref_id = sample['ref_id']
        src_ids = sample['src_ids']

        # load from paths
        images = [self.load_image(ip)for ip in sample['images']]
        cameras = [self.load_camera(cam) for cam in sample['cameras']]

        # convert to intrinsic / extrinsinc / ranges
        intrinsics = [camera[0] for camera in cameras]
        extrinsics = [camera[1] for camera in cameras]
        ref_ranges = cameras[0][2]
        ranges = self.get_depth_range(ref_ranges)

        # convert to numpy arrays
        np_intrinsics = np.stack(intrinsics)
        np_extrinsics = np.stack(extrinsics)

        # Nx3xHxW
        list_images = [to_tensor(image) for image in images]

        # Nx3x3
        if(self.options.get('resize')):
            size = self.options.get('resize')
        else:
            size = None

        torch_intrinsics = torch.from_numpy(np_intrinsics)
        if not (size is None):
            torch_intrinsics = self.resize_intrinsics(torch_intrinsics, list_images, *size)

        # Nx4x4
        torch_extrinsics = torch.from_numpy(np_extrinsics)

        if not (size is None):
            batch_images = torch.stack(self.resize_images(list_images, *size))
        else:
            batch_images = torch.stack(list_images)

        # if self.get_depth:
        # on training, additionally load in depths
        data = {
            'images': batch_images,
            'intrinsics': torch_intrinsics,
            'extrinsics': torch_extrinsics,
            'ranges': ranges
        }
        if ('depths' in sample) and (len(sample['depths']) > 0) and (not self.options.get('skip_depth')):
            depths = [self.load_depth(dp) for dp in sample['depths']]
            list_depths = [torch.from_numpy(depth).unsqueeze(0) for depth in depths]

            
            # resize depth give size
            # else if depth is not resized into 1/4
            if self.options.get('unscaled_depth'):
                data['depths']= torch.stack(self.resize_depths_by_scale(list_depths, 0.25))
            if not (size is None):
                H, W = size
                data['depths'] = torch.stack(self.resize_depths_fill(list_depths, H // 4, W // 4))
            else:
                data['depths'] = torch.stack(list_depths)
            d = data['depths']
            # data['ranges'] = [d[d>0].min() * 0.8, d.max() * 1.2]

        if(self.options.get('return_set_id')):
            data['set_name'] = set_name
            data['ref_id'] = ref_id
            data['src_ids'] = src_ids

        return data

    def get_depth_range(self, ranges):
        '''
        Arguments:
            depth_start(float): starting value of depth interpolation
            depth_interval(float): interval distance between depth planes
            num_intervals(int): number of planes to inerpolate

        Returns:
            torch.Tensor: Px4 plane equations in hessian normal form.
        '''
        if(len(ranges) == 2):
            # depth_start = 425.0
            # depth_end = 905.0
            if ranges[0] > ranges[1]:
                depth_start = 425.0
                depth_end = 905.0
            else:
                depth_start = ranges[0]
                depth_end = ranges[1]
            return [depth_start, depth_end]
        else:
            depth_start = ranges[0]
            depth_intv = ranges[1]
            depth_num = int(ranges[2])
            return [depth_start, depth_start + depth_intv * depth_num]

    def load_depth(self, depth_full_path:Path) -> np.ndarray:
        ''' Loads depths given full path '''
        return read_pfm(depth_full_path)

    def load_image(self, image_full_path:Path)-> Image:
        ''' Loads image given full path '''
        return Image.open(image_full_path)

    def load_camera(self, camera_full_path:Path)-> tuple:
        ''' Loads camera given full path to file '''
        k, e, r = read_camera(camera_full_path)
        if(self.options.get('scaled_camera')):
            k *= 4
            k[2, 2] = 1
        return k, e, r

    def resize_images(self, images, height, width):
        return [NF.interpolate(image.unsqueeze(0), size=(height, width), mode='bicubic')[0] for image in images]

    def resize_intrinsics(self, K, images, height, width):
        KC = K.clone()
        for i, NK in enumerate(KC):
            NK[0] *= float(width) / float(images[i].shape[-1])
            NK[1] *= float(height) / float(images[i].shape[-2])
        return KC

    def resize_depths_by_scale(self, depths, scale):
        return [NF.interpolate(depth.unsqueeze(0), scale_factor=scale, mode='nearest')[0] for depth in depths]

    def resize_depths_fill(self, depths, height, width):
        TH, TW = height, width
        resizeds = []
        for depth in depths:
            _, OH, OW = depth.shape
            kernel = (OH // TH, OW // TW)


            resized = NF.interpolate(NF.max_pool2d(depth.unsqueeze(0), kernel), (TH, TW), mode='nearest')
            resizeds.append(resized[0])
        return resizeds
