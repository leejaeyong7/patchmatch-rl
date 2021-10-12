import random
from os import path
from pathlib import Path
from .base_data_module import BaseDataModule
import torch


class DTU(BaseDataModule):
    train_sets = [2, 6, 7, 8, 14, 16, 18, 19, 20, 22, 30, 31, 36, 39, 41, 42,
                  44, 45, 46, 47, 50, 51, 52, 53, 55, 57, 58, 60, 61, 63, 64, 
                  65, 68, 69, 70, 71, 72, 74, 76, 83, 84, 85, 87, 88, 89, 90, 
                  91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 
                  105, 107, 108, 109, 111, 112, 113, 115, 116, 119, 120, 121, 
                  122, 123, 124, 125, 126, 127, 128]
    val_sets = [3, 5, 17, 21, 28, 35, 37, 38, 40, 43, 56, 59, 66, 67, 82, 86,
                106, 117]
    test_sets = [1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49,
                 62, 75, 77, 110, 114, 118]

    def __init__(self, dataset_dir, num_views=10, options={}):
        super(DTU, self).__init__(dataset_dir,  num_views, options=options)

    def get_train_sets(self)->list:
        self.mode = 'train'
        return [str(i) for i in DTU.train_sets]
    def get_val_sets(self)->list:
        self.mode = 'val'
        return [str(i) for i in DTU.val_sets]
    def get_test_sets(self)->list:
        self.mode = 'test'
        return [str(i) for i in DTU.test_sets]

    def get_image_path(self, image_folder: Path, image_id: int) -> Path:
        ''' returns full path to image file '''
        if(self.mode == 'train'):
            light_id = random.randint(0, 6)
        else:
            light_id = 3

        image_name = '{:08d}_{}.png'.format(image_id, light_id)
        return image_folder / image_name

    def get_depth_range(self, ranges):
        '''
        Arguments:
            depth_start(float): starting value of depth interpolation
            depth_interval(float): interval distance between depth planes
            num_intervals(int): number of planes to inerpolate
        Returns:
            list containing min, max depth
        '''
        depth_start = 425.0
        depth_end = 905.0
        return [depth_start, depth_end]

class DTULRT(DTU):
    def __init__(self, dataset_dir, num_views=10, options={}):
        super(DTULR, self).__init__(dataset_dir / 'dtu_mvsnet',  num_views, {**options, **{'scaled_camera': True, 'scaled_depth': True}})

    def parse_camera(self, camera_file_path: str) -> tuple:
        '''
        Loads camera from path
        Return:
            Extrinsic: 4x4 numpy array containing R | t
            Intrinsic: 3x3 numpy array containing intrinic matrix
            Range: min, interval, num_interval, max
        '''
        K, E, tokens = super.parse_camera(camera_file_path)
        K *= 4
        K[2, 2] = 1
        return K, E, tokens

class DTULR(DTU):
    def __init__(self, dataset_dir, num_views=10, options={}):
        super(DTULR, self).__init__(dataset_dir / 'dtu_mvsnet',  num_views, {**options, **{'scaled_camera': True, 'scaled_depth': True}})

    def parse_camera(self, camera_file_path: str) -> tuple:
        '''
        Loads camera from path
        Return:
            Extrinsic: 4x4 numpy array containing R | t
            Intrinsic: 3x3 numpy array containing intrinic matrix
            Range: min, interval, num_interval, max
        '''
        K, E, tokens = super.parse_camera(camera_file_path)
        K *= 4
        K[2, 2] = 1
        return K, E, tokens

    def get_samples_from_sets(self, target_sets, num_views, use_random_samples=False, get_depths=True):
        # iterate over sets
        samples = []
        target_sets = ['1']
        for target_set in target_sets:
            set_dir  = self.dataset_dir / target_set

            image_dir = set_dir / self.image_folder_name()
            camera_dir = set_dir / self.camera_folder_name()
            depth_dir = set_dir / self.depth_folder_name()
            pair_file = set_dir / self.pair_file()

            # pairs = {
            #   image_id: [neighbor_ids] 
            # }
            pairs = self.parse_pairs(pair_file)
            all_image_ids = list(pairs.keys())
            ref_image_ids = [0] * 10000
            for ref_image_id in ref_image_ids:
                neighbor_list = pairs[ref_image_id]
                neigh_image_ids = [ref_image_id] + neighbor_list

                neigh_image_hash = {}
                for n_id in neigh_image_ids:
                    neigh_image_hash[n_id] = n_id

                remaining_hash = {}
                for image_id in all_image_ids:
                    if not (image_id in neigh_image_hash):
                        remaining_hash[image_id] = image_id
                remainig_ids = list(remaining_hash.keys())
                selected_image_ids = neigh_image_ids[:num_views]
                src_image_ids = selected_image_ids[1:]

                sample = {
                    'set_name': target_set,
                    'ref_id': ref_image_id,
                    'src_ids': src_image_ids,
                    'images': [],
                    'cameras': [],
                    'depths': []
                }

                for image_id in selected_image_ids:
                    image_path = self.get_image_path(image_dir, image_id)
                    camera_path = self.get_camera_path(camera_dir, image_id)

                    sample['images'].append(image_path)
                    sample['cameras'].append(camera_path)
                    if get_depths and not(self.options.get('skip_depth')):
                        depth_path = self.get_depth_path(depth_dir, image_id)
                        sample['depths'].append(depth_path)
                samples.append(sample)
        return samples



class DTUHR(DTU):
    def __init__(self, dataset_dir, num_views=10):
        super(DTUHR, self).__init__(dataset_dir / 'dtu_mine',  num_views)

class MixedDTU(DTU):
    def __init__(self, dataset_dir, num_views=5, options={}):
        super(MixedDTU, self).__init__(dataset_dir, num_views, options=options)
        self.default_option = options
        self.dtu_option = {**options, **{'scaled_camera': True, 'scaled_depth': True}}

    def get_train_sets(self):
        self.options = self.dtu_option
        l = super().get_train_sets()
        return ['dtu_mvsnet/' + d for d in l]

    def get_val_sets(self):
        eth_sets = [
            'courtyard',
            'delivery_area',
            'electro',
            'facade',
            'kicker',
            'meadow',
            'office',
            'pipes',
            'playground',
            'relief',
            'relief_2',
            'terrace',
            'terrains',
        ]
        self.mode = 'val'
        self.options = self.default_option
        return ['eth3d_lr_1920/' + set_path for set_path in eth_sets]

    def get_image_path(self, image_folder: Path, image_id: int) -> Path:
        ''' returns full path to image file '''
        if(self.mode == 'train'):
            light_id = random.randint(0, 6)
            image_name = '{:08d}_{}.png'.format(image_id, light_id)
            return image_folder / image_name
        else:
            image_name = '{:08d}.jpg'.format(image_id)
            return image_folder / image_name


class DTUHR(DTU):
    def __init__(self, dataset_dir, num_views=10):
        super(DTUHR, self).__init__(dataset_dir / 'dtu_mine',  num_views)

class MixedDTU(DTU):
    def __init__(self, dataset_dir, num_views=5, options={}):
        super(MixedDTU, self).__init__(dataset_dir, num_views, options=options)
        self.default_option = options
        self.dtu_option = {**options, **{'scaled_camera': True, 'scaled_depth': True}}

    def get_train_sets(self):
        self.options = self.dtu_option
        l = super().get_train_sets()
        return ['dtu_mvsnet/' + d for d in l]

    def get_val_sets(self):
        eth_sets = [
            'courtyard',
            'delivery_area',
            'electro',
            'facade',
            'kicker',
            'meadow',
            'office',
            'pipes',
            'playground',
            'relief',
            'relief_2',
            'terrace',
            'terrains',
        ]
        self.mode = 'val'
        self.options = self.default_option
        return ['eth3d_lr_1920/' + set_path for set_path in eth_sets]

    def get_image_path(self, image_folder: Path, image_id: int) -> Path:
        ''' returns full path to image file '''
        if(self.mode == 'train'):
            light_id = random.randint(0, 6)
            image_name = '{:08d}_{}.png'.format(image_id, light_id)
            return image_folder / image_name
        else:
            image_name = '{:08d}.jpg'.format(image_id)
            return image_folder / image_name
