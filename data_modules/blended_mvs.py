import random
from os import path
from .base_data_module import BaseDataModule
from pathlib import Path

class BlendedMVS(BaseDataModule):
    def __init__(self, dataset_dir, num_views=5, options={}):
        super(BlendedMVS, self).__init__(dataset_dir / 'BlendedMVS', num_views, options=options)

    def get_train_sets(self):
        set_list_file = self.dataset_dir / 'BlendedMVS_training.txt'

        with open(set_list_file, 'r') as f:
            set_lines = f.readlines()
        return [set_line.strip() for set_line in set_lines]

    def get_val_sets(self):
        set_list_file = self.dataset_dir / 'validation_list.txt'
        with open(set_list_file, 'r') as f:
            set_lines = f.readlines()
        return [set_line.strip() for set_line in set_lines]

    def get_test_sets(self):
        raise Exception('No test sets found for Blended MVS Dataset')

    def depth_folder_name(self):
        return 'rendered_depth_maps'

    def image_folder_name(self):
        return 'blended_images'

    def camera_folder_name(self):
        return 'cams'

    def pair_file(self):
        return 'cams/pair.txt'
