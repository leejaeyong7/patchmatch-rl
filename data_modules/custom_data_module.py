import random
from os import path
from .base_data_module import BaseDataModule
from pathlib import Path

class CustomDataModule(BaseDataModule):
    def __init__(self, dataset_dir, num_views=5, options={}):
        super(CustomDataModule, self).__init__(dataset_dir, num_views, options=options)

    def camera_folder_name(self):
        return 'cams'
