import random
from .base_data_module import BaseDataModule
from pathlib import Path

class ETH3DHR_O(BaseDataModule):
    train_sets = [
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
    test_sets = [
        'botanical_garden',
        'boulders',
        'bridge',
        'door',
        'exhibition_hall',
        'lecture_room',
        'living_room',
        'lounge',
        'observatory',
        'old_computer',
        'statue',
        'terrace_2'
    ]
    def __init__(self, dataset_dir, num_views=5, options={}):
        super(ETH3DHR, self).__init__(dataset_dir / 'eth3d_hr', num_views, options={**options})

    def get_train_sets(self):
        return ETH3DHR.train_sets

    def get_val_sets(self):
        return ETH3DHR.val_sets

    def get_test_sets(self):
        return ETH3DHR.test_sets

    def get_image_path(self, image_folder: Path, image_id: int)-> Path:
        ''' returns full path to image file '''
        image_name = '{:08d}.jpg'.format(image_id)

        return image_folder / image_name

class ETH3DHR(BaseDataModule):
    val_sets = [
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
    test_sets = [
        'botanical_garden',
        'boulders',
        'bridge',
        'door',
        'exhibition_hall',
        'lecture_room',
        'living_room',
        'lounge',
        'observatory',
        'old_computer',
        'statue',
        'terrace_2'
    ]
    def __init__(self, dataset_dir, num_views=5, options={}):
        super(ETH3DHR, self).__init__(dataset_dir / 'eth3d_high_res_test_1920', num_views, options={**options})

    def get_train_sets(self):
        return []#ETH3DHR.train_sets

    def get_val_sets(self):
        return ETH3DHR.val_sets

    def get_test_sets(self):
        return ETH3DHR.test_sets

    def get_image_path(self, image_folder: Path, image_id: int)-> Path:
        ''' returns full path to image file '''
        image_name = '{:08d}.jpg'.format(image_id)

        return image_folder / image_name

    def camera_folder_name(self):
        return 'cams_1'

    def get_image_path(self, image_folder: Path, image_id: int)-> Path:
        ''' returns full path to image file '''
        image_name = '{:08d}.jpg'.format(image_id)

        return image_folder / image_name