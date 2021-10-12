import random
from os import path
from .base_data_module import BaseDataModule

class TanksAndTemples(BaseDataModule):
    def __init__(self, dataset_dir, num_views=5, batch_size=1, options={}):
        super(TanksAndTemples, self).__init__(dataset_dir / 'tnt', num_views=num_views, batch_size=batch_size, options=options)
        
    def get_train_sets(self):
        return [
            'Barn',
            'Caterpillar',
            'Church',
            'Courthouse',
            'Ignatius',
            'Meetingroom',
            'Truck'
        ]

    def get_val_sets(self):
        raise Exception('No validation set found for TnT Dataset')

    def get_test_sets(self):
        return [
            # advanced
            'Auditorium',
            'Ballroom',
            'Courtroom',
            'Museum',
            'Palace',
            'Temple',
            # intermediate
            'Family',
            'Francis',
            'Horse',
            'Lighthouse',
            'M60',
            'Panther',
            'Playground',
            'Train',
            # training
            'Ignatius',
            'Barn',
            'Meetingroom',
            'Truck',
            'Caterpillar',
            'Church',
            'Courthouse',
        ]
