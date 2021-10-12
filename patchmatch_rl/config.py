''' Defines default options for the run. '''
import patchmatch_rl.propagators as Propagators

import math

default_options = {
    'learning_rate': {
        'type': float,
        'default': 1e-3
    },
    'optimizer': {
        'type': str,
        'choices': ['SGD', 'Ranger', 'Adam'],
        'default': 'Adam'
    },
    'propagator':{
        'type': str,
        'choices': Propagators.__all__,
        'default': 'JaePropagator'
    },
    'patch_size': {
        'type': int,
        'default': 1
    },
    'patch_dilation': {
        'type': int,
        'default': 3
    },
    'feature_extractor_channel_scale': {
        'type': int,
        'default': 8
    },
    'view_scorer_channel_scale': {
        'type': int,
        'default': 8
    },
    'view_scorer_positional_encoding': {
        'type': int,
        'default': 5
    },
    'feature_scorer_channel_scale': {
        'type': int,
        'default': 4
    },
    'regularizer_channel_scale': {
        'type': int,
        'default': 8
    },
    'num_hidden_states': {
        'type': int,
        'default': 3
    },
    'num_hidden_channels': {
        'type': int,
        'default': 8
    },
    'scheduler_rate': {
        'type': float,
        'default': 0.85
    },
    'scheduler_last_epoch': {
        'type': int,
        'default': 10
    },
    'num_views': {
        'type': int,
        'default': 10
    },
    'num_train_views': {
        'type': int,
        'default': 7
    },
    'num_view_selection': {
        'type': int,
        'default': 2
    },
    'num_train_view_selection': {
        'type': int,
        'default': 1
    },
    'num_train_view_unselection': {
        'type': int,
        'default': 2
    },
    'num_train_random_views':{
        'type': int,
        'default': 3
    },
    'd_sigma': {
        'type': float,
        'default': 0.01 # 1mm
    },
    'd_sigma_train': {
        'type': float,
        'default': 1.0 # 10mm
    },
    'n_sigma': {
        'type': float,
        'default': 3.0 * math.pi / 180.0
    },
    'n_sigma_train': {
        'type': float,
        'default': 3.0 * math.pi / 180.0
    },
    'skip_pairwise': {
        'type': bool,
        'default': False
    },
    'image_log_interval': {
        'type': int,
        'default': 500
    },
}
