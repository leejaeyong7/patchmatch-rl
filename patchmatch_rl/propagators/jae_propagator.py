import torch
from .base_propagator import BasePropagator

class JaePropagator(BasePropagator):
    def __init__(self, hparams=None):
        super(JaePropagator, self).__init__(hparams)
        self.hparams = hparams

    def define_shape(self):
        return torch.FloatTensor([
            0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
            1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1,
            0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0
        ]).view(11, 11)
