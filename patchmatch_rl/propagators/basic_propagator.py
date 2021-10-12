import torch
from .base_propagator import BasePropagator

class BasicPropagator(BasePropagator):
    def __init__(self, hparams=None):
        super(BasicPropagator, self).__init__(hparams)
        self.hparams = hparams

    def define_shape(self):
        return torch.FloatTensor([
            0, 1, 0,
            1, 1, 1,
            0, 1, 0,
        ]).view(3, 3)