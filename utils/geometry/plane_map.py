import torch
import torch.nn.functional as NF

class PlaneMap:
    def __init__(self, shape, patch_size, device):
        self.P = patch_size
        self.H, self.W = shape
        self.device = device

        self.data = torch.zeros((1, self.H, self.W, 4), device=device)

    def initialize(self, range):
        return

    def propagate(self, propagator):
        return


    def resize(self, shape):
        OH, OW = self.H, self.W




        self.H, self.W = shape