import torch
from pytorch_lightning.metrics import Metric

class MeanAbsDist(Metric):
    def __init__(self, dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state('correct', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0.0), dist_reduce_fx='sum')

    def compute(self):
        return self.correct.float() / self.total.float()

    def update(self, x, y):
        v = (y != 0)
        dists = (x - y).abs()[v].sum()
        self.correct += dists
        self.total += v.sum()

class MeanAngDiff(Metric):
    def __init__(self, dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def compute(self):
        return self.correct.float() / self.total.float()

    def update(self, x, y):
        v = (y != 0).all(-1)
        diffs = (x * y).sum(-1).clamp(-1, 1).acos()[v].sum()
        self.correct += diffs
        self.total += v.sum()

class PercentInlier(Metric):
    def __init__(self, dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state('correct', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0.0), dist_reduce_fx='sum')

    def compute(self):
        return self.correct / self.total

    def update(self, x, y, theta):
        v = (y != 0).all(-1, keepdim=True)

        perc = (x - y).abs() < theta
        self.correct += perc[v].float().sum()
        self.total += v.sum()
