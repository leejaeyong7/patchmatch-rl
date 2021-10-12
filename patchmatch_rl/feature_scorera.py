import torch
import torch.nn as nn

class GroupCorrFeatureScorer(nn.Module):
    def __init__(self, hparams):
        super(GroupCorrFeatureScorer, self).__init__()
        self.hparams = hparams
        self.G = self.hparams.feature_scorer_channel_scale
        self.hparams.feat_scorer_output_channel = self.output_channel()

    def output_channel(self):
        return self.G

    def forward(self, features):
        G = self.G
        N, C, H, W = features.shape
        feats = features.permute(0, 2, 3, 1)
        ref_f = feats[:1].view(1, H, W, G, 1, -1)
        src_f = feats[1:].view(N-1, H, W, G, -1, 1)
        gc = (ref_f @ src_f).view(-1, H, W, G).permute(0, 3, 1, 2)
        return gc