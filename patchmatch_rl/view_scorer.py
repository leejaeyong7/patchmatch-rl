import torch
import torch.nn as nn
import torch.nn.functional as NF
import math

from scipy.ndimage import gaussian_filter
import numpy as np

class CNNViewScorer(nn.Module):
    def __init__(self, hparams):
        super(CNNViewScorer, self).__init__()
        self.hparams = hparams
        fo = self.hparams.feat_scorer_output_channel

        cs = self.hparams.view_scorer_channel_scale
        pe = self.hparams.view_scorer_positional_encoding

        self.weighter = nn.Sequential(
            nn.Conv2d(fo + pe * 2 * 4, cs, 5, padding=2),
            nn.BatchNorm2d(cs),
            nn.ReLU(cs),
            nn.Conv2d(cs, cs, 3, padding=1),
            nn.BatchNorm2d(cs),
            nn.ReLU(cs),
            nn.Conv2d(cs, 1, 1),
            nn.Sigmoid()
        )

    def positional_encoding(self, priors, L=6):
        freqs = 1e-1 ** (2 * (torch.arange(L, device=priors.device) + 1 // 2) / 2)
        encoded = []
        for freq in freqs:
            encoded.append((freq * priors).sin())
            encoded.append((freq * priors).cos())
        encoded = torch.cat(encoded, dim=1)
        return encoded

    def forward(self, group_corr, priors):
        '''
        Given features, priors and valid region, output scores for each pixel
        '''
        pe = self.hparams.view_scorer_positional_encoding
        encoded = self.positional_encoding(priors[1:], pe)
        view_f = torch.cat((group_corr, encoded), 1)
        view_w = self.weighter(view_f)
        return view_w
