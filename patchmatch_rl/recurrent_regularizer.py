import torch
import torch.nn as nn
import torch.nn.functional as NF
import math
from .gru import GRU

class RecurrentRegularizer(nn.Module):
    '''
    View Selector should be taking warped feature and output the pixel-wise 
    weights for each each view.

    Input:
        NxHxWxCxPxP

    Output:
        HxW weights for each view
    '''
    def __init__(self, hparams):
        super(RecurrentRegularizer, self).__init__()
        self.hparams = hparams
        # for unary cost
        fo = self.hparams.feat_scorer_output_channel

        # for binary cost
        NC = self.hparams.num_hidden_channels
        P = 1

        NB = self.hparams.num_hidden_states
        modules = []
        if(not self.hparams.skip_pairwise):
            raw_input_channel = fo * 1 + 4
        else:
            raw_input_channel = fo
        
        if(NB > 0):
            for i in range(NB):
                ks = 3 if i == 0 else 1
                ic = raw_input_channel if i == 0 else NC
                oc = NC if i == 0 else NC
                module = GRU(ic, oc, ks)
                modules.append(module)

            self.grus = nn.ModuleList(modules)
            self.mlp = nn.Conv2d(NC, 1, P, padding=P//2)
        else:
            self.mlp = nn.Conv2d(raw_input_channel, 1, P, padding=P//2)

    def forward(self, cost, belief_maps):
        out = cost
        NB = self.hparams.num_hidden_states
        if(NB > 0):
            outs = []
            for i, gru in enumerate(self.grus):
                out = gru(out, belief_maps[i].unsqueeze(0))
                outs.append(out)

            final_cost = self.mlp(out)
            updated_beliefs = torch.cat(outs)

            return final_cost, updated_beliefs
        else:
            final_cost = self.mlp(out)
            return final_cost, belief_maps