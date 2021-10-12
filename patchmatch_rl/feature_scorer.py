import torch
import torch.nn as nn
import torch.nn.functional as NF
from utils.geometry.plane import warp_plane_maps, compute_ref_patches
from utils.geometry.sampling import sample_features
import math

class GroupCorrFeatureScorer(nn.Module):
    def __init__(self, hparams):
        super(GroupCorrFeatureScorer, self).__init__()
        self.hparams = hparams
        ics = self.hparams.feature_extractor_output_channel
        cs = self.hparams.feature_scorer_channel_scale

        projectors = {}
        for o, ic in ics.items():
            projectors[str(o)] = nn.Conv1d(ic, ic, 1)

        self.projectors = nn.ModuleDict(projectors)

    def output_channel(self):
        return self.hparams.feature_scorer_channel_scale

    def forward(self, o, f):
        '''
        Given features, priors and valid region, output scores for each pixel

        Args:
            NxCxHxWxS features
        '''
        N, C, H, W, S = f.shape
        ref_p = f[:1]
        src_p = f[1:]

        proj_p = self.projectors[str(o)](ref_p.view(1, -1, H*W*S)).view(1, C, H, W, S)
        # Nx1xHxWxS
        att = ((proj_p[..., S // 2].unsqueeze(-1) * proj_p).sum(1, keepdim=True) / math.sqrt(C)).softmax(-1)

        # perform channel-wise correlation
        G = self.hparams.feature_scorer_channel_scale
        ref_f = ref_p.permute(0, 2, 3, 4, 1).view(1, H, W, S, G, 1, -1)
        src_f = src_p.permute(0, 2, 3, 4, 1).view(N-1, H, W, S, G, -1, 1)
        gc = (ref_f @ src_f).view(N-1, H, W, S, G).permute(0, 4, 1, 2, 3)

        return (gc * att).sum(-1)

    def compute_att(self, o, cameras, features, P, DIL, S):
        N, C, H, W = features.shape
        ref_p = compute_ref_patches(cameras, features[:1], P, DIL, S)
        PS = ref_p.shape[-1]
        proj_p = self.projectors[str(o)](ref_p.view(1, -1, H*W*PS)).view(1, C, H, W, PS)

        # Nx1xHxWxS
        att = ((proj_p[..., PS // 2].unsqueeze(-1) * proj_p).sum(1, keepdim=True) / math.sqrt(C)).softmax(-1)

        return ref_p, att

    def compute_gc(self, cameras, att, ref_p, features, plane_map, P, DIL, S):
        G = self.hparams.feature_scorer_channel_scale
        w_coords = warp_plane_maps(cameras, plane_map, P, DIL, S)[1:]
        N, H, W, PS, _ = w_coords.shape
        # iterate for P
        src_fs = features[1:]
        # NxHxWxPSxC
        ref_pp = ref_p.permute(0, 2, 3, 4, 1).view(1, H, W, PS, G, 1, -1)
        att_p = att.permute(0, 2, 3, 4, 1)

        gcs = 0
        for p in range(PS):

            # NxHxWx1
            a = att_p[..., p, :]

            # NxHxWxGx1x-1
            ref_f = ref_pp[..., p, :, :, :]

            # NxHxWxC
            src_f = sample_features(src_fs, w_coords[..., p, :]).permute(0, 2, 3, 1).view(N, H, W, G, -1, 1)
            gc = (ref_f @ src_f).view(N, H, W, G)
            gcs += gc * a

        return gcs.permute(0, 3, 1, 2), w_coords

    def compute_wgc(self, cameras, view_i, view_w, att, ref_p, features, plane_map, P, DIL, S):
        G = self.hparams.feature_scorer_channel_scale
        w_coords = warp_plane_maps(cameras, plane_map, P, DIL, S)[1:]

        N, H, W, PS, _ = w_coords.shape
        view_in = view_i.unsqueeze(-1).repeat(1, 1, 1, PS, 1)
        view_wn = NF.normalize(view_w, p=1, dim=0)

        # KxHxWxPSx2
        view_coords = w_coords.gather(0, view_in.repeat(1, 1, 1, 1, 2))
        batch_coords = (view_in / (N - 1) - 0.5) * 2
        V = len(batch_coords)

        # KxHxWxPSx3 => 1xKxHxWxPSx3
        v_coords = torch.cat((view_coords, batch_coords), dim=-1).unsqueeze(0)


        # NxCxHxW => 1xCxNxHxW
        src_fs = features[1:].unsqueeze(0).transpose(1, 2)

        # NxHxWxPSxC
        ref_pp = ref_p.permute(0, 2, 3, 4, 1).view(1, H, W, PS, G, 1, -1)
        att_p = att.permute(0, 2, 3, 4, 1)

        gcs = 0

        # iterate for P
        for p in range(PS):

            # NxHxWx1
            a = att_p[..., p, :]

            # NxHxWxGx1x-1
            ref_f = ref_pp[..., p, :, :, :]

            # 1xCxKxHxW => KxHxWx1xC
            src_f = NF.grid_sample(src_fs, v_coords[..., p, :], align_corners=True, mode='bilinear').permute(2, 3, 4, 0, 1).view(V, H, W, G, -1, 1)
            gc = (ref_f @ src_f).view(V, H, W, G)
            gcs += (gc * a * view_wn).sum(0, keepdim=True)

        return gcs.permute(0, 3, 1, 2)
