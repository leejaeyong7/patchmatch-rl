# library imports
import torch
import torch.nn as nn
import torch.nn.functional as NF
import torch.optim as optim
import torchvision.transforms.functional as F
import numpy as np
import pytorch_lightning as pl
from argparse import Namespace, ArgumentParser

# utility functions
from utils.visualization import *
# from utils.warping import *
from utils.metrics import MeanAbsDist, MeanAngDiff, PercentInlier
from utils.geometry.camera import MVSCamera
from utils.geometry.plane import generate_random_plane_maps, resize_plane_maps, compute_priors, warp_plane_maps, perturb_planes, compute_samples_from_patch, generate_patches, compute_ref_patches, plane_to_depth, normal_depth_to_disp, normal_depth_to_plane, depth_normal_consistency
from utils.geometry.depth import resize_depth, compute_normal_from_depth, generate_random_depths, perturb, get_gt_depth, compute_samples
# from utils.geometry.priors import compute_priors
from utils.geometry.sampling import sample_features, sample_patches
from utils.geometry.transforms import *
from utils.geometry.view_selection import compute_visibility_map

# submodule imports
import patchmatch_rl.propagators as Propagators
from .feature_extractor import FPNFeatureExtractor as FeatureExtractor
from .feature_scorer import GroupCorrFeatureScorer as FeatureScorer
from .view_scorer import CNNViewScorer as ViewScorer
from .recurrent_regularizer import RecurrentRegularizer
from .config import default_options

# for debugging purpose; remove later
import matplotlib.pyplot as plt
import math

class PatchMatchRL(pl.LightningModule):
    def __init__(self, 
                 conf: Namespace,
                 *args, **kwargs
    ):
        super(PatchMatchRL, self).__init__()
        self.save_hyperparameters(conf)

        # submodules
        self.feature_extractor = FeatureExtractor(self.hparams)
        self.feat_scorer = FeatureScorer(self.hparams)
        self.hparams.feat_scorer_output_channel = self.feat_scorer.output_channel()
        self.propagator = getattr(Propagators, self.hparams.propagator)(self.hparams)
        self.view_scorer = ViewScorer(self.hparams)
        self.depth_regularizer = RecurrentRegularizer(self.hparams)
        self.neigh_propagator = Propagators.NeighPropagator(self.hparams)

        # metrics
        self.mean_abs_dist = MeanAbsDist('Mean Absolute Distance')
        self.mean_ang_diff = MeanAngDiff('Mean Angle Diff')
        self.percent_inlier = PercentInlier('Percent Inlier')
                
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        for name, args in default_options.items():
            if(args['type'] == bool):
                parser.add_argument('--{}'.format(name), type=eval, choices=[True, False], default=str(args.get('default')))
            else:
                parser.add_argument('--{}'.format(name), **args)
        return parser

    def __repr__(self):
        return repr(self.hparams)

    # -- meta learning setups -- 
    def configure_optimizers(self):
        if(self.hparams.optimizer == 'SGD'):
            opt = optim.SGD
        elif(self.hparams.optimizer == 'Adam'):
            opt = optim.Adam
        else:
            raise NotImplementedError
        params = self.parameters()

        optimizer = opt(params, lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                              step_size=1,
                                              gamma=self.hparams.scheduler_rate, 
                                              verbose=True)

        return [optimizer], [scheduler]

    def select_views(self, octave, cameras, features, plane_map, P, DIL, K, soft, details=False):
        '''
        Arguments:
            - octave: octave of the feature map
            - cameras: MVSCamera object
            - features: NxCxHxW tensor
            - plane_map: 1xHxWx4 tensor
            - P: Patch radius
            - DIL: Patch dilation
            - K: Number of views to select
            - soft: True if we want 'soft' sampling.
        Returns:
            - view_w: KxHxWx1 view selection weights
            - view_p: KxHxWx1 view selection probs
        '''
        N, C, H, W= features.shape

        priors = compute_priors(cameras, plane_map)

        # extract feat features from cnn feature maps
        ref_p, ref_att = self.feat_scorer.compute_att(octave, cameras, features, P, DIL, 1)
        gc, feat_warped = self.feat_scorer.compute_gc(cameras, ref_att, ref_p, features, plane_map, P, DIL, 1)

        # Nx1xHxW
        inf_valids = ((feat_warped <= 1) & (feat_warped >= -1)).all(-1).unsqueeze(1)[..., 0]

        view_s = self.view_scorer(gc, priors)

        valid_view_s = (view_s * inf_valids).permute(0, 2, 3, 1)
        view_s = view_s.permute(0, 2, 3, 1)

        view_ps, view_i = self.sample(valid_view_s, soft=soft, softmax=False, K=K, rand_sample=True)
        view_p = view_ps.gather(0, view_i).sum(0, keepdims=True)
        view_w = valid_view_s.gather(0, view_i)

        if not details:
            return {
                'view_indices': view_i,
                'view_weights': view_w
            }
        else:
            J = self.hparams.num_train_view_unselection
            N, H, W, _ = view_s.shape
            neg_i = torch.topk(-view_s.view(N, H*W).t(), k=J, dim=-1).indices.t().view(J, H, W, 1)
            view_pos_s = view_s.gather(0, view_i).clamp(1e-5, 1-1e-5)
            view_neg_s = view_s.gather(0, neg_i).clamp(1e-5, 1-1e-5)
            view_posneg_p = NF.normalize(torch.cat((view_pos_s, view_neg_s)), p=1, dim=0).clamp(1e-5, 1-1e-5)
            view_nll = -view_posneg_p[:K].sum(0, keepdim=True).log()
            return {
                'view_indices': view_i,
                'view_weights': view_w,
                'view_scores': view_s,
                'valid_view_scores': valid_view_s,
                'view_probs': view_ps,
                'view_prob': view_p,
                'view_nll': view_nll,
                'view_valids': inf_valids.permute(0, 2, 3, 1)
            }

    def sample(self, scores, K=1, soft=False, softmax=True, rand_sample=False):
        '''
        Given 
        '''
        N, H, W, _ = scores.shape
        K = min(K, N)

        if(softmax):
            probs = scores.softmax(0).clamp(1e-5, 1 - 1e-5)
        else:
            probs = NF.normalize(scores, p=1, dim=0).clamp(1e-5, 1-1e-5)

        # select view based on scores
        if(soft):
            if(rand_sample):
                index = torch.multinomial(probs.view(N, H*W).t(), K).t().view(K, H, W, 1)
            else:
                dev = scores.device
                top_index = torch.topk(scores.view(N, H*W).t(), k=K, dim=-1).indices.t().view(K, H, W, 1)
                rand_index = torch.multinomial(probs.view(N, H*W).t(), K).t().view(K, H, W, 1)
                # rand_index = torch.randint(0, N, (K, H, W, 1), device=dev)
                rand_sel = torch.rand_like(top_index.float())
                rand_th = self.random_eps()
                index = torch.where(rand_sel < rand_th, rand_index, top_index)
        else:
            index = torch.topk(scores.view(N, H*W).t(), k=K, dim=-1).indices.t().view(K, H, W, 1)

        return probs, index

    def random_eps(self):
        steps = self.global_step
        base_eps = 0.9
        decay = 0.999
        min_epsilon = 0.01
        eps = max(min_epsilon, base_eps * (decay ** steps))
        return eps

    def lambda_depth(self):
        steps = self.global_step
        # return 0.001
        base_eps = 1
        decay = 0.999
        min_epsilon = 0.001
        eps = max(min_epsilon, base_eps * (decay ** steps))
        return eps


    def propagate(self, octave, cameras, plane_map, train=False):
        # warp from 1xHxWx4 => SxHxWx4 values
        prop_p = self.propagator(plane_map)

        # check to see if planes are all good
        if(octave == 0):
            planes = prop_p
        else:
            # extract out depth from propagated planes
            pert_p = self.perturb(cameras, plane_map)
            planes = torch.cat((prop_p, pert_p), 0)
        r = cameras.camera_rays()
        nr = NF.normalize(r, p=2, dim=-1)
        plane_n = planes[..., :3]
        plane_disp = planes[..., 3:]
        plane_d = (-plane_disp / (plane_n * r).sum(-1, keepdim=True)).clamp(*cameras.ranges)

        dots = (plane_n * nr).sum(-1, keepdim=True)
        curr_n = plane_map[..., :3]
        curr_d = plane_to_depth(cameras, plane_map)
        ref_n = torch.where((dots.abs() < 1e-2) | (dots >= 0), curr_n, plane_n)
        ref_d = torch.where(plane_d.isnan() | plane_d.isinf(), curr_d, plane_d)
        ref_p = normal_depth_to_plane(cameras, ref_n, ref_d)
        return ref_p


    def perturb(self, cameras, plane_map):
        _, H, W, _ = plane_map.shape
        dev = plane_map.device

        normal_map = plane_map[..., :3]
        depth_map = plane_to_depth(cameras, plane_map)
        s = cameras.scale()
        disp_map = s / depth_map
        # largest
        min_disp = s / cameras.ranges[0]
        # smallest
        max_disp = s / cameras.ranges[1]
        num_disp_scales = int(math.log(max_disp, 10) + 3)

        NR = max(min(6, num_disp_scales), 3)
        normal_pert_rate = 2 ** -torch.arange(NR).view(-1, 1, 1, 1).to(dev).float()
        normal_perturbs = (torch.rand((NR, H, W, 3), device=dev) * 2 - 1) * normal_pert_rate
        pert_n = NF.normalize((normal_map + normal_perturbs), p=2, dim=-1)
        r = NF.normalize(cameras.camera_rays(), p=2, dim=-1)
        dots = (pert_n * r).sum(-1, keepdim=True)
        pert_n = torch.where(dots >= 0, -pert_n, pert_n)
        def_n = plane_map[..., :3]
        pert_n = torch.where(dots.abs() < 1e-5, def_n, pert_n)

        d_std = (0.5 * depth_map ** 2 / s)
        d_pert_scale = d_std * 2 ** -torch.arange(NR).view(-1, 1, 1, 1).to(dev).float()
        d_perts = d_pert_scale * (torch.rand((NR, H, W, 1), device=dev) * 2 - 1)
        pert_d = depth_map + d_perts

        # pert_disp_rate = max_disp / 2 * 10 ** -torch.arange(NR).view(-1, 1, 1, 1).to(dev).float()
        # min_disp_d = (disp_map - pert_disp_rate).clamp_max(min_disp)
        # max_disp_d = (disp_map + pert_disp_rate).clamp_min(max_disp)
        # pert_disp = torch.rand((NR, H, W, 1), device=dev) * (min_disp_d - max_disp_d) + max_disp_d
        # pert_d = (s / pert_disp).clamp(*cameras.ranges)

        pert_p = normal_depth_to_plane(cameras, pert_n, pert_d)

        return pert_p
        
    def plane_similarity(self, cameras, ref_plane, src_plane, d_sigma, n_sigma):
        r = cameras.camera_rays()
        rs = self.neigh_propagator.propagate_rays(r).unsqueeze(0)

        # n . (o + l d) + disp = 0
        # n . r * d = disp
        # o + l * d = disp / n
        # dist = (disp / n - o) / r
        # n * p / disp = 1
        ref_n = ref_plane[..., :3].unsqueeze(0)
        ref_disp = ref_plane[..., 3:].unsqueeze(0)
        ref_plane_d = -ref_disp / (rs * ref_n).sum(-1, keepdim=True)

        src_n = src_plane[..., :3].unsqueeze(1)
        src_disp = src_plane[..., 3:].unsqueeze(1)
        src_plane_d = -src_disp / (rs * src_n).sum(-1, keepdim=True)

        ref_d = plane_to_depth(cameras, ref_plane)

        s = cameras.scale()
        std = 0.5 * ref_d ** 2 / s
        point_dists = ref_plane_d - src_plane_d

        norm_dists = (point_dists.abs().mean(1) / std)
        rs = (-0.5 * norm_dists ** 2).exp().clamp(1e-5, 1-1e-5)
        rs[rs.isnan()] = 0
        return rs

    def plane_similarities(self, cameras, plane_probs, plane_maps, gt_plane, d_sigma, n_sigma):
        r = cameras.camera_rays()
        rs = self.neigh_propagator.propagate_rays(r).unsqueeze(0)

        plane_ns = plane_maps[..., :3].unsqueeze(1)
        plane_disps = plane_maps[..., 3:].unsqueeze(1)
        # plane_ds = (-plane_disps) / (r * plane_ns).sum(-1, keepdim=True)
        plane_ds = -plane_disps / (rs * plane_ns).sum(-1, keepdim=True)

        gt_n = gt_plane[..., :3].unsqueeze(1)
        gt_disp = gt_plane[..., 3:].unsqueeze(1)
        # gt_d = -gt_disp / (r * gt_n).sum(-1, keepdim=True)
        gt_dc = -gt_disp / (r * gt_n).sum(-1, keepdim=True)
        gt_d = -gt_disp / (rs * gt_n).sum(-1, keepdim=True)
        gt_d[gt_d.isnan()] = cameras.ranges[0]

        s = cameras.scale()
        std = 0.5 * gt_dc ** 2 / s
        point_dists = plane_ds - gt_d

        norm_dists = (point_dists.abs().mean(1) / std).squeeze(0)
        rewards = (-0.5 * norm_dists ** 2).exp()
        rewards[rewards.isnan()] = 0

        # n_dist = (plane_ns * gt_n).sum(-1, keepdim=True).clamp(-1, 1).acos()
        # d_dist = point_dists.abs()

        # n_g = (-0.5 * (n_dist / n_sigma) ** 2).exp()
        # d_g = (-0.5 * (d_dist / std) ** 2).exp()
        # rs = (n_g * d_g)
        mask = (rs * gt_n == 0).all(-1, keepdim=True).any(1)


        nlls = -((rewards) * plane_probs.log()).sum(0, keepdim=True)
        nlls[mask] = 0
        return nlls

    def compute_pairwise_score(self, cameras, ref_plane, src_plane):
        r = cameras.camera_rays()
        s = cameras.scale()

        # compute depth
        ref_d = plane_to_depth(cameras, ref_plane).clamp(*cameras.ranges)
        ref_n = ref_plane[..., :3]
        std = to_bchw(0.5 * (ref_d ** 2) / s)

        src_planes = self.neigh_propagator.propagate_planes(src_plane)
        src_rs = self.neigh_propagator.propagate_points(r)

        src_ns = src_planes[..., :3]
        src_disps = src_planes[..., 3:]
        src_dots = (src_rs * src_ns).sum(-1, keepdim=True)
        src_ds = (-src_disps / src_dots).clamp(*cameras.ranges)

        # 1xHxWx3
        ref_p = r * ref_d
        # 4xHxWx3
        src_ps = src_ds * src_rs

        # ref->src consistency
        r2s = (ref_n * (src_ps - ref_p)).sum(dim=-1).abs()
        s2r = (src_ns * (ref_p - src_ps)).sum(dim=-1).abs()
        score = (r2s + s2r).unsqueeze(0)
        return (-score / std).exp()


    def select_plane(self, octave, cameras, features, curr_p, prop_p, hidden_states, view_i, view_w, P, DIL, soft):
        ref_p, ref_att = self.feat_scorer.compute_att(octave, cameras, features, P, DIL, 1)

        costs = []
        hiddens = []

        for i, plane_map in enumerate(prop_p):
            plane_map = plane_map.unsqueeze(0)
            wgc = self.feat_scorer.compute_wgc(cameras, view_i, view_w, ref_att, ref_p, features, plane_map, P, DIL, 1)
            joint_score = wgc 

            if(not self.hparams.skip_pairwise):
                pairwise = self.compute_pairwise_score(cameras, plane_map, curr_p)
                joint_score = torch.cat((wgc, pairwise), 1)

            cost, hidden = self.depth_regularizer(joint_score, hidden_states)
            costs.append(cost)
            hiddens.append(hidden)

        # aggregate and sample based on scores
        # NDxGxHxW
        cost_volume = torch.cat(costs).permute(0, 2, 3, 1)

        # NDxHNxHCxHxW
        hidden_volume = torch.stack(hiddens)

        #NDx1xHxW

        plane_ps, plane_i = self.sample(cost_volume, 1, soft=soft, softmax=True)
        plane_s = cost_volume.gather(0, plane_i)
        plane_is = plane_i.repeat(1, 1, 1, 4)
        plane_map = prop_p.gather(0, plane_is)

        HN, HC, H, W = hidden_states.shape
        hidden_i = plane_i.view(-1, 1, 1, H, W).repeat(1, HN, HC, 1, 1)
        hidden_s = hidden_volume.gather(0, hidden_i)

        return {
            'plane_maps': prop_p,
            'plane_map': plane_map,
            'plane_probs': plane_ps,
            'plane_score': plane_s,
            'hidden_states': hidden_s
        }



    def step(self, octave, cameras, features, plane_map, hidden_states, P, DIL, K, soft, details=False, train=False):
        # compute views based on current depth
        view_res = self.select_views(octave, cameras, features, plane_map, P, DIL, K, soft, details)
        view_i = view_res['view_indices']
        view_w = view_res['view_weights']

        # propagate d using plane
        plane_maps = self.propagate(octave, cameras, plane_map, train)

        # compute scores for each depth based on current views
        plane_res = self.select_plane(octave, cameras, features, plane_map, plane_maps, hidden_states, view_i, view_w, P, DIL, soft)
        return {**view_res, **plane_res}


    def forward(self, 
                images: torch.FloatTensor, 
                K: torch.FloatTensor, 
                E: torch.FloatTensor, 
                P: int,
                DIL: int,
                ranges: tuple, 
                soft: bool=False,
                num_views=2,
                num_iterations=8,
                num_refine=2,
                train=False,
                details=False)->tuple:
        '''
        Arguments
            - images: NxCxHxW tensor
            - K: Nx3x3 tensor
            - E: Nx4x4 tensor
            - ranges: tuple (min_d, max_d)
            - soft: bool (true to perform 'sampling' over 'top_k')

        Returns
            - plane_map: 1xHxWx4 tensor representing oriented points per pixel 
                         (i.e) a, b, c, d in the plane equation: ax + by + cz + d
            - steps: dict of lists, where each list contains step-wise data
        '''
        #######################################
        # Pre-process inputs before iteration
        cameras = MVSCamera(K, E, P, images.shape[-2:], ranges)

        # iterate from coarse to fine
        feature_layers = self.feature_extractor(images)

        plane_map = None
        hidden_states = None
        HN = self.hparams.num_hidden_states
        HC = self.hparams.num_hidden_channels
        K = num_views

        total_iter = 0
        steps = {}
        octaves = reversed(sorted(list(feature_layers.keys())))


        #################################
        # perform patch-match iteration
        for i, octave in enumerate(octaves):
            features = feature_layers[octave]
            H, W = features.shape[-2:]
            cameras.resize((H, W))
            steps[octave] = []
            num_iter = num_iterations if i == 0 else num_refine

            # resize plane / hidden states
            if(plane_map is None):
                plane_map = generate_random_plane_maps(cameras, 1, *ranges)
                hidden_states = torch.zeros((HN, HC, H, W), device=plane_map.device)
            else:
                plane_map = resize_plane_maps(cameras, plane_map, (H, W))
                hidden_states = NF.interpolate(hidden_states, size=(H, W), mode='nearest')


            for iteration in range(num_iter):
                res = self.step(octave, cameras, 
                                features, 
                                plane_map, hidden_states, 
                                P, DIL, K, 
                                soft, details, train)
                if(details):
                    steps[octave].append(res)
                total_iter += 1
                plane_map = res['plane_map']

        if(details):
            return plane_map, steps
        else:
            return plane_map

    def run_and_log(self, batch, batch_idx, train, log_prefix):
        ###########################
        # setup data for training
        images = batch['images']
        depths = to_bhwc(batch['depths'])
        K = batch['intrinsics'].clone()
        E = batch['extrinsics']
        ranges = batch['ranges']
        min_d, max_d = ranges
        n_sigma = self.hparams.n_sigma_train if train else self.hparams.n_sigma
        d_sigma = (max_d - min_d) / 256.0 if train else self.hparams.d_sigma
        num_views = self.hparams.num_train_view_selection if train else self.hparams.num_view_selection
        num_iter = 3 if train else 8
        num_refine = 1 if train else 2
        P = self.hparams.patch_size
        DIL = self.hparams.patch_dilation

        ##################################
        # perform patch match inference
        # perform inference
        inf_p, steps = self.forward(images, K, E, P, DIL, ranges, num_iterations=num_iter, num_refine=num_refine, num_views=num_views, soft=train, details=True, train=train)
        cameras = MVSCamera(K, E, P, images.shape[-2:], ranges)
        cameras.resize(inf_p.shape[1:3])

        inf_d = plane_to_depth(cameras, inf_p)
        inf_n = inf_p[..., :3]

        # resize intrinsics and obtain gt plane
        gt_ds = get_gt_depth(cameras, depths)
        gt_d = gt_ds[:1]
        gt_visible, gt_invisible, gt_valids = compute_visibility_map(cameras, gt_ds)
        gt_n = compute_normal_from_depth(cameras, gt_d, True)
        gt_p = normal_depth_to_plane(cameras, gt_n, gt_d)

        # compute confidence
        depth_diff = (inf_d - gt_d).abs()
        d_s = cameras.scale()
        d_std = (0.5 * gt_d ** 2 / d_s).clamp_max(d_sigma)
        gt_c = (depth_diff < d_std).float()
        # compute loss
        d_loss = 0
        n_loss = 0
        v_loss = 0
        gamma = 1.0


        min_octave = min(steps.keys())
        curr_octave = min_octave
        octaves = sorted(list(steps.keys()))

        # we want to iterate in reverse
        out_shape = inf_d.shape[1:3]
        g_is = []
        K = num_views
        for octave in octaves:
            
            step = steps[octave]
            step_keys = list(range(len(step)))
            step_size = step[step_keys[-1]]['plane_map'].shape[1:3]

            # resize camera
            cameras.resize(step_size)
            curr_octave = octave

            # compute visibility map / ground truth depth
            step_gts = get_gt_depth(cameras, depths)
            step_gt = step_gts[:1]
            step_gt_visible, step_gt_invisible, step_gt_valids = compute_visibility_map(cameras, step_gts)
            step_gt_clamped = step_gt.clamp(*ranges)
            step_gn = compute_normal_from_depth(cameras, step_gt_clamped, False)
            step_gp = normal_depth_to_plane(cameras, step_gn, step_gt_clamped)

            for i in reversed(step_keys):
                st = step[i]
                step_p = st['plane_map']
                v_nll = st['view_nll']
                step_ps = st['plane_maps']
                step_ds = plane_to_depth(cameras, step_ps)
                step_ns = step_ps[..., :3]
                step_p_probs = st['plane_probs']

                # compute depth loss
                d_s = cameras.scale()
                d_std = (0.5 * step_gt ** 2 / d_s) / 4
                d_std = d_std.clamp_max(d_sigma)
                d_dists = (step_gt - step_ds).abs()
                d_probs = (-d_dists / d_std).exp()
                d_nll = (-d_probs * step_p_probs.log()).sum(0, keepdim=True)
                d_loss += resize_bhwc(d_nll, out_shape)

                # compute normal loss
                n_dists = (step_gn * step_ns).sum(-1, keepdim=True).clamp(-1, 1).acos()
                n_std = 1.5 * math.pi / 180.0
                n_probs = (-n_dists.abs() / n_std).exp()

                # we want the n_probs to be contitioned on d_probs
                n_nll = -(d_probs * n_probs * step_p_probs.log()).sum(0, keepdim=True)
                n_loss += resize_bhwc(n_nll, out_shape)

                # compute step-wise reward
                rs = self.plane_similarity(cameras, step_p, step_gp, d_sigma, n_sigma)

                # compute all future reward
                g_i = resize_bhwc(rs, out_shape)
                g_is.append(g_i)
                g_t = torch.stack(g_is).sum(0)

                # compute likelihood of policy
                # pg_loss += g_t * self.resize_bhwc(d_nll, out_shape)
                v_loss += g_t * resize_bhwc(v_nll, out_shape)

        # compute loss
        # valid_points = (gt_visible.sum(0, keepdim=True) >= K) & (gt_n != 0).all(-1, keepdim=True)
        valid_ds = gt_d > 0
        valid_ns = (gt_n != 0).any(-1, keepdim=True)
        valid_vs = gt_visible.sum(0, keepdim=True) >= 1
        valid_cs = gt_d > 0
        valid_points = (gt_visible.sum(0, keepdim=True) >= 1) & (gt_n != 0).all(-1, keepdim=True) & ((~gt_p.isnan()).all(-1, keepdim=True))

        d_loss = d_loss[valid_ds].mean() * self.lambda_depth()
        n_loss = n_loss[valid_ns].mean() * 100
        vs_loss = v_loss[valid_vs].mean()

        loss = d_loss + n_loss + vs_loss
        if (loss.isnan() | ~loss.isfinite()):
            print('nan detected')

        ###############################
        # compute evaluation metrics
        mean_abs_dist = self.mean_abs_dist(inf_d, gt_d)
        reward = ((inf_d - gt_d).abs() < d_sigma).float()
        pi_1t = self.percent_inlier(inf_d, gt_d, d_sigma)
        pi_2t = self.percent_inlier(inf_d, gt_d, d_sigma * 2)
        pi_5t = self.percent_inlier(inf_d, gt_d, d_sigma * 5)
        pi_10t = self.percent_inlier(inf_d, gt_d, d_sigma * 10)
        pi_20t = self.percent_inlier(inf_d, gt_d, d_sigma * 20)
        pi_50t = self.percent_inlier(inf_d, gt_d, d_sigma * 50)

        #########################
        # log validation stuff
        self.log(f'{log_prefix}/loss', loss)
        self.log(f'{log_prefix}/vs_loss', vs_loss)
        self.log(f'{log_prefix}/d_loss', d_loss)
        self.log(f'{log_prefix}/n_loss', n_loss)
        self.log(f'{log_prefix}/pg_reward', g_t[gt_d > 0].mean())
        self.log(f'{log_prefix}/reward', reward[gt_d > 0].mean())
        self.log(f'{log_prefix}/mean_absolute_dist', mean_abs_dist)
        self.log(f'{log_prefix}/percent_inlier_1_theta', pi_1t)
        self.log(f'{log_prefix}/percent_inlier_2_theta', pi_2t)
        self.log(f'{log_prefix}/percent_inlier_5_theta', pi_5t)
        self.log(f'{log_prefix}/percent_inlier_10_theta', pi_10t)
        self.log(f'{log_prefix}/percent_inlier_20_theta', pi_20t)
        self.log(f'{log_prefix}/percent_inlier_50_theta', pi_50t)

        # add figure 
        if((batch_idx % self.hparams.image_log_interval) == 0):
            cameras.resize(inf_p.shape[1:3])
            gt_ranges = gt_d[gt_d > 0].min(), gt_d.max()
            gt_d_image = to_depth_image(gt_d, gt_ranges)
            gt_n_image = to_normal_image(gt_n)
            inf_d_image = to_depth_image(inf_d, gt_ranges)
            inf_dn = compute_normal_from_depth(cameras, inf_d, False)
            inf_n_image = to_normal_image(inf_n)
            inf_dn_image = to_normal_image(inf_dn)
            inf_dx_image = to_bin_image(inf_n[..., 0])
            inf_dy_image = to_bin_image(inf_n[..., 1])
            last_step = steps[min_octave][-1]

            view_scores = last_step['view_scores']
            view_probs = last_step['view_probs']
            view_valids = last_step['view_valids']

            view_score_image = to_view_prob_images(view_scores)
            view_probs_image = to_view_prob_images(view_probs)

            view_selection_image = to_view_selection_images(view_scores, K)
            inf_valids_image = to_view_prob_images(view_valids.float())
            inf_valid_points_image = to_conf_image((view_valids.sum(0, keepdim=True) >= K).float())

            gt_valid_image = to_view_prob_images(gt_valids.float())
            gt_visible_image = to_view_prob_images(gt_visible.float())
            gt_valid_points_image = to_conf_image(valid_points.float())

            self.logger.experiment.add_images(f'{log_prefix}/A_ref_image', NF.interpolate(images[:1], gt_d.shape[1:3]), self.global_step)
            self.logger.experiment.add_images(f'{log_prefix}/A_src_images', NF.interpolate(images[1:], gt_d.shape[1:3]), self.global_step)
            self.logger.experiment.add_image(f'{log_prefix}/A_Depths gt', gt_d_image, self.global_step)
            self.logger.experiment.add_image(f'{log_prefix}/A_Depths gtn', gt_n_image, self.global_step)
            self.logger.experiment.add_image(f'{log_prefix}/A_Depths inf', inf_d_image, self.global_step)
            self.logger.experiment.add_image(f'{log_prefix}/A_Normal inf', inf_n_image, self.global_step)
            self.logger.experiment.add_image(f'{log_prefix}/A_Normal infd', inf_dn_image, self.global_step)
            self.logger.experiment.add_image(f'{log_prefix}/A_Normal nx', inf_dx_image, self.global_step)
            self.logger.experiment.add_image(f'{log_prefix}/A_Normal ny', inf_dy_image, self.global_step)

            self.logger.experiment.add_images(f'{log_prefix}/B_view_probs', view_probs_image, self.global_step)
            self.logger.experiment.add_images(f'{log_prefix}/B_view_scores', view_score_image, self.global_step)

            self.logger.experiment.add_images(f'{log_prefix}/C_valids_gt', gt_valid_image, self.global_step)
            self.logger.experiment.add_images(f'{log_prefix}/C_valids_inf', inf_valids_image, self.global_step)
            self.logger.experiment.add_images(f'{log_prefix}/C_visible_gt', gt_visible_image, self.global_step)
            self.logger.experiment.add_image(f'{log_prefix}/C_valid_points_gt', gt_valid_points_image, self.global_step)

            self.logger.experiment.add_image(f'{log_prefix}/C_valid_points_inf', inf_valid_points_image, self.global_step)
            self.logger.experiment.add_images(f'{log_prefix}/C_view_selection', view_selection_image, self.global_step)

            for octave, step in steps.items():
                step_d_image_list = []
                step_n_image_list = []

                for i, s in enumerate(step):
                    step_p = s['plane_map']
                    H, W =step_p.shape[1:3]
                    cameras.resize((H, W))
                    step_d = plane_to_depth(cameras, step_p)
                    step_n = step_p[..., :3]
                    step_d_image = to_depth_image(step_d, gt_ranges)
                    step_n_image = to_normal_image(step_n)
                    step_d_image_list.append(step_d_image)
                    step_n_image_list.append(step_n_image)
                step_d_images = np.stack(step_d_image_list)
                step_n_images = np.stack(step_n_image_list)
                self.logger.experiment.add_images(f'{log_prefix}/D_depth_map_{octave}', step_d_images, self.global_step)
                self.logger.experiment.add_images(f'{log_prefix}/D_normal_map_{octave}', step_n_images, self.global_step)
        return loss

    # -- validation code -- #
    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        return self.run_and_log(batch, batch_idx, train=True, log_prefix='train')

    def validation_step(self, batch, batch_idx):
        self.run_and_log(batch, batch_idx, train=False, log_prefix='val')

    def test_step(self, batch, batch_idx):
        self.run_and_log(batch, batch_idx, train=False, log_prefix='test')