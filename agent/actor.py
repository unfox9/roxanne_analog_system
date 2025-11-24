import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd

import utils

class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p( - (-x).log1p()))

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()
    
    def _inverse(self, y):
        return self.atanh(y)

    def log_abs_det_jocobian(self, x, y):
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))

class SquashedNormal(pyd.transfomed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

        @property
        def mean(self):
            mu = self.loc
            for tr in self.transforms:
                mu = tr(mu)
            return mu
        
class DiagGaussianActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth,
                 log_std_bounds):
        super().__init__()

        self.log_std_bounds = log_std_bounds
        self.trunk = utils.mlp(obs_dim, hidden_dim, 2 * action_dim,
                         hidden_depth)
        
        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs):
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)