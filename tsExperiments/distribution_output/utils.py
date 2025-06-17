# This file was adapted from https://github.com/zalandoresearch/pytorch-ts/blob/master/pts/modules/distribution_output.py
# under MIT License.

from abc import ABC, abstractclassmethod
import warnings
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import (
    Distribution,
    Beta,
    NegativeBinomial,
    StudentT,
    Normal,
    Categorical,
    MixtureSameFamily,
    Independent,
    LowRankMultivariateNormal,
    MultivariateNormal,
    TransformedDistribution,
    AffineTransform,
    Poisson,
)
from gluonts.torch.distributions.distribution_output import DistributionOutput
from gluonts.core.component import validated


class LowRankMultivariateNormalOutput(DistributionOutput):
    @validated()
    def __init__(
        self,
        dim: int,
        rank: int,
        sigma_init: float = 1.0,
        sigma_minimum: float = 1e-3,
    ) -> None:
        self.distr_cls = LowRankMultivariateNormal
        self.dim = dim
        self.rank = rank
        self.sigma_init = sigma_init
        self.sigma_minimum = sigma_minimum
        self.args_dim = {"loc": dim, "cov_factor": dim * rank, "cov_diag": dim}

    def domain_map(self, loc, cov_factor, cov_diag):
        diag_bias = (
            self.inv_softplus(self.sigma_init**2) if self.sigma_init > 0.0 else 0.0
        )

        shape = cov_factor.shape[:-1] + (self.dim, self.rank)
        cov_factor = cov_factor.reshape(shape)
        cov_diag = F.softplus(cov_diag + diag_bias) + self.sigma_minimum**2

        return loc, cov_factor, cov_diag

    def inv_softplus(self, y):
        if y < 20.0:
            return np.log(np.exp(y) - 1.0)
        else:
            return y

    @property
    def event_shape(self) -> Tuple:
        return (self.dim,)
