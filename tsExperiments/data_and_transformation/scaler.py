# This file was adapted from https://github.com/zalandoresearch/pytorch-ts/blob/master/pts/modules/scaler.py
# under MIT License.

from abc import ABC, abstractmethod
from typing import Tuple
import torch
import torch.nn as nn
from torch.distributions import Normal
from gluonts.core.component import validated


class Scaler(ABC, nn.Module):
    def __init__(self, keepdim: bool = False, time_first: bool = True):
        super().__init__()
        self.keepdim = keepdim
        self.time_first = time_first

    @abstractmethod
    def compute_scale(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> torch.Tensor:
        pass

    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        data
            tensor of shape (N, T, C) if ``time_first == True`` or (N, C, T)
            if ``time_first == False`` containing the data to be scaled

        observed_indicator
            observed_indicator: binary tensor with the same shape as
            ``data``, that has 1 in correspondence of observed data points,
            and 0 in correspondence of missing data points.

        Returns
        -------
        Tensor
            Tensor containing the "scaled" data, shape: (N, T, C) or (N, C, T).
        Tensor
            Tensor containing the scale, of shape (N, C) if ``keepdim == False``,
            and shape (N, 1, C) or (N, C, 1) if ``keepdim == True``.
        """

        scale = self.compute_scale(data, observed_indicator)

        if self.time_first:
            dim = 1
        else:
            dim = 2
        if self.keepdim:
            scale = scale.unsqueeze(dim=dim)
            return data / scale, scale
        else:
            return data / scale.unsqueeze(dim=dim), scale


class MeanScaler(Scaler):
    """
    The ``MeanScaler`` computes a per-item scale according to the average
    absolute value over time of each item. The average is computed only among
    the observed values in the data tensor, as indicated by the second
    argument. Items with no observed data are assigned a scale based on the
    global average.

    Parameters
    ----------
    minimum_scale
        default scale that is used if the time series has only zeros.
    """

    @validated()
    def __init__(self, minimum_scale: float = 1e-10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("minimum_scale", torch.tensor(minimum_scale))

    def compute_scale(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> torch.Tensor:

        if self.time_first:
            dim = 1
        else:
            dim = 2

        # these will have shape (N, C)
        num_observed = observed_indicator.sum(dim=dim)
        sum_observed = (data.abs() * observed_indicator).sum(dim=dim)

        # first compute a global scale per-dimension
        total_observed = num_observed.sum(dim=0)
        denominator = torch.max(total_observed, torch.ones_like(total_observed))
        default_scale = sum_observed.sum(dim=0) / denominator

        # then compute a per-item, per-dimension scale
        denominator = torch.max(num_observed, torch.ones_like(num_observed))
        scale = sum_observed / denominator

        # use per-batch scale when no element is observed
        # or when the sequence contains only zeros
        scale = torch.where(
            sum_observed > torch.zeros_like(sum_observed),
            scale,
            default_scale * torch.ones_like(num_observed),
        )

        return torch.max(scale, self.minimum_scale).detach()


class NOPScaler(Scaler):
    """
    The ``NOPScaler`` assigns a scale equals to 1 to each input item, i.e.,
    no scaling is applied upon calling the ``NOPScaler``.
    """

    @validated()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_scale(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> torch.Tensor:
        if self.time_first:
            dim = 1
        else:
            dim = 2
        return torch.ones_like(data).mean(dim=dim)


class MeanStdScaler(Scaler):
    """
    The ``MeanStdScaler`` computes a per-item scale according to the mean
    and standard deviation over time of each item. The mean and standard
    deviation are computed only among the observed values in the data tensor,
    as indicated by the second argument. Items with no observed data are
    assigned a scale based on the global average and standard deviation.

    Parameters
    ----------
    minimum_scale
        default scale that is used if the time series has only zeros.
    """

    @validated()
    # def __init__(self, minimum_std: float = 1e-3, minimum_std_cst: float = 1e-4, add_minimum_std: bool = False, default_scale: bool = False, default_scale_cst: bool = False, *args, **kwargs):
    def __init__(
        self,
        minimum_std,
        minimum_std_cst,
        add_minimum_std,
        default_scale,
        default_scale_cst,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        if type(minimum_std) == str:
            minimum_std = float(minimum_std)
        else:
            minimum_std = minimum_std
        if type(minimum_std_cst) == str:
            minimum_std_cst = float(minimum_std_cst)
        else:
            minimum_std_cst = minimum_std_cst
        if type(add_minimum_std) == str:
            add_minimum_std = bool(add_minimum_std)
        else:
            add_minimum_std = add_minimum_std

        self.default_scale = default_scale
        self.default_scale_cst = default_scale_cst

        self.register_buffer("minimum_std", torch.tensor(minimum_std))
        self.register_buffer("minimum_std_cst", torch.tensor(minimum_std_cst))
        self.register_buffer("add_minimum_std", torch.tensor(add_minimum_std))

    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, std = self.compute_scale(data, observed_indicator)

        if self.time_first:
            dim = 1
        else:
            dim = 2
        if self.keepdim:
            mean = mean.unsqueeze(dim=dim)
            std = std.unsqueeze(dim=dim)
            return (data - mean) / std, mean, std
        else:
            return (data - mean.unsqueeze(dim=dim)) / std.unsqueeze(dim=dim), mean, std

    def compute_scale(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # Return both mean and std
        if self.time_first:
            dim = 1
        else:
            dim = 2

        # Compute mean
        num_observed = observed_indicator.sum(dim=dim)
        sum_observed = (data * observed_indicator).sum(dim=dim)
        denominator = torch.max(num_observed, torch.ones_like(num_observed))
        mean = sum_observed / denominator  # [batch_size, dim_ts]
        default_scale = sum_observed.sum(dim=0) / denominator

        # Compute standard deviation
        sum_squared = ((data - mean.unsqueeze(dim=dim)) ** 2 * observed_indicator).sum(
            dim=dim
        )
        std = torch.sqrt(sum_squared / denominator)  # [batch_size, dim_ts]

        # Use per-batch scale when no element is observed
        mean = torch.where(
            num_observed > torch.zeros_like(num_observed),
            mean,
            torch.zeros_like(mean),  # Default mean to 0 if no observation
        )

        # if self.default_scale:
        #     std = torch.where(
        #         num_observed > torch.zeros_like(num_observed),
        #         std,
        #         default_scale * torch.ones_like(std)  # Default std to 1 if no observation
        #     )
        # else :
        std = torch.where(
            num_observed > torch.zeros_like(num_observed),
            std,
            torch.ones_like(std),  # Default std to 1 if no observation
        )

        # Where the std is truly 0, we set it to 1
        # if self.default_scale_cst:
        # std = torch.where(std <= self.minimum_std_cst, default_scale * torch.ones_like(std), std)
        # else :
        std = torch.where(std <= self.minimum_std_cst, torch.ones_like(std), std)
        very_small_quantile = torch.quantile(std, 0.001)
        std = torch.where(
            std <= very_small_quantile, very_small_quantile * torch.ones_like(std), std
        )
        if self.add_minimum_std:
            std = std + self.minimum_std
            return mean.detach(), std.detach()
        else:
            return mean.detach(), std.detach()


class CenteredMeanScaler(MeanScaler):

    def compute_scale(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> torch.Tensor:

        if self.time_first:
            dim = 1
        else:
            dim = 2

        # these will have shape (N, C)
        num_observed = observed_indicator.sum(dim=dim)
        sum_observed = (data.abs() * observed_indicator).sum(dim=dim)

        # first compute a global scale per-dimension
        total_observed = num_observed.sum(dim=0)
        denominator = torch.max(total_observed, torch.ones_like(total_observed))
        default_scale = sum_observed.sum(dim=0) / denominator

        # then compute a per-item, per-dimension scale
        denominator = torch.max(num_observed, torch.ones_like(num_observed))
        scale = sum_observed / denominator

        # use per-batch scale when no element is observed
        # or when the sequence contains only zeros
        scale = torch.where(
            sum_observed > torch.zeros_like(sum_observed),
            scale,
            default_scale * torch.ones_like(num_observed),
        )

        scale = torch.max(scale, self.minimum_scale).detach()

        return scale, scale

    def forward(self, data: torch.Tensor, observed_indicator: torch.Tensor):
        # 1) Compute the empirical mean ignoring missing data
        mean = self.compute_scale(data, observed_indicator)[0]

        if self.time_first:
            dim = 1
        else:
            dim = 2

        if self.keepdim:
            mean = mean.unsqueeze(dim=dim)
            return (data - mean) / mean, mean, mean
        else:
            return (
                (data - mean.unsqueeze(dim=dim)) / mean.unsqueeze(dim=dim),
                mean,
                mean,
            )
