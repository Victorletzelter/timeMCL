"""
Copyright 2023 ServiceNow
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# This code is adapted from https://github.com/ServiceNow/TACTiS/blob/tactis-2/tactis/gluon/network.py
# network tactis
from gluonts.model import Input, InputSpec
from typing import Any, Dict

import torch
from torch import nn

from tsExperiments.models.project_models.tactis2.model.tactis import TACTiS


class TactisNetwork(nn.Module):
    """
    A shell on top of the TACTiS module, to be used during training only.
    loss ?
    """

    def __init__(
        self,
        num_series: int,
        model_parameters: Dict[str, Any],
        prediction_length: int,
        num_parallel_samples: int,
        context_length: int,
    ):
        """
        Parameters:
        -----------
        num_series: int
            Number of series of the data which will be sent to the model.
        model_parameters: Dict[str, Any]
            The parameters of the underlying TACTiS model, as a dictionary.
        """
        super().__init__()

        self.num_series = num_series
        self.target_dim = num_series

        self.model = TACTiS(num_series, **model_parameters)
        self.num_parallel_samples = num_parallel_samples
        self.prediction_length = prediction_length
        self.context_length = context_length

    def loss(
        self,
        past_target_norm: torch.Tensor,
        future_target_norm: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters:
        -----------
        past_target_norm: torch.Tensor [batch, time steps, series]
            The historical data that will be available at inference time.
        future_target_norm: torch.Tensor [batch, time steps, series]
            The data to be forecasted at inference time.

        Returns:
        --------
        loss: torch.Tensor []
            The loss function, averaged over all batches.
        """
        # The data coming from Gluon is not in the shape we use in the model, so transpose it.
        hist_value = past_target_norm.transpose(1, 2)
        pred_value = future_target_norm.transpose(1, 2)

        # For the time steps, we take for granted that the data is aligned with a constant frequency
        hist_time = torch.arange(
            0, hist_value.shape[2], dtype=int, device=hist_value.device
        )[None, :].expand(hist_value.shape[0], -1)
        pred_time = torch.arange(
            hist_value.shape[2],
            hist_value.shape[2] + pred_value.shape[2],
            dtype=int,
            device=pred_value.device,
        )[None, :].expand(pred_value.shape[0], -1)

        return self.model.loss(
            hist_time=hist_time,
            hist_value=hist_value,
            pred_time=pred_time,
            pred_value=pred_value,
        )

    def describe_inputs(self, batch_size=1) -> InputSpec:
        return InputSpec(
            {
                "past_target_norm": Input(
                    shape=(batch_size, self.context_length, self.num_series),
                    dtype=torch.float,
                ),
                "future_target_norm": Input(
                    shape=(
                        batch_size,
                        self.prediction_length,
                        self.num_series,
                    ),
                    dtype=torch.float,
                ),
            },
            zeros_fn=torch.zeros,
        )

    def forward(self, past_target_norm, future_target_norm=None) -> torch.Tensor:
        """
        Parameters:
        -----------
        past_target_norm: torch.Tensor [batch, time steps, series]
            The historical data that are available.

        Returns:
        --------
        samples: torch.Tensor [samples, batch, time steps, series]
            Samples from the forecasted distribution.
        """
        # The data coming from Gluon is not in the shape we use in the model, so transpose it.
        hist_value = past_target_norm.transpose(1, 2)

        # For the time steps, we take for granted that the data is aligned with a constant frequency
        hist_time = torch.arange(
            0, hist_value.shape[2], dtype=int, device=hist_value.device
        )[None, :].expand(hist_value.shape[0], -1)
        pred_time = torch.arange(
            hist_value.shape[2],
            hist_value.shape[2] + self.prediction_length,
            dtype=int,
            device=hist_value.device,
        )[None, :].expand(hist_value.shape[0], -1)

        samples = self.model.sample(
            num_samples=self.num_parallel_samples,
            hist_time=hist_time,
            hist_value=hist_value,
            pred_time=pred_time,
        )

        # The model decoder returns both the observed and sampled values, so removed the observed ones.
        # Also, reorder from [batch, series, time steps, samples] to GluonTS expected [batch, samples, time steps, series].
        return samples[:, :, -self.prediction_length :, :].permute((0, 3, 2, 1))
