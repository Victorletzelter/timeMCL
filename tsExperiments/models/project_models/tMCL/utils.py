"""
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

import torch
from typing import Optional
import torch
import torch.nn.functional as F
from gluonts.core.component import validated
from typing import Tuple
from gluonts.torch.distributions.distribution_output import DistributionOutput
import torch
import torch.nn as nn

# defining the output distrbution in our data. here it is MCL.
class MCLOutput(DistributionOutput):
    @validated()
    def __init__(self, diffusion, input_size, cond_size):
        self.args_dim = {"cond": cond_size}
        self.diffusion = diffusion
        self.dim = input_size

    @classmethod
    def domain_map(cls, cond):
        """It is used to convert arguments to the right shape and domain!

        Depends on the type of distribution"""
        return (cond,)

    def distribution(self, distr_args, scale=None):
        """we overrided the distribution method!
        The goal was to retun a distribution objet. We return a DIFFUSION OBJECT"""
        (cond,) = (
            distr_args  # les distr_args ne servent à rien ici ? On renvoie self.diffusion
        )
        if scale is not None:
            self.diffusion.scale = scale
        self.diffusion.cond = cond

        return self.diffusion

    @property
    def event_shape(self) -> Tuple:
        # number of event dimension i.e. event shape.
        return (self.dim,)


class MeanLayer(nn.Module):
    def __init__(self, dim=None, keepdim=False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return torch.mean(x, dim=self.dim, keepdim=self.keepdim)


class tMCL(nn.Module):
    def __init__(
        self,
        cond_dim,
        dim_ts,
        hidden_dim,
        n_hypotheses,
        device,
        score_loss_weight,
        mcl_loss_type,
        wta_mode,
        wta_mode_params,
        single_linear_layer,
        backbone_deleted,
        div_by_std,
    ):
        super().__init__()
        self.scale = None
        self.wta_mode = wta_mode
        self.wta_mode_params = wta_mode_params
        self.dim_ts = dim_ts
        self.mcl_loss_type = mcl_loss_type
        self.n_hypotheses = n_hypotheses
        self.score_loss_weight = score_loss_weight
        self.div_by_std = div_by_std
        # =========================
        # (1) backbone
        # =========================
        # Identity function
        hidden_dim = cond_dim
        self.backbone = nn.Identity()
        # =========================
        # (2) Heads de prédiction
        # =========================
        # We want to have dim_ts for each time-step
        if single_linear_layer:
            self.prediction_heads = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(hidden_dim, dim_ts)).to(device)
                    for _ in range(n_hypotheses)
                ]
            )
        else:
            self.prediction_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(p=0.1),
                        nn.Linear(hidden_dim, dim_ts),
                    ).to(device)
                    for _ in range(n_hypotheses)
                ]
            )

        # =========================
        # (3) Heads de score
        # =========================
        # We want a scalar score for each time-step
        self.score_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, 1), nn.Sigmoid(), nn.Flatten(start_dim=0)
                ).to(device)
                for _ in range(n_hypotheses)
            ]
        )

        self.score_aggregator = nn.Sequential(
            MeanLayer(dim=1),
        ).to(device)

        self.score_loss = nn.BCELoss()
        self.nb_call_log_prob = 0

    def forward(self, distr_args):
        # distr_args : [B, T, cond_dim]
        B, T, C_in = distr_args.shape

        # (A) Flatten (batch, time) => [B*T, cond_dim]
        x = distr_args.reshape(B * T, C_in)

        # (B) Pass through the backbone => [B*T, hidden_dim]
        features = self.backbone(x)

        # (C) Prediction heads => we want [B, K, T, dim_ts]
        pred_list = []
        for head in self.prediction_heads:
            # => [B*T, dim_ts]
            out = head(features)
            # => [B, T, dim_ts]
            out = out.view(B, T, -1)

            pred_list.append(out)
        # => [B, K, T, dim_ts]
        prediction_list = torch.stack(pred_list, dim=1)

        # (D) Score heads => [B, K, T]
        scr_list = []
        for head in self.score_heads:
            scr = head(features)  # => [B*T] after Flatten
            scr = scr.view(B, T)  # => [B, T]
            # Apply score aggregator (currently a mean over the time dimension)
            scr = self.score_aggregator(scr)  # [B]
            scr_list.append(scr)
        score_list = torch.stack(scr_list, dim=1)  # => [B, K]

        return prediction_list, score_list

    def compute_loss_min_ext_sum(self, prediction_list, score_list, target_list):
        # prediction_list : [batch_size, K, longueur_ts, dim_ts]. here : loss outside the sum.
        # target_list     : [batch_size, longueur_ts, dim_ts]
        # pairwise_mse = torch.sum((prediction_list - target_list.unsqueeze(1))**2, dim=(2,3)) #sum on dim 2 and 3.
        # mcl_loss, target_assignment = pairwise_mse.min(dim=1)  #returning the best hypothesis.
        if self.scale is not None and not isinstance(self.scale, dict):
            pairwise_mse = torch.sum(
                (prediction_list - target_list.unsqueeze(1)) ** 2, dim=-1
            )  # shape [batch_size, K, longueur_ts]
            pairwise_mse = torch.sqrt(pairwise_mse)
        else:
            pairwise_mse = torch.sum(
                (prediction_list - target_list.unsqueeze(1)) ** 2, dim=-1
            )  # shape [batch_size, K, longueur_ts]
        pairwise_mse = pairwise_mse.mean(dim=-1)  # shape [batch_size, K]
        if self.wta_mode == "wta":
            mcl_loss, target_assignment = pairwise_mse.min(dim=1)
        elif self.wta_mode == "relaxed-wta":
            n_hypotheses = pairwise_mse.shape[1]
            assert (
                n_hypotheses > 1
            ), "relaxed-wta is only supported for at least 2 hypotheses"
            epsilon = self.wta_mode_params["epsilon"]
            winner, target_assignment = pairwise_mse.min(
                dim=1
            )  # Winner and target_assigment of shape [batch_size]
            mcl_loss = (1 - epsilon * (n_hypotheses) / (n_hypotheses - 1)) * winner + (
                epsilon / (n_hypotheses - 1)
            ) * pairwise_mse.sum(dim=1)
        elif self.wta_mode == "awta":
            _, target_assignment = pairwise_mse.min(dim=1)
            amcl_weights = torch.softmax(
                -pairwise_mse / self.wta_mode_params["temperature"], dim=1
            ).detach()  # shape [batch_size,K]
            mcl_loss = amcl_weights * pairwise_mse
        mcl_loss_mean = (
            mcl_loss.mean()
        )  # for the global mcl Loss. We take the mean over the batch.

        # additionnal features
        prediction_assignment = torch.nn.functional.one_hot(
            target_assignment, num_classes=prediction_list.shape[1]
        ).float()  # [batch_size, K]
        # prediction_assignment = prediction_assignment.unsqueeze(2).repeat(1, 1, prediction_list.shape[2])  # [batch_size, K, longueur_ts]

        score_loss = self.score_loss(score_list, prediction_assignment)

        return mcl_loss_mean, score_loss, target_assignment

    def loss_in_sum(self, prediction_list, score_list, target_list):
        # 1) Squared error point by point :
        error = (prediction_list - target_list.unsqueeze(1)) ** 2
        error_timestep = error.sum(dim=3)
        min_error_timestep, target_assignment = error_timestep.min(dim=1)
        sum_over_time = min_error_timestep.sum(dim=1)
        loss = sum_over_time.sum()

        target_assignment_mode, _ = torch.mode(target_assignment, dim=1)  # [batch_size]
        prediction_assignment = torch.nn.functional.one_hot(
            target_assignment_mode, num_classes=prediction_list.shape[1]
        ).float()  # [batch_size, K]
        score_loss = self.score_loss(score_list, prediction_assignment)

        return loss, score_loss, target_assignment

    def log_prob(self, target, distr_args):
        self.nb_call_log_prob += 1

        if self.scale is not None:
            if not isinstance(self.scale, dict):
                pass
            elif len(self.scale) == 1:
                target /= self.scale["scale"]  # We scale as in timegrad.
            elif len(self.scale) == 2:
                mean = self.scale["mean"]
                std = self.scale["std"]
                if self.div_by_std:
                    target = (target - mean) / std
                else:
                    target = target - mean
            else:
                raise ValueError(
                    "Scale must be a scalar or a dictionary with 'mean' and (optionally) 'std' key"
                )

        prediction_list, score_list = self.forward(
            distr_args
        )  # batch_size*K*length_ts*dim_ts

        if self.mcl_loss_type == "min_ext_sum":
            mcl_loss, score_loss, target_assignment = self.compute_loss_min_ext_sum(
                prediction_list, score_list, target
            )  # computing the loss i.e. taking the best pred at each time.
        elif self.mcl_loss_type == "min_in_sum":
            mcl_loss, score_loss, target_assignment = self.loss_in_sum(
                prediction_list, score_list, target
            )

        return (
            mcl_loss + self.score_loss_weight * score_loss,
            target_assignment,
            score_loss,
        )  # .unsqueeze(1) - hyperparameter

    def sample(self, cond):
        # TODO. implementing the sample for MCL.
        prediction_list, score_list = self.forward(cond)
        # batch_size, length, dim_ts = cond.shape
        # output = torch.randn(batch_size, length, self.dim_ts)
        if self.scale is not None:
            if not isinstance(self.scale, dict):
                prediction_list = self.scale(prediction_list, "denorm")
            elif len(self.scale) == 1:
                prediction_list *= self.scale["scale"].unsqueeze(1)  # the scale
            elif len(self.scale) == 2:
                mean = self.scale["mean"]
                std = self.scale["std"]
                if self.div_by_std:
                    prediction_list = prediction_list * std.unsqueeze(
                        1
                    ) + mean.unsqueeze(1)
                else:
                    prediction_list = prediction_list + mean.unsqueeze(1)

        return prediction_list, score_list
