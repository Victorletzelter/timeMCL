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

import lightning.pytorch as pl
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tsExperiments.models.project_models.tMCL.timeMCL_network import timeMCLNetwork
from gluonts.itertools import select
from gluonts.torch.model.lightning_util import has_validation_loop
import numpy as np


class timeMCL_lighting(pl.LightningModule):

    def __init__(
        self,
        model_kwargs: dict,
        optim_kwargs: dict,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_kwargs = model_kwargs

        self.model = timeMCLNetwork(
            **model_kwargs,
        )

        # Training parameters
        self.lr = optim_kwargs["lr"]
        self.weight_decay = optim_kwargs["weight_decay"]
        self.patience = optim_kwargs["patience"]

        self.inputs = self.model.describe_inputs()
        self.example_input_array = self.inputs.zeros()
        self.first_training_step = True

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def scheduler_temperature(self, epoch):
        if self.model_kwargs["wta_mode_params"]["scheduler_mode"] == "constant":
            return self.model_kwargs["wta_mode_params"]["temperature_ini"]
        elif self.model_kwargs["wta_mode_params"]["scheduler_mode"] == "linear":
            return (
                self.model_kwargs["wta_mode_params"]["temperature_ini"]
                - (self.model_kwargs["wta_mode_params"]["temperature_ini"])
                * epoch
                / self.trainer.max_epochs
            )
        elif self.model_kwargs["wta_mode_params"]["scheduler_mode"] == "exponential":
            return (
                self.model_kwargs["wta_mode_params"]["temperature_ini"]
                * self.model_kwargs["wta_mode_params"]["temperature_decay"] ** epoch
            )

    def on_train_epoch_start(self) -> None:
        """Lightning hook that is called when a training epoch starts."""
        if "wta_mode" in self.model_kwargs and "awta" in self.model_kwargs["wta_mode"]:
            # set the temperature
            self.temperature = self.scheduler_temperature(self.current_epoch)

            if "temperature_lim" in self.model_kwargs["wta_mode_params"]:
                temperature_lim = self.model_kwargs["wta_mode_params"][
                    "temperature_lim"
                ]
            else:
                temperature_lim = 1e-10

            if self.temperature < temperature_lim:
                if (
                    "wta_after_temperature_lim" in self.model_kwargs["wta_mode_params"]
                    and self.model_kwargs["wta_mode_params"][
                        "wta_after_temperature_lim"
                    ]
                    is True
                ):
                    self.model_kwargs["wta_mode"] = "wta"
                else:
                    self.temperature = temperature_lim

            # update the loss
            self.model_kwargs["wta_mode_params"]["temperature"] = self.temperature
            self.model.update_wta_mode_params(
                dict_of_params=self.model_kwargs["wta_mode_params"]
            )  # Useful only for awta. this has no effect otherwise.
            self.log_dict({"temperature": self.temperature})

    def training_step(self, batch, batch_idx: int):
        train_loss = self.model.loss(
            **select(self.inputs, batch),
            # future_observed_values=batch["future_observed_values"],
            # future_target_cdf = batch["future_target_cdf"] #little update on the real name neeeded/
        )[
            0
        ]  # cause we return (loss.mean(), likelihoods, distr_args)

        self.log("train_loss", train_loss, on_epoch=True, on_step=False, prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_idx: int):
        val_loss_tuple = self.model.loss(**select(self.inputs, batch))
        val_loss = val_loss_tuple[0].mean()
        target_assignement = val_loss_tuple[-2]
        score_loss = val_loss_tuple[-1].mean()

        num_correct_per_hyp = np.zeros(self.model.n_hypotheses)

        # Log the target assignement
        for hyp in range(self.model.n_hypotheses):
            # Count the number of time target assignement is equal to hyp
            num_correct = (target_assignement == hyp).sum().item()
            self.log(
                f"val_target_assignement_correct_{hyp}",
                num_correct,
                on_epoch=True,
                on_step=False,
                prog_bar=False,
            )
            num_correct_per_hyp[hyp] = num_correct

        # Log the number of active hypotheses
        self.log(
            "val_num_active_hypotheses",
            (num_correct_per_hyp > 0).sum(),
            on_epoch=True,
            on_step=False,
            prog_bar=False,
        )

        self.log(
            "val_score_loss", score_loss, on_epoch=True, on_step=False, prog_bar=False
        )

        self.log("val_loss", val_loss, on_epoch=True, on_step=False, prog_bar=True)

    def configure_optimizers(self):
        """
        Returns the optimizer to use.
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        monitor = "val_loss" if has_validation_loop(self.trainer) else "train_loss"

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer=optimizer,
                    mode="min",
                    factor=0.5,
                    patience=self.patience,
                ),
                "monitor": monitor,
            },
        }
