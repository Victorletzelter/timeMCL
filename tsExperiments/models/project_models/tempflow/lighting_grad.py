# The lighting part of the tempflow we personnalized to be compatible with Lightning.

import lightning.pytorch as pl
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tsExperiments.models.project_models.tempflow.tempflow_network import (
    TempFlowNetwork,
)
from gluonts.core.component import validated
from gluonts.itertools import select
from gluonts.torch.model.lightning_util import has_validation_loop

from typing import Optional, Dict, Any


class TempFlowLighting(pl.LightningModule):

    def __init__(
        self,
        model_kwargs: dict,
        optim_kwargs: dict,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = TempFlowNetwork(**model_kwargs)
        self.lr = optim_kwargs["lr"]
        self.weight_decay = optim_kwargs["weight_decay"]
        self.patience = optim_kwargs["patience"]
        self.inputs = self.model.describe_inputs()
        self.example_input_array = self.inputs.zeros()  # almost always the same

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx: int):
        train_loss = self.model.loss(
            **select(self.inputs, batch),
        )[0]

        self.log("train_loss", train_loss, on_epoch=True, on_step=False, prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_idx: int):
        val_loss = self.model.loss(**select(self.inputs, batch))[0].mean()

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
