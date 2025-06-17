# the lighting part of the tactis we personnalized (here).
# lighting tactis

import lightning.pytorch as pl
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tsExperiments.models.project_models.tactis2.network import TactisNetwork
from gluonts.itertools import select
from gluonts.torch.model.lightning_util import has_validation_loop
from torch.optim import Adam, RMSprop


class TatcisLighting(pl.LightningModule):

    def __init__(
        self,
        model_kwargs: dict,
        optim_kwargs: dict,
        num_batch_epoch_phase_1: int,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = TactisNetwork(**model_kwargs)
        self.lr = optim_kwargs["lr"]
        self.weight_decay = optim_kwargs["weight_decay"]
        self.patience = optim_kwargs["patience"]
        self.inputs = self.model.describe_inputs()
        self.example_input_array = self.inputs.zeros()
        self.num_batches_phase_1 = num_batch_epoch_phase_1
        self.has_switched_to_phase_2 = False
        self.current_batch_num = 0
        self.phase = 1

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @staticmethod
    def disable_grads(net, disable_grads):
        for j, p in enumerate(net.model.named_parameters()):
            for param in disable_grads:
                if param in p[0]:
                    p[1].requires_grad = False

    def initialize_stage_1(self, net, optimizer_name, ckpt=None):
        net.model.set_stage(1)
        if optimizer_name == "rmsprop":
            optim = RMSprop(
                net.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif optimizer_name == "adam":
            optim = Adam(
                net.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

        return net, optim

    def switch_to_stage_2(self, net, optimizer_name, ckpt=None):
        disable_grads = []

        net.model.set_stage(2)
        net.model.initialize_stage2()
        net.to(self.device)

        params_input_encoder_flow = [
            "flow_series_encoder",
            "flow_time_encoding",
            "flow_input_encoder",
        ]
        params_encoder_flow = ["flow_encoder"]
        params_decoder_flow = ["decoder.marginal"]

        parameter_names_to_optimize = [
            "copula_series_encoder",
            "copula_time_encoding",
            "copula_input_encoder",
            "copula_encoder",
            "decoder.copula",
        ]
        params_to_optimize_in_stage2 = []
        for name, param in net.named_parameters():
            if any(pname in name for pname in parameter_names_to_optimize):
                params_to_optimize_in_stage2.append(param)

        if optimizer_name == "rmsprop":
            optim = RMSprop(
                params_to_optimize_in_stage2,
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif optimizer_name == "adam":
            optim = Adam(
                params_to_optimize_in_stage2,
                lr=self.lr,
                weight_decay=self.weight_decay,
            )

        disable_grads.extend(params_decoder_flow)
        disable_grads.extend(params_input_encoder_flow)
        disable_grads.extend(params_encoder_flow)
        TatcisLighting.disable_grads(net, disable_grads)

        self.model = net
        self.phase = 2  # starting phase 2
        self.new_optimiseur = optim
        self.current_batch_num = 0

    def training_step(self, batch, batch_idx: int):

        if self.phase == 2 and self.current_batch_num == 0:
            self.trainer.optimizers = [self.new_optimiseur]

        _ = self.model.loss(**select(self.inputs, batch))

        if self.phase == 1:
            loss = -self.model.model.marginal_logdet.mean()
            self.log("train_loss_phase_1", loss, on_epoch=True, prog_bar=True)

        else:
            loss = self.model.model.copula_loss.mean()
            self.log("train_loss_phase_2", loss, on_epoch=True, prog_bar=True)

        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.current_batch_num += 1
        return loss

    def validation_step(self, batch, batch_idx: int):

        _ = self.model.loss(**select(self.inputs, batch))
        marginal_logdet, copula_loss = (
            self.model.model.marginal_logdet,
            self.model.model.copula_loss,
        )

        if self.phase == 1:
            val_loss = -marginal_logdet.mean()
        else:
            val_loss = copula_loss.mean()

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
