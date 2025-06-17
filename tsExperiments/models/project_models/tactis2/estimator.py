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

# This code was adapted from https://github.com/ServiceNow/TACTiS/blob/tactis-2/tactis/gluon/estimator.py

# estimator tactis
from typing import Any, Dict, Optional, Iterable
import os, sys
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
sys.path.append(os.path.join(os.environ["PROJECT_ROOT"], ".."))
from gluonts.dataset.loader import as_stacked_batches
import numpy as np
import torch
import torch.nn as nn
from gluonts.dataset.field_names import FieldName
from gluonts.env import env
from gluonts.dataset.common import Dataset
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform import Transformation
from gluonts.transform import (
    AddObservedValuesIndicator,
    CDFtoGaussianTransform,
    Chain,
    InstanceSampler,
    InstanceSplitter,
    RenameFields,
    ValidationSplitSampler,
    TestSplitSampler,
)
from gluonts.itertools import Cyclic
from tsExperiments.models.project_models.tactis2.lighting import TatcisLighting
from utils import (
    RankedLogger,
)
from tsExperiments.Estimator import PyTorchLightningEstimator
from tsExperiments.Estimator.pytorchLightingEstimator import TrainOutput
from gluonts.itertools import Cached
import lightning.pytorch as pl
import logging

logger = logging.getLogger(__name__)
import numpy as np

if not hasattr(np, "bool"):
    np.bool = bool
from utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class SingleInstanceSampler(InstanceSampler):
    """
    Randomly pick a single valid window in the given time series.
    This fix the bias in ExpectedNumInstanceSampler which leads to varying sampling frequency
    of time series of unequal length, not only based on their length, but when they were sampled.
    """

    """End index of the history"""

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        a, b = self._get_bounds(ts)
        window_size = b - a + 1

        if window_size <= 0:
            return np.array([], dtype=int)

        indices = np.random.randint(window_size, size=1)

        return indices + a


class TACTiSEstimator(PyTorchLightningEstimator):
    """
    The compatibility layer between TACTiS and GluonTS / PyTorchTS.
    """

    def __init__(
        self,
        model_parameters: Dict[str, Any],
        target_dim: int,
        context_length: int,
        prediction_length: int,
        freq: str,
        nb_epoch_phase_1: int,
        cdf_normalization: bool = False,
        num_parallel_samples: int = 1,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        data_kwargs: Optional[Dict[str, Any]] = None,
        optim_kwargs: Optional[Dict[str, Any]] = None,
        num_hypotheses: int = 1,
        choosing_the_best_model_phase_1: bool = True,
        beta: float = 1.0,
        **kwargs,
    ):
        """
        A PytorchTS wrapper for TACTiS

        Parameters:
        -----------
        model_parameters: Dict[str, Any]
            The parameters that will be sent to the TACTiS model.
        target_dim: int
            The number of series in the multivariate data.
        context_length: int
            How many time steps will be sent to the model as observed.
        prediction_length: int
            How many time steps will be sent to the model as unobserved, to be predicted.
        freq: str
            The frequency of the series to be forecasted.
        trainer: Trainer
            A Pytorch-TS trainer object
        cdf_normalization: bool, default to False
            If set to True, then the data will be transformed using an estimated CDF from the
            historical data points, followed by the inverse CDF of a Normal(0, 1) distribution.
            Should not be used concurrently with the standardization normalization option in TACTiS.
        num_parallel_samples: int, default to 1
            How many samples to draw at the same time during forecast.
        choosing_the_best_model_phase_1
            By defalut we choose the best model after the phase 1.
        """
        super().__init__(trainer_kwargs=trainer_kwargs, **kwargs)

        self.model_parameters = model_parameters
        self.nb_epochs_phase_1 = nb_epoch_phase_1

        self.target_dim = target_dim
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.freq = freq

        self.cdf_normalization = cdf_normalization
        self.num_parallel_samples = num_parallel_samples

        self.trainer_kwargs = trainer_kwargs
        self.optim_kwargs = optim_kwargs
        self.max_epochs = trainer_kwargs["max_epochs"]
        self.gradient_clip_val = trainer_kwargs["gradient_clip_val"]

        self.batch_size = data_kwargs["batch_size"]
        self.num_batches_per_epoch = data_kwargs["num_batches_per_epoch"]
        self.num_batches_val_per_epoch = data_kwargs["num_batches_val_per_epoch"]
        self.shuffle_buffer_length = data_kwargs["shuffle_buffer_length"]

        self.inputs_names = ["past_target_norm", "future_target_norm"]
        self.choosing_the_best_model_phase_1 = choosing_the_best_model_phase_1

    def _create_instance_splitter(self, module, mode: str) -> Transformation:
        """
        Create and return the instance splitter needed for training, validation or testing.

        Parameters:
        -----------
        mode: str, "training", "validation", or "test"
            Whether to split the data for training, validation, or test (forecast)

        Returns
        -------
        Transformation
            The InstanceSplitter that will be applied entry-wise to datasets,
            at training, validation and inference time based on mode.
        """
        assert mode in ["training", "validation", "test"]

        if mode == "training":
            instance_sampler = SingleInstanceSampler(
                min_past=self.context_length,
                min_future=self.prediction_length,
            )
        elif mode == "validation":
            instance_sampler = ValidationSplitSampler(
                min_past=self.context_length,
                min_future=self.prediction_length,
            )
        elif mode == "test":
            # This splitter takes the last valid window from each multivariate series,
            # so any multi-window split must be done in the data definition.
            instance_sampler = TestSplitSampler()

        if self.cdf_normalization:
            normalize_transform = CDFtoGaussianTransform(
                cdf_suffix="_norm",
                target_field=FieldName.TARGET,
                target_dim=self.target_dim,
                max_context_length=self.context_length,
                observed_values_field=FieldName.OBSERVED_VALUES,
            )
        else:
            normalize_transform = RenameFields(
                {
                    f"past_{FieldName.TARGET}": f"past_{FieldName.TARGET}_norm",
                    f"future_{FieldName.TARGET}": f"future_{FieldName.TARGET}_norm",
                }
            )

        instance_sampler = (
            InstanceSplitter(
                target_field=FieldName.TARGET,
                is_pad_field=FieldName.IS_PAD,
                start_field=FieldName.START,
                forecast_start_field=FieldName.FORECAST_START,
                instance_sampler=instance_sampler,
                past_length=self.context_length,
                future_length=self.prediction_length,
                time_series_fields=[FieldName.OBSERVED_VALUES],
            )
            + normalize_transform
        )

        return instance_sampler

    def create_transformation(self) -> Transformation:
        """
        Add a transformation that replaces NaN in the input data with zeros,
        and mention whether the data was a NaN or not in another field.

        Returns:
        --------
        transformation: Transformation
            The chain of transformations defined for TACTiS.
        """
        return Chain(
            [
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                ),
            ]
        )

    def create_training_data_loader(self, data: Dataset, module, **kwargs) -> Iterable:
        data = Cyclic(data).stream()
        instances = self._create_instance_splitter(module, "training").apply(
            data, is_train=True
        )
        return as_stacked_batches(
            instances,
            batch_size=self.batch_size,
            shuffle_buffer_length=self.shuffle_buffer_length,
            field_names=self.inputs_names,
            output_type=torch.tensor,
            num_batches_per_epoch=self.num_batches_per_epoch,
        )

    def create_validation_data_loader(
        self, data: Dataset, module, **kwargs
    ) -> Iterable:
        data = Cyclic(data).stream()
        instances = self._create_instance_splitter(module, mode="validation").apply(
            data, is_train=True
        )

        return as_stacked_batches(
            instances,
            batch_size=self.batch_size,
            field_names=self.inputs_names,
            output_type=torch.tensor,
            num_batches_per_epoch=self.num_batches_val_per_epoch,
        )

    def create_lightning_module(self) -> nn.Module:
        return TatcisLighting(
            model_kwargs={
                "num_series": self.target_dim,
                "model_parameters": self.model_parameters,
                "prediction_length": self.prediction_length,
                "num_parallel_samples": self.num_parallel_samples,
                "context_length": self.context_length,
            },
            optim_kwargs=self.optim_kwargs,
            num_batch_epoch_phase_1=self.num_batches_per_epoch * self.nb_epochs_phase_1,
        )

    def create_predictor(
        self, transformation: Transformation, module
    ) -> PyTorchPredictor:
        prediction_splitter = self._create_instance_splitter(module, "test")
        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            input_names=self.inputs_names,
            prediction_net=module,
            batch_size=self.batch_size,
            prediction_length=self.prediction_length,
            device="auto",
        )

    def train_model(
        self,
        training_data: Dataset,
        validation_data: Optional[Dataset] = None,
        from_predictor: Optional[PyTorchPredictor] = None,
        shuffle_buffer_length: Optional[int] = None,
        cache_data: bool = False,
        ckpt_path: Optional[str] = None,
        **kwargs,
    ) -> TrainOutput:
        # Create the transformation for the data.
        transformation = self.create_transformation()

        # Apply the transformation and create training data loader.
        with env._let(max_idle_transforms=max(len(training_data), 100)):
            transformed_training_data: Dataset = transformation.apply(
                training_data, is_train=True
            )
            if cache_data:
                transformed_training_data = Cached(transformed_training_data)
            training_network = self.create_lightning_module()
            training_data_loader = self.create_training_data_loader(
                transformed_training_data,
                training_network,
                shuffle_buffer_length=shuffle_buffer_length,
            )

        # Create validation data loader if validation_data is provided.
        validation_data_loader = None
        if validation_data is not None:
            with env._let(max_idle_transforms=max(len(validation_data), 100)):
                transformed_validation_data: Dataset = transformation.apply(
                    validation_data, is_train=True
                )
                if cache_data:
                    transformed_validation_data = Cached(transformed_validation_data)
                validation_data_loader = self.create_validation_data_loader(
                    transformed_validation_data,
                    training_network,
                )

        # If a predictor is provided, load its state.
        if from_predictor is not None:
            training_network.load_state_dict(from_predictor.network.state_dict())

        # Common trainer parameters.
        custom_callbacks = self.trainer_kwargs.pop("callbacks", [])
        logger_param = self.trainer_kwargs.pop("logger", None)
        trainer_kwargs_copy = self.trainer_kwargs.copy()

        #################################
        # Phase 1 Training
        #################################
        monitor_phase_1 = (
            "train_loss_phase_1" if validation_data is None else "val_loss"
        )
        checkpoint_phase1 = pl.callbacks.ModelCheckpoint(
            monitor=monitor_phase_1,
            mode="min",
            verbose=True,
            filename="phase1-{epoch:02d}-{checkpoint_phase1:.2f}",
            dirpath="checkpoints_1/",
        )
        trainer_phase1 = pl.Trainer(
            max_epochs=self.nb_epochs_phase_1,  # Number of epochs for phase 1
            callbacks=[checkpoint_phase1],  # + custom_callbacks,
            accelerator="auto",
            logger=logger_param,
            gradient_clip_val=self.gradient_clip_val,
        )
        trainer_phase1.fit(
            model=training_network,
            train_dataloaders=training_data_loader,
            val_dataloaders=validation_data_loader,
            ckpt_path=ckpt_path,
        )

        # Load the best model from Phase 1. (is we decide to chose it)
        if (
            checkpoint_phase1.best_model_path != ""
            and self.choosing_the_best_model_phase_1
        ):
            logger.info(
                f"Loading best model from Phase 1: {checkpoint_phase1.best_model_path}"
            )
            mutable_lightning_model = training_network.__class__.load_from_checkpoint(
                checkpoint_phase1.best_model_path
            )
        else:
            mutable_lightning_model = training_network

        #################################
        # Transition to Phase 2
        #################################
        log.info("======> SWITCH PHASE 2 ")
        # Switch the model to phase 2 and update its internal state.
        mutable_lightning_model.switch_to_stage_2(mutable_lightning_model.model, "adam")
        ### In the class now.
        ##########################################
        # mutable_lightning_model.model = net
        # mutable_lightning_model.phase = 2 # starting phase 2
        # mutable_lightning_model.new_optimiseur = optimiseur
        ##########################################
        # we change the model of the LightingModule
        #################################
        # Phase 2 Training
        #################################
        # Make sure you have defined self.num_epoch_phase_2 (e.g., in __init__).
        monitor_phase_2 = (
            "train_loss_phase_2" if validation_data is None else "val_loss"
        )
        checkpoint_phase2 = pl.callbacks.ModelCheckpoint(
            monitor=monitor_phase_2,
            mode="min",
            verbose=True,
            filename="phase2-{epoch:02d}-{checkpoint_phase2:.2f}",
            dirpath="checkpoints_2/",
        )
        trainer_phase2 = pl.Trainer(
            max_epochs=self.max_epochs
            - self.nb_epochs_phase_1,  # Number of epochs for phase 2
            callbacks=[checkpoint_phase2],  # + custom_callbacks,
            accelerator="auto",
            logger=logger_param,
            gradient_clip_val=self.gradient_clip_val,
        )
        trainer_phase2.fit(
            model=mutable_lightning_model,
            train_dataloaders=training_data_loader,
            val_dataloaders=validation_data_loader,
            ckpt_path=ckpt_path,
        )

        # Load the best model from Phase 2.
        if checkpoint_phase2.best_model_path != "":
            logger.info(
                f"Loading best model from Phase 2: {checkpoint_phase2.best_model_path}"
            )
            checkpoint_data = torch.load(checkpoint_phase2.best_model_path)
            mutable_lightning_model.load_state_dict(checkpoint_data["state_dict"])

        return TrainOutput(
            transformation=transformation,
            trained_net=mutable_lightning_model,
            trainer=trainer_phase2,
            predictor=self.create_predictor(transformation, mutable_lightning_model),
        )
