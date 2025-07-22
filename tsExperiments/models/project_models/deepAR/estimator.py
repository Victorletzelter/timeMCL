# This code was adapted from https://github.com/zalandoresearch/pytorch-ts/blob/master/pts/model/deepvar/deepvar_estimator.py
# under MIT License

import rootutils
import os, sys

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
sys.path.append(os.path.dirname(os.environ["PROJECT_ROOT"]))

from gluonts.evaluation.backtest import make_evaluation_predictions
import matplotlib.pyplot as plt
from typing import List, Optional, Iterable, Dict, Any
from typing import List, Optional, Callable
from gluonts.dataset.common import Dataset
import numpy as np
from gluonts.evaluation import MultivariateEvaluator
import torch
from gluonts.core.component import validated
from gluonts.dataset.field_names import FieldName
from gluonts.time_feature import TimeFeature
from gluonts.torch.distributions import DistributionOutput
from gluonts.torch.util import copy_parameters
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.model.predictor import Predictor
from gluonts.transform import (
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    CDFtoGaussianTransform,
    Chain,
    ExpandDimArray,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    ValidationSplitSampler,
    TestSplitSampler,
    SetFieldIfNotPresent,
    RenameFields,
    SetField,
    TargetDimIndicator,
    Transformation,
    VstackFeatures,
    RemoveFields,
    AddAgeFeature,
    cdf_to_gaussian_forward_transform,
)
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.repository import get_dataset
import inspect
import torch.nn as nn
from tsExperiments.models.project_models.deepAR.lighting_grad import LightingDeepVAR

from tsExperiments.Estimator import PyTorchLightningEstimator
from distribution_output import LowRankMultivariateNormalOutput
from tsExperiments.utils.utils import fourier_time_features_from_frequency, lags_for_fourier_time_features_from_frequency
from gluonts.time_feature import TimeFeature
from gluonts.dataset.loader import as_stacked_batches
from gluonts.dataset.common import Dataset
from gluonts.itertools import Cyclic
from utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class deepVAREstimator(PyTorchLightningEstimator):

    def __init__(
        self,
        freq: str,
        prediction_length: int,
        target_dim: int,
        context_length: int,
        num_layers: int,
        num_cells: int,
        cell_type: str,
        num_parallel_samples: int,
        dropout_rate: float,
        embedding_dimension: int,
        dist_type: str,
        dist_params: Optional[Dict[str, Any]],
        scaling: bool,
        pick_incomplete: bool,
        time_features: Optional[List[TimeFeature]],
        conditioning_length: int,
        trainer_kwargs: Dict[str, Any],
        data_kwargs: Dict[str, Any],
        optim_kwargs: Dict[str, Any],
        num_feat_dynamic_real: int,
        scaler_type: str,
        div_by_std: bool,
        minimum_std: float,
        minimum_std_cst: float,
        default_scale: bool,
        default_scale_cst: bool,
        add_minimum_std: bool,
        beta: float = 1.0,
        **kwargs,
    ):

        log.info(f"kwargs passed (not used): {kwargs}")

        self.trainer_kwargs = trainer_kwargs
        self.optim_kwargs = optim_kwargs
        self.max_epochs = trainer_kwargs["max_epochs"]
        self.gradient_clip_val = trainer_kwargs["gradient_clip_val"]
        self.batch_size = data_kwargs["batch_size"]
        self.num_batches_per_epoch = data_kwargs["num_batches_per_epoch"]
        self.num_batches_val_per_epoch = data_kwargs["num_batches_val_per_epoch"]
        self.shuffle_buffer_length = data_kwargs["shuffle_buffer_length"]

        super().__init__(trainer_kwargs=trainer_kwargs)

        self.dist_type = dist_type
        self.dist_params = dist_params
        self.num_parallel_samples = num_parallel_samples

        self.scaler_type = scaler_type
        self.div_by_std = div_by_std
        self.minimum_std = minimum_std
        self.minimum_std_cst = minimum_std_cst
        self.default_scale = default_scale
        self.default_scale_cst = default_scale_cst
        self.add_minimum_std = add_minimum_std

        self.freq = freq
        self.context_length = (
            context_length if context_length is not None else prediction_length
        )

        self.inputs_names = [
            "target_dimension_indicator",
            "past_target_cdf",
            "past_observed_values",
            "past_is_pad",
            "future_time_feat",
            "past_time_feat",
            "future_target_cdf",
            "future_observed_values",
        ]

        self.prediction_length = prediction_length
        self.target_dim = target_dim
        self.num_layers = num_layers
        self.num_cells = num_cells
        self.cell_type = cell_type
        self.num_parallel_samples = num_parallel_samples
        self.dropout_rate = dropout_rate
        self.embedding_dimension = embedding_dimension

        self.conditioning_length = conditioning_length
        self.num_feat_dynamic_real = num_feat_dynamic_real

        self.lags_seq = lags_for_fourier_time_features_from_frequency(freq_str=freq)

        self.time_features = (
            time_features
            if time_features is not None
            else fourier_time_features_from_frequency(self.freq)
        )

        self.history_length = self.context_length + max(self.lags_seq)
        self.pick_incomplete = pick_incomplete
        self.scaling = scaling

        self.train_sampler = ExpectedNumInstanceSampler(
            num_instances=1.0,
            min_past=0 if pick_incomplete else self.history_length,
            min_future=prediction_length,
        )

        self.val_sampler = ExpectedNumInstanceSampler(
            num_instances=1.0,
            min_past=0 if pick_incomplete else self.history_length,
            min_future=prediction_length,
        )

    def create_transformation(self) -> Transformation:
        return Chain(
            [
                AsNumpyArray(
                    field=FieldName.TARGET,
                    expected_ndim=2,
                ),
                # maps the target to (1, T)
                # if the target data is uni dimensional
                ExpandDimArray(
                    field=FieldName.TARGET,
                    axis=None,
                ),
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                ),
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=self.time_features,
                    pred_length=self.prediction_length,
                ),
                VstackFeatures(
                    output_field=FieldName.FEAT_TIME,
                    input_fields=[FieldName.FEAT_TIME],
                ),
                SetFieldIfNotPresent(field=FieldName.FEAT_STATIC_CAT, value=[0]),
                TargetDimIndicator(
                    field_name="target_dimension_indicator",
                    target_field=FieldName.TARGET,
                ),
                AsNumpyArray(field=FieldName.FEAT_STATIC_CAT, expected_ndim=1),
            ]
        )

    def _create_instance_splitter(self, module, mode: str):
        assert mode in ["training", "validation", "test"]

        instance_sampler = {
            "training": self.train_sampler,
            "validation": self.val_sampler,
            "test": TestSplitSampler(),
        }[mode]

        return InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=self.history_length,
            future_length=self.prediction_length,
            time_series_fields=[
                FieldName.FEAT_TIME,
                FieldName.OBSERVED_VALUES,
            ],
        ) + (
            RenameFields(
                {
                    f"past_{FieldName.TARGET}": f"past_{FieldName.TARGET}_cdf",
                    f"future_{FieldName.TARGET}": f"future_{FieldName.TARGET}_cdf",
                }
            )
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
            shuffle_buffer_length=self.shuffle_buffer_length,
            output_type=torch.tensor,
            num_batches_per_epoch=self.num_batches_val_per_epoch,
        )

    # simple/validated
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

    def create_lightning_module(self) -> nn.Module:
        return LightingDeepVAR(
            model_kwargs={
                "num_layers": self.num_layers,
                "num_cells": self.num_cells,
                "cell_type": self.cell_type,
                "context_length": self.context_length,
                "prediction_length": self.prediction_length,
                "dist_type": self.dist_type,
                "dist_params": self.dist_params,
                "dropout_rate": self.dropout_rate,
                "lags_seq": self.lags_seq,
                "target_dim": self.target_dim,
                "embedding_dimension": self.embedding_dimension,
                "num_feat_dynamic_real": self.num_feat_dynamic_real,
                "scaling": self.scaling,
                "num_parallel_samples": self.num_parallel_samples,
                "scaler_type": self.scaler_type,
                "div_by_std": self.div_by_std,
                "minimum_std": self.minimum_std,
                "minimum_std_cst": self.minimum_std_cst,
                "default_scale": self.default_scale,
                "default_scale_cst": self.default_scale_cst,
                "add_minimum_std": self.add_minimum_std,
            },
            optim_kwargs=self.optim_kwargs,
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
