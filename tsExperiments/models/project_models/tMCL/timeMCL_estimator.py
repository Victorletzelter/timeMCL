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

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Callable, Iterator, Iterable
import rootutils

# Setup project root
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
sys.path.append(os.path.dirname(os.environ["PROJECT_ROOT"]))
from tsExperiments.plottimeMCL import plot_mcl
from gluonts.dataset.repository import get_dataset
from tsExperiments.models.project_models.tMCL.personnalized_evaluator import (
    MultivariateEvaluator,
)
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.core.component import validated
from gluonts.dataset.common import Dataset, DataEntry
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import DataLoader, as_stacked_batches
from tsExperiments.models.project_models.tMCL.personnalized_evaluator import (
    MultivariateEvaluator,
)  # our custom evaluator.
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.itertools import Cyclic, select
from gluonts.model.forecast import Forecast, SampleForecast
from gluonts.model.forecast_generator import ForecastGenerator, to_numpy
from gluonts.time_feature import TimeFeature
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform import (
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpandDimArray,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    TestSplitSampler,
    SetFieldIfNotPresent,
    RenameFields,
    TargetDimIndicator,
    Transformation,
    VstackFeatures,
)

from tsExperiments.utils.utils import lags_for_fourier_time_features_from_frequency
from tsExperiments.models.project_models.tMCL.lighting_grad import timeMCL_lighting
from tsExperiments.Estimator import PyTorchLightningEstimator
from tsExperiments.models.project_models.tMCL.data_preprocessing import (
    fourier_time_features_from_frequency,
)

from utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

OutputTransform = Callable[[DataEntry, np.ndarray], np.ndarray]


def make_predictions(prediction_net, inputs: dict):
    try:
        # Feed inputs as positional arguments for MXNet block predictors
        import mxnet as mx

        if isinstance(prediction_net, mx.gluon.Block):
            return prediction_net(*inputs.values())
    except ImportError:
        pass
    return prediction_net(**inputs)


# to execute in the terminal ))))
@to_numpy.register(torch.Tensor)
def _(x: torch.Tensor) -> np.ndarray:
    return x.cpu().detach().numpy()


class Custom_MCL_SampleForecastGenerator(ForecastGenerator):
    @validated()
    def __init__(self, sample_hyps=True):
        self.sample_hyps = sample_hyps  # This args allows to sample the hyps from the distribution of the scores to approximate the metrics account for the values of the scores.

    def __call__(
        self,
        inference_data_loader: DataLoader,
        prediction_net,
        input_names: List[str],
        output_transform: Optional[OutputTransform],
        num_samples: Optional[int],
        **kwargs,
    ) -> Iterator[Forecast]:

        for batch in inference_data_loader:
            inputs = select(input_names, batch, ignore_missing=True)
            outputs = to_numpy(make_predictions(prediction_net, inputs))
            if output_transform is not None:
                outputs = output_transform(batch, outputs)
            ### Outputs of shape (B,K,T,dim_ts+1)
            ### extraction of the scores here
            B, K, T, dim_ts = outputs.shape
            dim_ts = dim_ts - 1
            scores = outputs[:, :, :, -1]  # (B,K,T) - scores
            scores_mean = np.mean(scores, axis=2)  # (B,K)
            scores_normalized = scores_mean / np.sum(scores_mean, axis=1, keepdims=True)

            if self.sample_hyps is True:
                N = 1000  # the nb of samples we want by resampling with the hyps.
                B, K = scores_normalized.shape  # Batches et hypothèses
                outputs = outputs[:, :, :, :-1]  # (B,K,T,dim_ts)

                sampled_indices = (
                    []
                )  # list of the indices of samples we are going to choose !
                for b in range(B):
                    indices = np.random.choice(
                        K, size=N, p=scores_normalized[b], replace=True
                    )
                    sampled_indices.append(indices)

                sampled_indices = np.array(sampled_indices)  # Shape (B, N)

                new_outputs = np.zeros((B, N, T, dim_ts))

                # extraction of the sample concerned
                for b in range(B):
                    for n in range(N):
                        idx = sampled_indices[b, n]
                        new_outputs[b, n] = outputs[b, idx]
            else:
                scores_normalized_repeated = np.repeat(
                    scores_normalized[:, :, np.newaxis], T, axis=2
                )
                outputs[:, :, :, -1] = scores_normalized_repeated
                new_outputs = outputs

            i = -1
            for i, output in enumerate(new_outputs):
                yield SampleForecast(
                    output,
                    start_date=batch[FieldName.FORECAST_START][i],
                    item_id=(
                        batch[FieldName.ITEM_ID][i]
                        if FieldName.ITEM_ID in batch
                        else None
                    ),
                    info=batch["info"][i] if "info" in batch else None,
                )
            assert i + 1 == len(batch[FieldName.FORECAST_START])


class timeMCL_estimator(PyTorchLightningEstimator):

    def __init__(
        self,
        freq: str,
        prediction_length: int,
        target_dim: int,
        num_hypotheses: int,
        mcl_hidden_dim: int,
        context_length: int,
        num_layers: int,
        loss_type: str,  # training arg
        num_cells: int,
        cell_type: str,
        num_parallel_samples: int,
        dropout_rate: float,
        embedding_dimension: int,
        conditioning_length: int,
        residual_layers: int,
        residual_channels: int,
        dilation_cycle_length: int,
        scaling: bool,
        pick_incomplete: bool,
        time_features: Optional[List[TimeFeature]],
        mcl_loss_type: str,
        num_feat_dynamic_real: int,
        score_loss_weight: float,
        wta_mode: str,
        wta_mode_params: dict,
        trainer_kwargs: dict,
        data_kwargs: dict,
        optim_kwargs: dict,
        sample_hyps: bool,
        single_linear_layer: bool,
        backbone_deleted: bool,
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

        log.info(f"kwargs (not used): {kwargs}")

        self.backbone_deleted = backbone_deleted
        self.optim_kwargs = optim_kwargs
        self.wta_mode = wta_mode
        self.wta_mode_params = wta_mode_params
        self.sample_hyps = sample_hyps
        self.scaler_type = scaler_type
        self.div_by_std = div_by_std
        self.minimum_std = minimum_std
        self.minimum_std_cst = minimum_std_cst
        self.default_scale = default_scale
        self.default_scale_cst = default_scale_cst
        self.add_minimum_std = add_minimum_std
        self.single_linear_layer = single_linear_layer
        self.trainer_kwargs = trainer_kwargs
        self.max_epochs = trainer_kwargs["max_epochs"]
        self.gradient_clip_val = trainer_kwargs["gradient_clip_val"]

        self.batch_size = data_kwargs["batch_size"]
        self.num_batches_per_epoch = data_kwargs["num_batches_per_epoch"]
        self.num_batches_val_per_epoch = data_kwargs["num_batches_val_per_epoch"]
        self.shuffle_buffer_length = data_kwargs["shuffle_buffer_length"]

        super().__init__(trainer_kwargs=trainer_kwargs)

        self.mcl_hidden_dim = mcl_hidden_dim
        self.score_loss_weight = score_loss_weight
        self.freq = freq
        self.mcl_loss_type = mcl_loss_type
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
        self.n_hypotheses = num_hypotheses
        self.shuffle_buffer_length = None
        self.prediction_length = prediction_length
        self.target_dim = target_dim
        self.num_layers = num_layers
        self.num_cells = num_cells
        self.cell_type = cell_type
        self.num_parallel_samples = num_parallel_samples
        self.dropout_rate = dropout_rate
        self.embedding_dimension = embedding_dimension
        self.num_feat_dynamic_real = num_feat_dynamic_real

        self.conditioning_length = conditioning_length
        self.loss_type = loss_type
        self.residual_layers = residual_layers
        self.residual_channels = residual_channels
        self.dilation_cycle_length = dilation_cycle_length
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
            "validation": self.val_sampler,  # self.validation_sampler,
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
        # initialisation of the neural network we are going to take for the training.
        return timeMCL_lighting(
            model_kwargs={
                "mcl_n_hypotheses": self.n_hypotheses,
                "mcl_hidden_dim": self.mcl_hidden_dim,
                "num_layers": self.num_layers,
                "num_cells": self.num_cells,
                "cell_type": self.cell_type,
                "context_length": self.context_length,
                "prediction_length": self.prediction_length,
                "dropout_rate": self.dropout_rate,
                "target_dim": self.target_dim,
                "lags_seq": self.lags_seq,
                "mcl_loss_type": self.mcl_loss_type,
                "conditioning_length": self.conditioning_length,
                "scaling": self.scaling,
                "num_feat_dynamic_real": self.num_feat_dynamic_real,
                "score_loss_weight": self.score_loss_weight,
                "wta_mode": self.wta_mode,
                "wta_mode_params": self.wta_mode_params,
                "embedding_dimension": self.embedding_dimension,
                "single_linear_layer": self.single_linear_layer,
                "backbone_deleted": self.backbone_deleted,
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
            forecast_generator=Custom_MCL_SampleForecastGenerator(
                sample_hyps=self.sample_hyps
            ),
        )


def distorsion(target: np.ndarray, forecast: np.ndarray) -> float:
    r"""
    .. math::

        mse = mean((Y - \hat{Y})^2)

    See [HA21]_ for more details.
    """
    # forecast.samples.shape -> (num_samples,pred_length). we have all our pred for a given target. We want to take the best..
    # target.data -> (pred_lentgh,)

    rmse_values = np.mean(
        (forecast.samples - target.data) ** 2, axis=1
    )  # to add mse after.
    best_mse = np.min(rmse_values)

    return best_mse  # np.mean(np.square(target - mean_fcst)) #we return the best one.


def personnalized_mse(target: np.ndarray, forecast: np.ndarray) -> float:
    r"""
    .. math::

        mse = mean((Y - \hat{Y})^2)

    See [HA21]_ for more details.
    """
    return np.mean(np.square(target - forecast))
