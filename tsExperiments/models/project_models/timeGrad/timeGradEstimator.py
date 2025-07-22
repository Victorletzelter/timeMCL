# This code was adapted from https://github.com/zalandoresearch/pytorch-ts/blob/master/pts/model/time_grad/time_grad_estimator.py
# under MIT License

import rootutils
import os, sys

#
# Setup project root
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
from gluonts.dataset.field_names import FieldName
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
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.repository import get_dataset
import torch.nn as nn
from tsExperiments.models.project_models.timeGrad.lighting_grad import timeGrad
from tsExperiments.utils.utils import fourier_time_features_from_frequency, lags_for_fourier_time_features_from_frequency
from tsExperiments.Estimator import PyTorchLightningEstimator
from gluonts.time_feature import TimeFeature
from gluonts.dataset.loader import as_stacked_batches
from gluonts.dataset.common import Dataset
from gluonts.itertools import Cyclic

from utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class TimEstimatorGrad(PyTorchLightningEstimator):

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
        conditioning_length: int,
        diff_steps: int,
        loss_type: str,
        beta_end: float,
        beta_schedule: str,
        residual_layers: int,
        residual_channels: int,
        dilation_cycle_length: int,
        scaling: bool,
        pick_incomplete: bool,
        time_features: Optional[List[TimeFeature]],
        shuffle_buffer_length: Optional[int],
        trainer_kwargs: Optional[Dict[str, Any]],
        data_kwargs: Optional[Dict[str, Any]],
        optim_kwargs: Optional[Dict[str, Any]],
        num_feat_dynamic_real: int,
        scaler_type: str,
        div_by_std: bool,
        minimum_std: float,
        minimum_std_cst: float,
        default_scale: bool,
        default_scale_cst: bool,
        add_minimum_std: bool,
        **kwargs,
    ):

        log.info(f"kwargs (not used): {kwargs}")

        self.trainer_kwargs = trainer_kwargs
        self.optim_kwargs = optim_kwargs
        self.max_epochs = trainer_kwargs["max_epochs"]
        self.gradient_clip_val = trainer_kwargs["gradient_clip_val"]

        self.batch_size = data_kwargs["batch_size"]
        self.num_batches_per_epoch = data_kwargs["num_batches_per_epoch"]
        self.num_batches_val_per_epoch = data_kwargs["num_batches_val_per_epoch"]
        self.shuffle_buffer_length = data_kwargs["shuffle_buffer_length"]

        self.scaler_type = scaler_type
        self.div_by_std = div_by_std
        self.minimum_std = minimum_std
        self.minimum_std_cst = minimum_std_cst
        self.default_scale = default_scale
        self.default_scale_cst = default_scale_cst
        self.add_minimum_std = add_minimum_std

        super().__init__(trainer_kwargs=trainer_kwargs)

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

        self.shuffle_buffer_length = shuffle_buffer_length
        self.prediction_length = prediction_length
        self.target_dim = target_dim
        self.num_layers = num_layers
        self.num_cells = num_cells
        self.cell_type = cell_type
        self.num_parallel_samples = num_parallel_samples
        self.dropout_rate = dropout_rate
        self.embedding_dimension = embedding_dimension

        self.conditioning_length = conditioning_length
        self.diff_steps = diff_steps
        self.loss_type = loss_type
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.residual_layers = residual_layers
        self.residual_channels = residual_channels
        self.dilation_cycle_length = dilation_cycle_length
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
        # initialisation of the neural network we are going to take for the training.
        return timeGrad(
            model_kwargs={
                "num_parallel_samples": self.num_parallel_samples,
                "num_layers": self.num_layers,
                "num_cells": self.num_cells,
                "cell_type": self.cell_type,
                "context_length": self.context_length,
                "prediction_length": self.prediction_length,
                "dropout_rate": self.dropout_rate,
                "target_dim": self.target_dim,
                "num_feat_dynamic_real": self.num_feat_dynamic_real,
                "lags_seq": self.lags_seq,
                "conditioning_length": self.conditioning_length,
                "diff_steps": self.diff_steps,
                "loss_type": self.loss_type,
                "beta_end": self.beta_end,
                "beta_schedule": self.beta_schedule,
                "residual_layers": self.residual_layers,
                "residual_channels": self.residual_channels,
                "dilation_cycle_length": self.dilation_cycle_length,
                "embedding_dimension": self.embedding_dimension,
                "scaling": self.scaling,
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


if __name__ == "__main__":

    dataset_name = "solar_nips"  # OK
    # dataset_name = "solar_nips" # OK
    # dataset_name = "electricity_nips" # OK
    # dataset_name = "traffic_nips" # OK
    # dataset_name = "taxi_30min" # OK, but long to run.
    # dataset_name = "wiki-rolling_nips" # OK.

    dataset = get_dataset(dataset_name, regenerate=False)

    datasets_params = {
        "exchange_rate_nips": {"num_feat_dynamic_real": 4},  # 2 ?
        "solar_nips": {"num_feat_dynamic_real": 4},  # ok
        "electricity_nips": {"num_feat_dynamic_real": 4},  # ok
        "traffic_nips": {"num_feat_dynamic_real": 4},  # ok
        "taxi_30min": {"num_feat_dynamic_real": 6},  # ok
        "wiki-rolling_nips": {"num_feat_dynamic_real": 2},  # ok
    }

    train_grouper = MultivariateGrouper(
        max_target_dim=min(2000, int(dataset.metadata.feat_static_cat[0].cardinality))
    )

    test_grouper = MultivariateGrouper(
        num_test_dates=int(len(dataset.test) / len(dataset.train)),
        max_target_dim=min(2000, int(dataset.metadata.feat_static_cat[0].cardinality)),
    )

    print(
        f"Using {int(len(dataset.test)/len(dataset.train))} rolling dates for testing"
    )

    dataset_train = train_grouper(dataset.train)
    dataset_test = test_grouper(dataset.test)

    estimatorTimeGrad = TimEstimatorGrad(
        num_layers=2,
        num_cells=40,
        cell_type="LSTM",
        num_parallel_samples=100,
        dropout_rate=0.1,
        embedding_dimension=0,
        conditioning_length=100,
        diff_steps=2,
        loss_type="l2",
        beta_end=0.1,
        beta_schedule="linear",
        residual_layers=8,
        residual_channels=8,
        dilation_cycle_length=2,
        scaling=True,
        pick_incomplete=False,
        time_features=None,
        # num_feat_dynamic_real=2,
        num_feat_dynamic_real=datasets_params[dataset_name]["num_feat_dynamic_real"],
        scaler_type="mean_std",
        div_by_std=True,
        minimum_std=1e-3,
        shuffle_buffer_length=10,
        minimum_std_cst=1e-4,
        default_scale=False,
        default_scale_cst=False,
        add_minimum_std=False,
        # input_size=552,
        freq="H",
        prediction_length=29,
        target_dim=int(dataset.metadata.feat_static_cat[0].cardinality),
        context_length=49,
        trainer_kwargs={"max_epochs": 1, "gradient_clip_val": 0.5},
        data_kwargs={
            "batch_size": 10,
            "num_batches_per_epoch": 20,
            "num_batches_val_per_epoch": 1,
            "shuffle_buffer_length": 10,
        },
        optim_kwargs={
            "lr": 1e-3,
            "weight_decay": 1e-8,
            "patience": 10,
        },
    )

    predictor = estimatorTimeGrad.train(dataset_train)
    # makes evaluations and prediction : the most important function for testing and evaluating.

    torch.manual_seed(1)
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset_test, predictor=predictor, num_samples=10
    )

    forecasts = list(forecast_it)
    targets = list(ts_it)

    test_identical = False
    if test_identical:
        torch.manual_seed(2)
        forecast_it_2, ts_it_2 = make_evaluation_predictions(
            dataset=dataset_test, predictor=predictor, num_samples=10
        )
        forecasts_2 = list(forecast_it_2)
        targets_2 = list(ts_it_2)

        for i in range(len(forecasts)):
            # a = forecasts[i].samples.all()
            # b = forecasts_2[i].samples.all()
            a = targets[i].values
            b = targets_2[i].values
            assert (
                targets[i].values == targets_2[i].values
            ).all(), "not the same targets, the test doesn't work"
            assert (
                forecasts[i].samples == forecasts_2[i].samples
            ).all(), "not the same"
        print("same")

    evaluator = MultivariateEvaluator(
        quantiles=(np.arange(20) / 20.0)[1:], target_agg_funcs={"sum": np.sum}
    )

    agg_metric, _ = evaluator(targets, forecasts, num_series=len(dataset_test))

    print("CRPS: {}".format(agg_metric["mean_wQuantileLoss"]))
    print("ND: {}".format(agg_metric["ND"]))
    print("NRMSE: {}".format(agg_metric["NRMSE"]))
    print("MSE: {}".format(agg_metric["MSE"]))

    print("CRPS-Sum: {}".format(agg_metric["m_sum_mean_wQuantileLoss"]))
    print("ND-Sum: {}".format(agg_metric["m_sum_ND"]))
    print("NRMSE-Sum: {}".format(agg_metric["m_sum_NRMSE"]))
    print("MSE-Sum: {}".format(agg_metric["m_sum_MSE"]))
