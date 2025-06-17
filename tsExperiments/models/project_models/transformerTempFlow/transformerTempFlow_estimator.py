# This code was adapted from https://github.com/zalandoresearch/pytorch-ts/blob/master/pts/model/estimator.py
# under MIT License

from typing import List, Optional
import os, sys

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
sys.path.append(os.path.join(os.environ["PROJECT_ROOT"], ".."))
import torch
import torch.nn as nn
from gluonts.core.component import validated
from gluonts.dataset.field_names import FieldName
from gluonts.time_feature import TimeFeature
from gluonts.torch.model.predictor import PyTorchPredictor
from tsExperiments.models.project_models.timeGrad.data_preprocessing import (
    fourier_time_features_from_frequency,
)
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform import (
    Transformation,
    Chain,
    InstanceSplitter,
    ValidationSplitSampler,
    TestSplitSampler,
    ExpectedNumInstanceSampler,
    RenameFields,
    AsNumpyArray,
    ExpandDimArray,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    VstackFeatures,
    SetFieldIfNotPresent,
    TargetDimIndicator,
)
from gluonts.evaluation.backtest import make_evaluation_predictions
import numpy as np
from gluonts.evaluation import MultivariateEvaluator
from gluonts.dataset.repository import get_dataset
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from typing import List, Dict, Any, Iterable
from tsExperiments.models.project_models.timeGrad.data_preprocessing import (
    fourier_time_features_from_frequency,
)
from tsExperiments.models.project_models.transformerTempFlow.lighting_grad import (
    TransTempFlowLighting,
)

from tsExperiments.Estimator import PyTorchLightningEstimator
from gluonts.dataset.loader import as_stacked_batches
from gluonts.dataset.common import Dataset
from gluonts.itertools import Cyclic
from pandas.tseries.frequencies import to_offset


class TransformerTempFlowEstimator(PyTorchLightningEstimator):
    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        target_dim: int,
        context_length: Optional[int] = None,
        d_model: int = 32,
        dim_feedforward_scale: int = 4,
        act_type: str = "gelu",
        num_heads: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        num_parallel_samples: int = 100,
        dropout_rate: float = 0.1,
        use_feat_dynamic_real: bool = False,
        flow_type="RealNVP",
        n_blocks=3,
        hidden_size=100,
        n_hidden=2,
        conditioning_length: int = 100,
        dequantize: bool = False,
        scaling: bool = True,
        pick_incomplete: bool = False,
        lags_seq: Optional[List[int]] = None,
        time_features: Optional[List[TimeFeature]] = None,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        data_kwargs: Optional[Dict[str, Any]] = None,
        optim_kwargs: Optional[Dict[str, Any]] = None,
        num_feat_dynamic_real: int = 4,
        embedding_dimension: int = 0,
        num_hypotheses: int = 1,  # Not used here
        beta: float = 1.0,
        scaler_type: str = "mean",
        div_by_std: bool = False,
        minimum_std: float = 1e-6,
        minimum_std_cst: float = 1e-6,
        default_scale: bool = False,
        default_scale_cst: bool = False,
        add_minimum_std: bool = False,
        **kwargs,
    ) -> None:

        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.trainer_kwargs = trainer_kwargs
        self.optim_kwargs = optim_kwargs
        self.max_epochs = trainer_kwargs["max_epochs"]
        self.gradient_clip_val = trainer_kwargs["gradient_clip_val"]
        self.embedding_dimension = embedding_dimension

        self.batch_size = data_kwargs["batch_size"]
        self.num_batches_per_epoch = data_kwargs["num_batches_per_epoch"]
        self.num_batches_val_per_epoch = data_kwargs["num_batches_val_per_epoch"]
        self.shuffle_buffer_length = data_kwargs["shuffle_buffer_length"]

        super().__init__(trainer_kwargs=trainer_kwargs, **kwargs)

        self.freq = freq
        self.context_length = (
            context_length if context_length is not None else prediction_length
        )

        self.inputs_names = [
            "target_dimension_indicator",
            "past_time_feat",
            "past_target_cdf",
            "past_observed_values",
            "past_is_pad",
            "future_time_feat",
            "future_target_cdf",
            "future_observed_values",
        ]

        # self.input_size = input_size
        self.prediction_length = prediction_length
        self.target_dim = target_dim

        self.d_model = d_model
        self.num_heads = num_heads
        self.act_type = act_type
        self.dim_feedforward_scale = dim_feedforward_scale
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers

        self.num_parallel_samples = num_parallel_samples
        self.dropout_rate = dropout_rate

        self.use_feat_dynamic_real = use_feat_dynamic_real

        self.flow_type = flow_type
        self.n_blocks = n_blocks
        self.hidden_size = hidden_size
        self.n_hidden = n_hidden
        self.conditioning_length = conditioning_length
        self.dequantize = dequantize

        self.lags_seq = (
            lags_seq
            if lags_seq is not None
            else lags_for_fourier_time_features_from_frequency(freq_str=freq)
        )

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

        self.validation_sampler = ValidationSplitSampler(
            min_past=0 if pick_incomplete else self.history_length,
            min_future=prediction_length,
        )

        self.scaler_type = scaler_type
        self.div_by_std = div_by_std
        self.minimum_std = minimum_std
        self.minimum_std_cst = minimum_std_cst
        self.default_scale = default_scale
        self.default_scale_cst = default_scale_cst
        self.add_minimum_std = add_minimum_std

    def create_transformation(self) -> Transformation:
        # remove_field_names = [FieldName.FEAT_DYNAMIC_CAT]
        # if not self.use_feat_dynamic_real:
        #     remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)

        return Chain(
            [
                # RemoveFields(field_names=remove_field_names),
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
                    input_fields=[FieldName.FEAT_TIME]
                    + (
                        [FieldName.FEAT_DYNAMIC_REAL]
                        if self.use_feat_dynamic_real
                        else []
                    ),
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
            "validation": self.validation_sampler,
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

    def create_lightning_module(self) -> nn.Module:
        return TransTempFlowLighting(
            model_kwargs={
                "num_parallel_samples": self.num_parallel_samples,
                # "input_size": self.input_size,
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "act_type": self.act_type,
                "dropout_rate": self.dropout_rate,
                "dim_feedforward_scale": self.dim_feedforward_scale,
                "num_encoder_layers": self.num_encoder_layers,
                "num_decoder_layers": self.num_decoder_layers,
                "history_length": self.history_length,
                "context_length": self.context_length,
                "prediction_length": self.prediction_length,
                "lags_seq": self.lags_seq,
                "target_dim": self.target_dim,
                "conditioning_length": self.conditioning_length,
                "flow_type": self.flow_type,
                "n_blocks": self.n_blocks,
                "hidden_size": self.hidden_size,
                "n_hidden": self.n_hidden,
                "dequantize": self.dequantize,
                "num_feat_dynamic_real": self.num_feat_dynamic_real,
                "scaling": self.scaling,
                "embed_dim": self.embedding_dimension,
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

    def create_training_data_loader(self, data: Dataset, module, **kwargs) -> Iterable:
        data = Cyclic(
            data
        ).stream()  # permet de construire un itérable. Just the perfect one.
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

    def create_predictor(
        self, transformation: Transformation, module
    ) -> PyTorchPredictor:
        prediction_splitter = self._create_instance_splitter(module, "test")
        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            input_names=self.inputs_names,  # TODO. PREDICTION INPUTS NAMES IS NOT INPUTS_NAMES
            prediction_net=module,
            batch_size=self.batch_size,
            prediction_length=self.prediction_length,
            device="auto",
        )


def lags_for_fourier_time_features_from_frequency(
    freq_str: str, num_lags: Optional[int] = None
) -> List[int]:
    offset = to_offset(freq_str)
    multiple, granularity = offset.n, offset.name

    if granularity == "M":
        lags = [[1, 12]]
    elif granularity == "D":
        lags = [[1, 7, 14]]
    elif granularity == "B":
        lags = [[1, 2]]
    elif granularity == "H":
        lags = [[1, 24, 168]]
    elif granularity in ("T", "min"):
        lags = [[1, 4, 12, 24, 48]]
    else:
        lags = [[1]]

    # use less lags
    output_lags = list([int(lag) for sub_list in lags for lag in sub_list])
    output_lags = sorted(list(set(output_lags)))
    return output_lags[:num_lags]
