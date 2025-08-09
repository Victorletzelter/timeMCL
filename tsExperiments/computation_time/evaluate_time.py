import os, sys

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
sys.path.append(os.path.join(os.environ["PROJECT_ROOT"], ".."))

import pandas as pd
import yaml
import time

# specific imports
from tsExperiments.models.project_models.tactis2.estimator import TACTiSEstimator
from tsExperiments.models.project_models.timeGrad.timeGradEstimator import (
    TimEstimatorGrad,
)
from tsExperiments.models.project_models.tMCL.timeMCL_estimator import timeMCL_estimator
from tsExperiments.models.project_models.deepAR.estimator import deepAREstimator
import pandas as pd

from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.dataset.repository import get_dataset
from gluonts.dataset.multivariate_grouper import MultivariateGrouper

DATASET_NAME = []


def convert_numeric_values(d):
    """
    Recursively convert string values in a dictionary (or list) to numeric values (int or float)
    if possible.
    """
    if isinstance(d, dict):
        new_d = {}
        for key, value in d.items():
            new_d[key] = convert_numeric_values(value)
        return new_d
    elif isinstance(d, list):
        return [convert_numeric_values(item) for item in d]
    elif isinstance(d, str):
        try:
            # Attempt to convert the string to a float
            num = float(d)
            # If the value is an integer, return it as int
            if num.is_integer():
                return int(num)
            else:
                return num
        except ValueError:
            # If conversion fails, return the original string
            return d
    else:
        # For other types, return as is
        return d


def compute_metric(
    model_name,
    dataset_name,
    model_config="configs/model/deepVAR.yaml",
    nb_hyp=100,
    dataset_path=None,
):

    # trainer kwarfs and datakawags have no importance here, as we don't train

    trainer_kwargs = {
        "max_epochs": 0,
        "gradient_clip_val": 10,
    }

    data_kwargs = {
        "batch_size": 64,
        "num_batches_per_epoch": 5,
        "num_batches_val_per_epoch": 1,
        "shuffle_buffer_length": 10,
    }

    with open(model_config, "r") as f:
        cfg = yaml.safe_load(f)

    model_params = cfg["params"]
    if dataset_name == "taxi_30min":
        num_feat_dynamic_real = 6
    elif dataset_name == "wiki-rolling_nips":
        num_feat_dynamic_real = 2
    else:
        num_feat_dynamic_real = 4

    if model_name != "tactis2":
        model_params["num_feat_dynamic_real"] = num_feat_dynamic_real

    model_params = convert_numeric_values(model_params)
    # Transform dataset_path to Path object
    from pathlib import Path

    dataset_path = Path(dataset_path)
    dataset = get_dataset(dataset_name, regenerate=False, path=dataset_path)
    metadata = dataset.metadata

    train_grouper = MultivariateGrouper(
        max_target_dim=min(2000, int(metadata.feat_static_cat[0].cardinality))
    )
    test_grouper = MultivariateGrouper(
        num_test_dates=int(len(dataset.test) / len(dataset.train)),
        max_target_dim=min(2000, int(metadata.feat_static_cat[0].cardinality)),
    )

    dataset_train = train_grouper(dataset.train)
    dataset_test = test_grouper(dataset.test)

    if model_name == "timeMCL":
        model_params["num_hypotheses"] = nb_hyp
        model_params["num_parallel_samples"] = 1
        estimator = timeMCL_estimator(
            freq=metadata.freq,
            prediction_length=metadata.prediction_length,
            target_dim=min(2000, int(metadata.feat_static_cat[0].cardinality)),
            context_length=metadata.prediction_length,
            trainer_kwargs=trainer_kwargs,
            data_kwargs=data_kwargs,
            **model_params,
        )

    elif model_name == "timeGrad":
        model_params["num_parallel_samples"] = 1
        estimator = TimEstimatorGrad(
            freq=metadata.freq,
            prediction_length=metadata.prediction_length,
            target_dim=min(2000, int(metadata.feat_static_cat[0].cardinality)),
            context_length=metadata.prediction_length,
            trainer_kwargs=trainer_kwargs,
            data_kwargs=data_kwargs,
            **model_params,
        )

    elif model_name == "deepAR":
        model_params["num_parallel_samples"] = 1
        model_params["dist_params"]["dim"] = min(
            2000, int(metadata.feat_static_cat[0].cardinality)
        )

        estimator = deepAREstimator(
            freq=metadata.freq,
            prediction_length=metadata.prediction_length,
            target_dim=min(2000, int(metadata.feat_static_cat[0].cardinality)),
            context_length=metadata.prediction_length,
            trainer_kwargs=trainer_kwargs,
            data_kwargs=data_kwargs,
            **model_params,
        )

    elif model_name == "tempflow":
        from tsExperiments.models.project_models.tempflow.tempFlow_estimator import (
            TempFlowEstimator,
        )

        model_params["num_parallel_samples"] = 1
        estimator = TempFlowEstimator(
            freq=metadata.freq,
            prediction_length=metadata.prediction_length,
            target_dim=min(2000, int(metadata.feat_static_cat[0].cardinality)),
            context_length=metadata.prediction_length,
            trainer_kwargs=trainer_kwargs,
            data_kwargs=data_kwargs,
            **model_params,
        )

    elif model_name == "transformer_tempflow":
        from tsExperiments.models.project_models.transformerTempFlow.transformerTempFlow_estimator import (
            TransformerTempFlowEstimator,
        )

        model_params["num_parallel_samples"] = 1
        estimator = TransformerTempFlowEstimator(
            freq=metadata.freq,
            prediction_length=metadata.prediction_length,
            target_dim=min(2000, int(metadata.feat_static_cat[0].cardinality)),
            context_length=metadata.prediction_length,
            trainer_kwargs=trainer_kwargs,
            data_kwargs=data_kwargs,
            **model_params,
        )

    elif model_name == "tactis2":
        # trainer_kwargs["is_tactis"] = True
        model_params["nb_epoch_phase_1"] = 0
        model_params["num_parallel_samples"] = 1
        estimator = TACTiSEstimator(
            target_dim=min(2000, int(metadata.feat_static_cat[0].cardinality)),
            context_length=metadata.prediction_length,
            prediction_length=metadata.prediction_length,
            freq=metadata.freq,
            trainer_kwargs=trainer_kwargs,
            data_kwargs=data_kwargs,
            **model_params,
        )

    elif model_name == "ETS":
        from tsExperiments.models.project_models.ETS.utils import (
            creating_target_list,
            forecast_toy,
        )

        dataset_train = creating_target_list(dataset_train)[
            0
        ]  # the dataset for training
        from tsExperiments.models.project_models.ETS.model import ETSForecastModel

        predictor = ETSForecastModel(
            forecast_steps=metadata.prediction_length,
            context_length=metadata.prediction_length,
        )
        targets = creating_target_list(dataset_test)

    # training on 0 epoch
    if model_name == "ETS":
        predictor.fit(dataset_train)
    else:
        predictor = estimator.train(training_data=dataset_train)

    if model_name == "ETS":
        start_time = time.time()
        forecasts = forecast_toy(
            target_list=targets,
            context_length=metadata.prediction_length,
            trained_model=predictor,
            num_samples=nb_hyp,
            pred_length=metadata.prediction_length,
        )
        inference_time = time.time() - start_time
    else:
        start_time = time.time()
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=dataset_test, predictor=predictor, num_samples=nb_hyp
        )
        forecasts = list(forecast_it)
        targets = list(ts_it)  # we do iterations. Important to let this part.
        inference_time = time.time() - start_time

    return inference_time


def main(
    list_model_names,
    list_models_config,
    list_dataset_names,
    nb_simu=1,
    list_nb_hyp=[100],
    dataset_path=None,
):
    """nb simu : the number of time you want to compute the models time
    nb_hyp: the number of hypothesis"""

    model_name = list_model_names
    model_configs = list_models_config

    datasets_name = list_dataset_names
    out = pd.DataFrame()
    nb = 0
    for nb_hyp in list_nb_hyp:
        for number_simu in range(nb_simu):
            for i in range(len(model_name)):
                for j in range(len(datasets_name)):
                    inf_time = compute_metric(
                        model_name=model_name[i],
                        dataset_name=datasets_name[j],
                        model_config=model_configs[i],
                        nb_hyp=nb_hyp,
                        dataset_path=dataset_path,
                    )
                    out.loc[nb, "dataset_name"] = datasets_name[j]
                    out.loc[nb, "model"] = model_name[i]
                    out.loc[nb, "time"] = inf_time
                    out.loc[nb, "simu_number"] = number_simu  # the number of the simu
                    out.loc[nb, "nb_hyp"] = nb_hyp
                    nb += 1
                    print(out)
    return out


if __name__ == "__main__":

    # ------- JUST NEED TO RUN THE FOLLOWING ----------------------

    # list_model_names = ["timeMCL"]
    list_model_names = [
        "ETS",
        "timeMCL",
        "tempflow",
        "transformer_tempflow",
        "timeGrad",
        "tactis2",
        "deepAR",
    ]
    list_models_config = [
        "configs/model/ETS.yaml",
        "configs/model/timeMCL.yaml",
        "configs/model/tempflow.yaml",
        "configs/model/transformer_tempflow.yaml",
        "configs/model/timeGrad.yaml",
        "configs/model/tactis2.yaml",
        "configs/model/deepAR.yaml",
    ]
    root_path = os.path.join(os.environ["PROJECT_ROOT"], "tsExperiments")
    list_models_config = [os.path.join(root_path, e) for e in list_models_config]
    list_dataset_names = [
        "exchange_rate_nips",
        "electricity_nips",
        "solar_nips",
        "taxi_30min",
        "traffic_nips",
        "wiki-rolling_nips",
    ]

    # Parser for the number of hypothesis
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="exchange_rate_nips")
    parser.add_argument("--nb_simu", type=int, default=16)
    args = parser.parse_args()
    dataset_name = args.dataset_name
    list_dataset_names = [dataset_name]
    nb_simu = args.nb_simu

    dataset_path = "~/.gluonts/datasets"

    dataset_time = main(
        list_model_names,
        list_models_config,
        list_dataset_names,
        nb_simu=nb_simu,  # update the total number you want to compute time
        list_nb_hyp=[1, 2, 3, 4, 5, 8, 16],
        dataset_path=dataset_path,
    )  # the nb of hyp you want to compute.

    name_of_file = f"time_of_runs_{dataset_name}_nbsimu_{nb_simu}.csv"
    # Save the dataset
    dataset_time.to_csv(
        f"{os.path.join(os.environ['PROJECT_ROOT'], 'tsExperiments', 'computation_time', 'results', name_of_file)}"
    )  # saving the dataset.
