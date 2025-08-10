# src/train.py
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import sys
import pytorch_lightning as pl
from lightning.pytorch.loggers import Logger
import numpy as np
import torch
import rootutils
from typing import List
import pickle

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
sys.path.append(os.path.dirname(os.environ["PROJECT_ROOT"]))

from tsExperiments.models.project_models.tactis2.estimator import TACTiSEstimator
from tsExperiments.models.project_models.timeGrad.timeGradEstimator import (
    TimEstimatorGrad,
)
from tsExperiments.models.project_models.tMCL.timeMCL_estimator import timeMCL_estimator
from tsExperiments.models.project_models.deepAR.estimator import (
    deepVAREstimator
)
from tsExperiments.models.project_models.tempflow.tempFlow_estimator import (
    TempFlowEstimator,
)
from tsExperiments.utils.utils import compute_metric_forecast
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.dataset.repository import get_dataset
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from tsExperiments.models.project_models.tMCL.personnalized_evaluator import (
    MultivariateEvaluator,
)
from utils import (
    RankedLogger,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    split_train_val,
)

log = RankedLogger(__name__, rank_zero_only=True)

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.seed)

    if cfg.data.train.type == "Gluonts_ds":
        try:
            dataset = get_dataset(
                cfg.data.train.dataset_name, regenerate=False, path=cfg.paths.dataset_path
            )
            metadata = dataset.metadata
        except:
            dataset = get_dataset(cfg.data.train.dataset_name, regenerate=True)
            metadata = dataset.metadata
        target_dim = min(2000, int(metadata.feat_static_cat[0].cardinality))

        train_grouper = MultivariateGrouper(
            max_target_dim=target_dim
        )
        test_grouper = MultivariateGrouper(
            num_test_dates=int(len(dataset.test) / len(dataset.train)),
            max_target_dim=target_dim,
        )
        log.info(
            f"Using {int(len(dataset.test)/len(dataset.train))} rolling windows for testing"
        )

        dataset_train = train_grouper(dataset.train)
        dataset_test = test_grouper(dataset.test)

        if cfg.model.name == "ETS":
            cfg.data.train.split_train_val = False

        if "split_train_val" in cfg.data.train and cfg.data.train.split_train_val:
            log.info("Splitting train and validation datasets")
            dataset_train, dataset_val = split_train_val(
                dataset_name=cfg.data.train.dataset_name,
                grouped_train=dataset_train,
                n_pred_steps_val=cfg.data.train.n_pred_steps_val,
            )

    elif cfg.data.train.type == "financial_data":
        from tsExperiments.stock_market_data.data_processors.creating_dataset import YahooFinanceDataset
        log.info(f'Loading data, ticker list: {cfg.data.crypto_tickers}')
        if cfg.data.load_from_csv:
            dataLoader = YahooFinanceDataset(cfg.data.start_date,cfg.data.end_date,time_interval="60m",ticker_list=list(cfg.data.crypto_tickers), adress_to_save_file=cfg.data.save_path, load_from_csv=True, load_from_csv_path=cfg.data.load_from_csv_path)
        else:
            dataLoader = YahooFinanceDataset(cfg.data.start_date,cfg.data.end_date,time_interval="60m",ticker_list=list(cfg.data.crypto_tickers), adress_to_save_file=cfg.data.save_path)
        dataset_train = dataLoader.creating_train_or_val_dataset(start_date=cfg.data.train.start_date,end_date=cfg.data.train.end_date)
        dataset_val = dataLoader.creating_train_or_val_dataset(start_date=cfg.data.valid.start_date,end_date=cfg.data.valid.end_date)
        dataset_test = dataLoader.creating_test_dataset(start_date=cfg.data.test.start_date,end_date=cfg.data.test.end_date,num_tests=cfg.data.test.num_tests)
        metadata = dataLoader.generating_metaData()
        target_dim=min(2000,int(metadata.feat_static_cat[0].cardinality))

        log.info(f'Financial data successfully loaded')

    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))
    callbacks = instantiate_callbacks(cfg.callbacks)

    trainer_kwargs = {}
    trainer_kwargs["callbacks"] = callbacks
    trainer_kwargs["logger"] = logger

    # Ajouter les autres trainer_kwargs
    for key, value in cfg.trainer.items():
        if key not in ["callbacks", "logger"]:
            trainer_kwargs[key] = value

    # instanciate in function of the model called...
    model_name = cfg.model.name
    model_params = cfg.model.params

    if model_name == "timeMCL":
        estimator = timeMCL_estimator(
            freq=metadata.freq,
            prediction_length=metadata.prediction_length,
            target_dim=target_dim,
            context_length=metadata.prediction_length,
            trainer_kwargs=trainer_kwargs,
            data_kwargs=cfg.data,
            **model_params,
        )

    elif model_name == "timeGrad":
        estimator = TimEstimatorGrad(
            freq=metadata.freq,
            prediction_length=metadata.prediction_length,
            target_dim=target_dim,
            context_length=metadata.prediction_length,
            trainer_kwargs=trainer_kwargs,
            data_kwargs=cfg.data,
            **model_params,
        )

    elif model_name == "deepAR":

        log.info(
            f"Setting the output dimension of the distribution to the data dimension: {metadata.feat_static_cat[0].cardinality}"
        )
        model_params["dist_params"]["dim"] = target_dim

        estimator = deepVAREstimator(
            freq=metadata.freq,
            prediction_length=metadata.prediction_length,
            target_dim=target_dim,
            context_length=metadata.prediction_length,
            trainer_kwargs=trainer_kwargs,
            data_kwargs=cfg.data,
            **model_params,
        )

    elif model_name == "tempflow":
        estimator = TempFlowEstimator(
            freq=metadata.freq,
            prediction_length=metadata.prediction_length,
            target_dim=target_dim,
            context_length=metadata.prediction_length,
            trainer_kwargs=trainer_kwargs,
            data_kwargs=cfg.data,
            **model_params,
        )

    elif model_name == "transformer_tempflow":
        from tsExperiments.models.project_models.transformerTempFlow.transformerTempFlow_estimator import (
            TransformerTempFlowEstimator,
        )

        estimator = TransformerTempFlowEstimator(
            freq=metadata.freq,
            prediction_length=metadata.prediction_length,
            target_dim=target_dim,
            context_length=metadata.prediction_length,
            trainer_kwargs=trainer_kwargs,
            data_kwargs=cfg.data,
            **model_params,
        )

    elif model_name == "tactis2":
        estimator = TACTiSEstimator(
            target_dim=target_dim,
            context_length=metadata.prediction_length,
            prediction_length=metadata.prediction_length,
            freq=metadata.freq,
            trainer_kwargs=trainer_kwargs,
            data_kwargs=cfg.data,
            **model_params,
        )

    elif model_name == "ETS":
        from tsExperiments.models.project_models.ETS.model import ETSForecastModel
        from tsExperiments.models.project_models.ETS.utils import (
            creating_target_list,
            forecast_ets,
        )

        dataset_train = creating_target_list(dataset_train)[0]  # the dataset for training
        estimator = ETSForecastModel(
            forecast_steps=metadata.prediction_length,
            context_length=metadata.prediction_length,
        )

    else:
        raise ValueError(f"Model {model_name} not supported")

    object_dict = {
        "cfg": cfg,
        "callbacks": callbacks,
        "model": (
            estimator.create_lightning_module() if cfg.model.name != "ETS" else None
        ),
        "logger": logger,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict, logger)

    if cfg.ckpt_path is not None or (
        cfg.ckpt_path_phase1 is not None and cfg.ckpt_path_phase2 is not None
    ):
        transformation = estimator.create_transformation()
        training_network = estimator.create_lightning_module()
        if cfg.model.name == "tactis2":
            assert (
                cfg.train is False
            ), "Retraining Tactis2 from checkpoint is not supported"
            log.info(f"Loading checkpoint from {cfg.ckpt_path_phase1}")
            predictor = training_network.__class__.load_from_checkpoint(
                cfg.ckpt_path_phase1
            )
            log.info(f"Loading checkpoint from {cfg.ckpt_path_phase2}")
            predictor.switch_to_stage_2(predictor.model, "adam")
            predictor.load_state_dict(torch.load(cfg.ckpt_path_phase2)["state_dict"])
        else:
            log.info(f"Loading checkpoint from {cfg.ckpt_path}")
            predictor = training_network.__class__.load_from_checkpoint(cfg.ckpt_path)
        log.info(f"Creating predictor")
        predictor = estimator.create_predictor(transformation, predictor)

    if cfg.train is True:
        log.info(f"Training the model")
        if cfg.model.name == "ETS":
            estimator.fit(dataset_train)
        elif "split_train_val" in cfg.data.train :
            predictor = estimator.train(
                training_data=dataset_train,
                validation_data=dataset_val if not cfg.data.discard_validation else None,
                ckpt_path=cfg.ckpt_path if cfg.ckpt_path is not None else None,
            )
        else:
            predictor = estimator.train(
                training_data=dataset_train,
                ckpt_path=cfg.ckpt_path if cfg.ckpt_path is not None else None,
            )

    if (cfg.data.train.type == "Gluonts_ds" and cfg.test is True) or (cfg.data.train.type == "financial_data" and cfg.test is True):
        log.info(f"Evaluating the model")
        if cfg.model.name == "ETS":
            targets = creating_target_list(dataset_test)
            forecasts = forecast_ets(
                target_list=targets,
                context_length=metadata.prediction_length,
                trained_model=estimator,
                num_samples=cfg.model.params.num_hypotheses,
                pred_length=metadata.prediction_length,
            )
        else:
            forecast_it, ts_it = make_evaluation_predictions(
                dataset=dataset_test,
                predictor=predictor,
                num_samples=cfg.model.params.num_hypotheses,
            )
            forecasts = list(forecast_it)
            targets = list(ts_it)

        ### Computing the metrics
        evaluator = MultivariateEvaluator(
            quantiles=(np.arange(20) / 20.0)[1:], target_agg_funcs={"sum": np.sum}
        )
        if cfg.compute_usual_metrics:
            agg_metric, _, real_distorsion = evaluator(
                targets, forecasts, num_series=len(dataset_test)
            )
            agg_metric["Distorsion"] = real_distorsion
        else:
            agg_metric = {}

        tv = compute_metric_forecast(forecasts, metric_func="total_variation")
        log.info(f"Total variation: {tv}")
        agg_metric["total_variation"] = tv

        for key, value in agg_metric.items():
            if (
                key.split("_")[0].isdigit()
                or ("QuantileLoss" in key and "mean" not in key)
                or ("Coverage" in key and "MAE" not in key)
            ):  # We exclude those, they are not metrics.
                pass
            else:
                for logg in logger:
                    if hasattr(logg, "log_metrics"):
                        logg.log_metrics({key: value})
                    else:
                        log.info(f"Logger {logg} does not have a log_metrics method.")

        if cfg.compute_usual_metrics:
            log.info("RMSE: {}".format(agg_metric["m_sum_RMSE"]))
            log.info("Distorsion: {}".format(agg_metric["Distorsion"]))
            log.info("CRPS-Sum: {}".format(agg_metric["m_sum_mean_wQuantileLoss"]))

    ### Plotting the forecasts
    if cfg.model.plot_forecasts:
        if cfg.test is False:
            forecast_it, ts_it = make_evaluation_predictions(
                dataset=dataset_test,
                predictor=predictor,
                num_samples=cfg.model.params.num_hypotheses,
            )
            forecasts = list(forecast_it)
            targets = list(ts_it)

        from tsExperiments.scripts_plot.plottimeMCL import plot_mcl

        target_df = targets[0]
        hypothesis_forecasts = forecasts[
            0
        ].samples  # shape (N, forecast_length, target_dim)

        # Check if we're using timeMCL or not
        is_mcl = cfg.model.name == "timeMCL"

        # Save all the data to produce the plots

        # Save target_df as pickle instead of CSV and Numpy binary
        with open(f"{cfg.paths.output_dir}/target_df.pkl", "wb") as f:
            pickle.dump(target_df, f)

        # Save other NumPy arrays and metadata as pickle
        with open(f"{cfg.paths.output_dir}/hypothesis_forecasts.pkl", "wb") as f:
            pickle.dump(hypothesis_forecasts, f)

        with open(f"{cfg.paths.output_dir}/forecast_length.pkl", "wb") as f:
            pickle.dump(metadata.prediction_length, f)

        with open(f"{cfg.paths.output_dir}/context_points.pkl", "wb") as f:
            pickle.dump(metadata.prediction_length, f)

        with open(f"{cfg.paths.output_dir}/freq_type.pkl", "wb") as f:
            pickle.dump(metadata.freq, f)

        with open(f"{cfg.paths.output_dir}/is_mcl.pkl", "wb") as f:
            pickle.dump(is_mcl, f)

        plot_mcl(
            target_df=target_df,
            hypothesis_forecasts=hypothesis_forecasts,
            forecast_length=metadata.prediction_length,
            context_points=metadata.prediction_length,
            rows=3,
            cols=2,
            plot_mean=True,
            freq_type=metadata.freq,
            save_path=cfg.paths.output_dir,
            is_mcl=is_mcl,
            extract_unique=is_mcl,
        )

    if cfg.visualize_specific_date is True:
        from tsExperiments.scripts_plot.train_viz import plotting_from_a_date,creating_target_list,plot_forecasts_for_dimension
        test_data = dataLoader.creating_test_dataset(start_date=cfg.start_date_viz,end_date=cfg.end_date_viz,num_tests=1)
        targets_loaded = creating_target_list(test_data) 
        if cfg.model.name == "timeMCL":
            contexte_df, forecast_array,start_date,probabilities = plotting_from_a_date(date_of_pred=cfg.date_of_pred,
                                                                        plot_context_size=metadata.prediction_length,
                                                                        target_list=targets_loaded,
                                                                        pred_length=metadata.prediction_length,
                                                                        trained_model=predictor,
                                                                        num_samples=1000,
                                                                        is_mcl = True) 
        else:
             contexte_df, forecast_array,start_date,probabilities = plotting_from_a_date(date_of_pred=cfg.date_of_pred,
                                                                        plot_context_size=metadata.prediction_length,
                                                                        target_list=targets_loaded,
                                                                        pred_length=metadata.prediction_length,
                                                                        trained_model=predictor,
                                                                        num_samples=cfg.model.params.num_hypotheses,
                                                                        is_mcl = False)

        # for dimension_to_plot in [1,2,5,13]:
        for dimension_to_plot in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]:
            model_name = cfg.model.name
            fig_path = f"{os.environ['PROJECT_ROOT']}/tsExperiments/scripts_plot/{model_name}/{dimension_to_plot}" 
            # fig_path = f"{os.environ['PROJECT_ROOT']}/tsExperiments/logs/plots/{model_name}/{dimension_to_plot}" 
            # mkdir the folder if it doesn't exist
            if not os.path.exists(fig_path):
                os.makedirs(fig_path)
            plot_forecasts_for_dimension(contexte_df, forecast_array, start_date, target=dimension_to_plot, freq=None, save_path=fig_path,probabilities = probabilities, pkl_path_name=f"{dimension_to_plot}")

    if cfg.model.compute_flops:
        from tsExperiments.computation_flops.flops_computation import count_flops_for_predictions

        log.info("Computing FLOPs")
        if (
            hasattr(predictor, "prediction_net")
            and hasattr(predictor.prediction_net, "model")
            and hasattr(predictor.prediction_net.model, "num_parallel_samples")
        ):
            log.info(
                f"Setting the num parallel samples to {cfg.model.params.num_parallel_samples}"
            )
            predictor.prediction_net.model.num_parallel_samples = (
                cfg.model.params.num_parallel_samples
            )

        prediction_flops, total_flops = count_flops_for_predictions(
            predictor, dataset_test, model_name=cfg.model.name
        )

        for logg in logger:
            if hasattr(logg, "log_metrics"):
                logg.log_metrics(
                    {"prediction_flops": prediction_flops, "total_flops": total_flops}
                )
            else:
                log.info(f"Logger {logger} does not have a log_metrics method.")
        log.info("FLOPs computed")
        log.info(f"Prediction FLOPs: {prediction_flops}, Total FLOPs: {total_flops}")

    log.info("Finished experiment with run_name: {}".format(cfg.run_name))


if __name__ == "__main__":
    main()
