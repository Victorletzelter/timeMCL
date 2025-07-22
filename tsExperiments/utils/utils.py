import warnings
from importlib.util import find_spec
from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np
import torch
from typing import List, Optional
import numpy as np
import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.dataset.repository.datasets import (
    get_dataset,
)

from omegaconf import DictConfig
from pandas.tseries.frequencies import to_offset
from typing import List
from utils import pylogger, rich_utils
from pandas.tseries.frequencies import to_offset
from gluonts.time_feature import TimeFeature, norm_freq_str
from gluonts.time_feature import TimeFeature
from gluonts.core.component import validated

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

class FourierDateFeatures(TimeFeature):
    @validated()
    def __init__(self, freq: str) -> None:
        super().__init__()
        # reocurring freq
        freqs = [
            "month",
            "day",
            "hour",
            "minute",
            "weekofyear",
            "weekday",
            "dayofweek",
            "dayofyear",
            "daysinmonth",
        ]

        assert freq in freqs
        self.freq = freq

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        values = getattr(index, self.freq)
        num_values = max(values) + 1
        steps = [x * 2.0 * np.pi / num_values for x in values]
        return np.vstack([np.cos(steps), np.sin(steps)])

def fourier_time_features_from_frequency(freq_str: str) -> List[TimeFeature]:
    # This function was copied https://github.com/zalandoresearch/pytorch-ts/blob/master/pts/feature/fourier_date_feature.py
    offset = to_offset(freq_str)
    granularity = norm_freq_str(offset.name)

    features = {
        "M": ["weekofyear"],
        "W": ["daysinmonth", "weekofyear"],
        "D": ["dayofweek"],
        "B": ["dayofweek", "dayofyear"],
        "H": ["hour", "dayofweek"],
        "min": ["minute", "hour", "dayofweek"],
        "T": ["minute", "hour", "dayofweek"],
    }

    assert granularity in features, f"freq {granularity} not supported"

    feature_classes: List[TimeFeature] = [
        FourierDateFeatures(freq=freq) for freq in features[granularity]
    ]
    return feature_classes


def lags_for_fourier_time_features_from_frequency(
    freq_str: str, num_lags: Optional[int] = None
) -> List[int]:
    # Fonction copied from https://github.com/zalandoresearch/pytorch-ts/blob/master/pts/feature/lags.py
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


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(
    metric_dict: Dict[str, Any], metric_name: Optional[str]
) -> Optional[float]:
    """Safely retrieves value of the metric logged in LightningModule.

    :param metric_dict: A dict containing metric values.
    :param metric_name: If provided, the name of the metric to retrieve.
    :return: If a metric name was provided, the value of the metric.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


def split_train_val(dataset_name, grouped_train, n_pred_steps_val):
    """Function to split the dataset into train and validation datasets.

    Args:
        dataset_name (str): name of the dataset
        grouped_train (ListDataset): train dataset after MultivariateGrouper grouping
        val_length (int): length of the validation dataset (in number of prediction steps)

    Returns:
        train_multivar_ds (ListDataset): train dataset
        val_multivar_ds (ListDataset): validation dataset
    """

    dataset = get_dataset(dataset_name, regenerate=False)

    val_length = n_pred_steps_val * dataset.metadata.prediction_length

    grouped_train_list = list(
        grouped_train
    )  # typically 1 item per group if fully aligned
    # Suppose we have exactly 1 group. Then:
    big_multivar_item = grouped_train_list[0]

    # big_multivar_item["target"] shape: (D, T)
    # you can slice big_multivar_item["target"] by time
    # e.g., last `val_length` timesteps become validation.

    D, T = big_multivar_item["target"].shape

    # 2.1) Create a training item that excludes last val_length timesteps
    train_item = big_multivar_item.copy()
    train_item["target"] = big_multivar_item["target"][:, : (T - val_length)]

    # 2.2) Create a validation item from the last val_length timesteps
    val_item = big_multivar_item.copy()
    val_item["target"] = big_multivar_item["target"][:, (T - val_length) :]
    # shift the "start" accordingly, if you want correct timestamps
    # e.g. if freq = 'H'
    # val_item["start"] = val_item["start"] + (T - val_length) * pd.tseries.frequencies.to_offset('H')

    # Rebuild them into ListDataset
    train_multivar_ds = ListDataset(
        [train_item], freq=dataset.metadata.freq, one_dim_target=False
    )
    val_multivar_ds = ListDataset(
        [val_item], freq=dataset.metadata.freq, one_dim_target=False
    )

    return train_multivar_ds, val_multivar_ds


def compute_metric_forecast(forecasts: List, metric_func="total_variation"):
    """input : the forecasts from
    forecasts = list(forecast_it)

    metric func in "total_variation","""

    out = []
    for i in range(len(forecasts)):
        if metric_func == "total_variation":
            data = forecasts[0].samples
            out.append(total_variation(data))

    return np.mean(out)


def total_variation(data: np.ndarray):
    """
    input : np.array of dim (num_simu,pred_length,target_dim)
    output : mean of total variation on the num_simu
    """
    assert data.ndim == 3

    # diff on the time
    diff = np.diff(
        data, axis=1
    )  # (nb_simulations, longueur_time_serie-1, dimension_time_serie)

    # norm on each dim
    variations = np.linalg.norm(
        diff, axis=-1
    )  # forme: (nb_simulations, longueur_time_serie-1)

    variation_totale_sim = np.sum(variations, axis=1)  # forme: (nb_simulations,)

    moyenne_variation = np.mean(variation_totale_sim)

    return moyenne_variation


def units_test_total_variations():
    # 1. Tests on constant time series, total variation should be 0.
    array_exemple = np.ones((1, 24, 137))  # only ones
    total_variation_value = total_variation(array_exemple)
    assert (
        total_variation_value == 0
    ), f"The total variation must be 0, not {total_variation_value}"

    array_exemple = np.ones((1, 24, 1))  # only ones
    total_variation_value = total_variation(array_exemple)
    assert (
        total_variation_value == 0
    ), f"The total variation  must be 0, not {total_variation_value}"

    array_exemple = np.ones((100, 24, 137))  # only ones
    total_variation_value = total_variation(array_exemple)
    assert (
        total_variation_value == 0
    ), f"The total variation  must be 0, not {total_variation_value}"

    # 2. tests on a lineary increasing serie (by 1), total variation must be equal to np.sqrt(target_dim)

    time_steps = np.arange(10).reshape(1, 10, 1)
    array_timesteps = np.tile(time_steps, (10, 1, 1))
    total_variation_value = total_variation(array_timesteps)
    assert (
        total_variation_value == 9
    ), f"The total variation must be 9, not {total_variation_value}"

    time_steps = np.arange(20).reshape(1, 20, 1)
    array_timesteps = np.tile(time_steps, (10, 1, 30))  # (10, 20, 30)

    total_variation_value = total_variation(array_timesteps)
    assert (
        abs(total_variation_value - np.sqrt(30) * 19) < 10e-6
    ), f"The total variation must be {np.sqrt(30)*19}, not {total_variation_value}"

    time_component = np.arange(10).reshape(10, 1, 1)

    axis1_component = np.arange(5).reshape(1, 5, 1)
    axis2_component = np.arange(10).reshape(1, 1, 10)

    result = time_component + axis1_component + axis2_component

    total_variation_value = total_variation(result)
    assert (
        abs(total_variation_value - np.sqrt(10) * 4) < 10e-6
    ), f"The total must be {np.sqrt(10)*4}, not {total_variation_value}"

    print("all unit test succed")


def weighted_average(
    x: torch.Tensor, weights: Optional[torch.Tensor] = None, dim=None
) -> torch.Tensor:
    """
    Computes the weighted average of a given tensor across a given dim, masking
    values associated with weight zero,
    meaning instead of `nan * 0 = nan` you will get `0 * 0 = 0`.

    Parameters
    ----------
    x
        Input tensor, of which the average must be computed.
    weights
        Weights tensor, of the same shape as `x`.
    dim
        The dim along which to average `x`

    Returns
    -------
    Tensor:
        The tensor with values averaged along the specified `dim`.
    """
    if weights is not None:
        weighted_tensor = torch.where(weights != 0, x * weights, torch.zeros_like(x))
        sum_weights = torch.clamp(
            weights.sum(dim=dim) if dim else weights.sum(), min=1.0
        )
        return (
            weighted_tensor.sum(dim=dim) if dim else weighted_tensor.sum()
        ) / sum_weights
    else:
        return x.mean(dim=dim)