import numpy as np
from typing import List

# first metric : total variation.


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
