import pandas as pd
from gluonts.model.forecast import SampleForecast  #
import random
from tsExperiments.models.project_models.ETS.model import BaseForecastModel
from typing import List
import numpy as np
from gluonts.dataset.split import split
from gluonts.torch.model.predictor import PyTorchPredictor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def creating_target_list(dataset_test) -> List[pd.DataFrame]:
    target = []
    for item in dataset_test:
        df = pd.DataFrame(item["target"]).transpose()
        start_period = item["start"]
        index = pd.period_range(
            start=start_period, periods=len(df), freq=start_period.freq
        )
        df.index = index
        target.append(df)
    return target


# forecast toy
def forecast_ets(
    target_list: List[pd.DataFrame],
    context_length: int,
    pred_length: int,
    trained_model: BaseForecastModel,
    num_samples: int = 100,
) -> SampleForecast:
    """compute the forecasts from the BaseForecastModel"""

    forecasts = []

    for df in target_list:
        n_total = len(df)

        # context = df.iloc[n_total - context_length-pred_length-1:n_total-pred_length-1]
        context = df.iloc[
            n_total - pred_length - context_length : n_total - pred_length
        ]

        forecast_array = trained_model.predict(
            context, num_samples
        )  # num_samples*pred_length*dim_data

        # start_date = df.index[n_total-pred_length-1]
        start_date = df.index[n_total - pred_length]
        assert forecast_array.shape[0] == num_samples
        assert forecast_array.shape[1] == pred_length

        # creating the forecast object
        forecast_obj = SampleForecast(
            samples=forecast_array, start_date=start_date, item_id=None, info=None
        )
        forecasts.append(forecast_obj)
    return forecasts
