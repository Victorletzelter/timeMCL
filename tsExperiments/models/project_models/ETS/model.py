"""
In this file, we implement the ETS model, while trying the align the implementation with the one of the TACTIS paper.
Drouin, A., Marcotte, É., & Chapados, N. (2022, June). Tactis: Transformer-attentional copulas for time series. In International Conference on Machine Learning (pp. 5447-5493). PMLR..
"""

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from abc import ABC, abstractmethod
import numpy as np


class BaseForecastModel(ABC):
    def __init__(self, forecast_steps, context_length):
        """
        forecast_steps: number of future time steps to predict.
        context_length: number of observations used as context (optional, used for some models).
        """
        self.forecast_steps = forecast_steps
        self.context_length = context_length

    @abstractmethod
    def fit(self, df):
        """
        Train the model using the data in the DataFrame df.
        """
        pass

    @abstractmethod
    def predict(self, context, num_samples=1):
        """
        Generate num_samples future trajectories based on the provided context.

        context: DataFrame containing the most recent observations (the context).
        num_samples: number of trajectories to generate.

        Returns a np.ndarray of shape (num_samples, forecast_steps, nb_dim),
        where nb_dim depends on the model (e.g. the number of series).
        """
        pass


class ETSForecastModel(BaseForecastModel):
    def __init__(self, context_length, forecast_steps):
        """
        context_length: number of observations used as context.
        forecast_steps: number of future time steps to predict.
        """
        super().__init__(forecast_steps, context_length)
        self.trained_models = (
            {}
        )  # Dictionary to store the fitted ETS model for each series

    def fit(self, df):
        """
        Train an ETS model on the entire dataset.
        Each column in the DataFrame is independently fitted with an ETS model.

        In this implementation, we use an additive trend and no seasonality.
        Note: It is assumed that the data is already preprocessed as needed (e.g., no normalization is required).
        """
        self.trained_models = {}
        for col in df.columns:
            series = df[col]
            # Configure the ETS model with additive trend and no seasonal component.
            # The seasonal_periods parameter is arbitrary here (set to 24) since no seasonality is used.

            # The implementation of the library covers the functionality of the R library as much as possible whilst still being Pythonic.
            model = ExponentialSmoothing(
                series, trend="add", seasonal="add", seasonal_periods=24
            )  # automatical fit. And also automatical parametres search.
            fit_model = model.fit(optimized=True)
            self.trained_models[col] = fit_model

    def predict(self, context, num_samples=1):
        """
        For each series in the context, generate num_samples future trajectories
        using the simulate method of the fitted ETS model.

        context: DataFrame containing the most recent context_length observations.
        num_samples: number of trajectories to generate.

        Returns a np.ndarray with shape (num_samples, forecast_steps, nb_dim),
        where nb_dim is the number of series (columns).
        """
        predictions = {}
        for col in context.columns:
            if col not in self.trained_models:
                raise ValueError(f"The model for series '{col}' has not been trained.")
            model_fit = self.trained_models[col]
            # The simulate method typically returns an array of shape (forecast_steps, num_samples),
            # so we transpose it to get (num_samples, forecast_steps).
            samples = model_fit.simulate(self.forecast_steps, repetitions=num_samples).T
            if samples.shape[0] != num_samples:
                # unsqueeze in the first dimension
                assert (
                    num_samples == 1
                ), "Unexpected situation, num_samples is not 1 and the shape of the samples is not (num_samples, forecast_steps)"
                samples = samples[np.newaxis, :]
            predictions[col] = samples

        # Ensure the column order is the same as in the context DataFrame.
        cols = list(context.columns)
        # Stack predictions along a new third dimension to obtain shape
        # (num_samples, forecast_steps, number_of_series).
        result = np.stack([predictions[col] for col in cols], axis=2)
        return result
