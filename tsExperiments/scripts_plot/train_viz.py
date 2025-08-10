import pandas as pd
from gluonts.model.forecast import SampleForecast
from typing import List
import numpy as np
from gluonts.dataset.split import split
from gluonts.torch.model.predictor import PyTorchPredictor
import matplotlib.pyplot as plt
from tsExperiments.scripts_plot.plottimeMCL import extract_unique_forecasts
import pickle
from pathlib import Path


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


def plotting_from_a_date(
    date_of_pred: str,
    plot_context_size: int,
    target_list: List[pd.DataFrame],
    pred_length: int,
    trained_model: PyTorchPredictor,
    num_samples: int = 1000,
    is_mcl=True,
) -> SampleForecast:
    """ "returns :
    - contexte_df : the real data, around of the point of interest
    - forecast_array : the predictions of the neural network
    - start_date: the starting date of predictions of the model."""

    assert len(target_list) == 1, "the data_test must be of length 1"

    for df in target_list:

        df_context = df.loc[:date_of_pred]
        n_total = len(
            df_context
        )  # creating the dataset_test from scracth. The main idea is taking data from df
        dataset_test = {}
        dataset_test["target"] = df_context.values.transpose()  # the good dimensions
        dataset_test["start"] = df_context.index[0]
        dataset_test["feat_static_cat"] = np.array([0], dtype=int)

        window_length = pred_length
        _, test_template = split([dataset_test], offset=-window_length)
        test_data = test_template.generate_instances(window_length)
        pred = trained_model.predict(
            test_data.input, num_samples=num_samples
        )  # it unrolls on the past and then it sample.
        pred = next(iter(pred))  # we extract the prediction

        forecast_array = pred.samples
        probabilities = None  # if there is no probabilities.
        if is_mcl:
            forecast_array, probabilities = extract_unique_forecasts(forecast_array)

        start_date = df_context.index[
            n_total - pred_length
        ]  # we start predicting at this date.
        assert pred.start_date == start_date
        contexte_df = df.iloc[
            n_total - 1 - pred_length - plot_context_size : n_total
        ]  # the context we want to plot

    return contexte_df, forecast_array, start_date, probabilities


def plot_forecasts_for_dimension(
    context_df,
    forecast_array,
    start_date,
    target,
    freq=None,
    save_path=None,
    probabilities=None,
    pkl_path_name=None,
):
    """
    Plot the history and forecasts with the mean trajectory.
    Also save the data in a pkl file.
    """
    # Conversion of start_date to Timestamp (handling of pd.Period)
    if isinstance(start_date, pd.Period):
        start_date = start_date.to_timestamp()
    else:
        start_date = pd.to_datetime(start_date)

    # Determination of the index and the name of the target column
    if isinstance(target, int):
        target_idx = target
        col_name = context_df.columns[target]
    else:
        col_name = target
        target_idx = context_df.columns.get_loc(col_name)

    # Extraction of the historical series for the chosen dimension
    historical = context_df[col_name]

    # Conversion of the index to DatetimeIndex if necessary
    if isinstance(historical.index, pd.PeriodIndex):
        hist_index = historical.index.to_timestamp()
    else:
        hist_index = historical.index

    # Determination of the frequency
    if freq is None:
        if isinstance(context_df.index, pd.PeriodIndex):
            freq = (
                context_df.index.freqstr
                if context_df.index.freqstr is not None
                else "D"
            )
        else:
            freq = pd.infer_freq(context_df.index)
            if freq is None:
                freq = "D"

    # Forecast length according to the shape of forecast_array
    forecast_length = forecast_array.shape[1]

    # Creation of the forecast dates from start_date
    forecast_dates = pd.date_range(start=start_date, periods=forecast_length, freq=freq)

    # Creation of the graph
    plt.figure(figsize=(12, 6))

    # Plot of the complete history
    plt.plot(hist_index, historical, label="History", color="blue")

    num_samples = forecast_array.shape[0]

    if probabilities is None:
        # Plot of each forecast in orange if no probabilities are provided
        for i in range(num_samples):
            sample_forecast = forecast_array[i, :, target_idx]
            plt.plot(
                forecast_dates,
                sample_forecast,
                color="orange",
                alpha=0.3,
                label="Forecast Sample" if i == 0 else None,
            )
    else:
        # Check the consistency of the dimensions
        if probabilities.shape != forecast_array.shape:
            raise ValueError(
                "The probabilities argument must have the same shape as forecast_array."
            )

        # Use the first probability of each trajectory to define the color
        first_probs = probabilities[:, 0, target_idx]
        pmin = np.min(first_probs)
        pmax = np.max(first_probs)

        # Normalization function on a narrow range (e.g. [0.4, 0.8])
        def norm(p):
            if pmax > pmin:
                return 0.4 + (p - pmin) / (pmax - pmin) * (0.8 - 0.4)
            else:
                return 0.6

        # Plot of each forecast with the color based on the first probability
        for i in range(num_samples):
            sample_forecast = forecast_array[i, :, target_idx]
            p_value = first_probs[i]
            color_value = plt.cm.Greens(norm(p_value))
            plt.plot(
                forecast_dates,
                sample_forecast,
                color=color_value,
                linewidth=2,
                label=f"Forecast Sample (Prob: {p_value:.3f})",
            )

    # Calcul and plot of the mean trajectory
    if probabilities is None:
        # Simple mean if no probabilities
        mean_forecast = np.mean(forecast_array[:, :, target_idx], axis=0)
        plt.plot(
            forecast_dates,
            mean_forecast,
            color="red",
            linewidth=2,
            label="Mean of predictions",
            zorder=5,
        )
    else:
        # Weighted mean by probabilities
        # Normalize the probabilities
        probs = probabilities[:, 0, target_idx]
        weights = probs / np.sum(probs)
        weighted_mean = np.sum(
            forecast_array[:, :, target_idx] * weights[:, np.newaxis], axis=0
        )
        plt.plot(
            forecast_dates,
            weighted_mean,
            color="red",
            linewidth=2,
            label="Weighted mean of predictions",
            zorder=5,
        )

    # Mark the start date of the forecast
    plt.axvline(x=start_date, color="red", linestyle="--", label="Start of Forecast")

    plt.xlabel("Time")
    plt.ylabel(col_name)
    plt.title(f"Historical data and forecasts for {col_name}")
    plt.legend()
    plt.grid(True)

    if save_path is not None:
        # Create the folder to save the data
        save_dir = Path(save_path).parent / Path(save_path).stem
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save the plot
        plt.savefig(save_path)

        # Prepare the data to save
        data_to_save = {
            "historical_dates": hist_index,
            "historical_values": historical.values,
            "forecast_dates": forecast_dates,
            "forecast_array": forecast_array[:, :, target_idx],
            "probabilities": (
                probabilities[:, :, target_idx] if probabilities is not None else None
            ),
            "mean_forecast": (
                weighted_mean if probabilities is not None else mean_forecast
            ),
            "target_name": col_name,
            "start_date": start_date,
        }

        # Save the data
        pkl_path = save_dir / f"{pkl_path_name}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(data_to_save, f)

        print(f"Data saved in {pkl_path}")

    plt.show()
