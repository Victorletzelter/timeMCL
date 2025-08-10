# %%

import rootutils, sys, os

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
sys.path.append(os.path.dirname(os.environ["PROJECT_ROOT"]))
import numpy as np

# %% plot function


def extract_unique_forecasts(samples, is_mcl=True):
    """Extract unique forecasts and their probabilities.

    Args:
        samples: Array of shape (n_hyp, forecast_length, target_dim) for TimeGrad
                or (n_hyp, forecast_length, target_dim+1) for timeMCL
        is_mcl: Whether the samples come from timeMCL model (with scores)
    """
    n_hyp = samples.shape[0]

    reshaped_forecasts = samples.reshape(n_hyp, -1)
    unique_forecasts, unique_indices = np.unique(
        reshaped_forecasts, axis=0, return_index=True
    )
    hypothesis_forecasts = samples[unique_indices]

    if is_mcl:
        # Use provided scores for probabilities
        probabilities = np.zeros_like(hypothesis_forecasts)
        for i in range(hypothesis_forecasts.shape[0]):
            probabilities[i] = 0
            for j in range(samples.shape[0]):
                if np.all(samples[j] == hypothesis_forecasts[i]):
                    probabilities[i] += 1
        probabilities = probabilities / probabilities.sum(axis=0, keepdims=True)
    else:
        # For TimeGrad, use equal probabilities
        probabilities = np.ones_like(hypothesis_forecasts) / probabilities.sum(
            axis=0, keepdims=True
        )

    return hypothesis_forecasts, probabilities


def plot_mcl(
    target_df,
    hypothesis_forecasts,
    forecast_length: int,
    rows=4,
    cols=4,
    plot_mean=True,
    context_points=None,
    freq_type="H",
    fname="Predictions_plot.png",
    extract_unique=True,
    save_path=None,
    is_mcl=True,  # Parameter to distinguish between timeMCL and TimeGrad
):
    """
    Plot forecasts from either timeMCL or TimeGrad models.

    Args:
        ... (existing args) ...
        is_mcl: Whether the forecasts come from timeMCL model (with scores) or TimeGrad
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    # Handle forecasts based on model type
    if is_mcl:
        # For timeMCL, extract unique forecasts if requested
        if extract_unique:
            hypothesis_forecasts, probabilities = extract_unique_forecasts(
                hypothesis_forecasts
            )
        else:
            # Extract scores for non-unique forecasts
            scores = hypothesis_forecasts[:, :, -1]
            hypothesis_forecasts = hypothesis_forecasts[:, :, :-1]
            probabilities = scores / scores.sum(axis=0, keepdims=True)
    else:
        # For TimeGrad, use equal probabilities for all samples
        probabilities = (
            np.ones_like(hypothesis_forecasts) / hypothesis_forecasts.shape[0]
        )

    # 1) check PeriodIndex
    if isinstance(target_df.index, pd.PeriodIndex):
        target_df.index = target_df.index.to_timestamp()

    # entire data
    time_index = target_df.index
    values = target_df.values
    full_len, target_dim = values.shape

    k, fcst_len, d_ = hypothesis_forecasts.shape
    assert fcst_len == forecast_length, "forecast_length mismatch"

    # 2) define how many context points we want before the 'end of training'
    #    if not given, default to 2 * forecast_length
    if context_points is None:
        context_points = 2 * forecast_length

    # 3) suppose "end of training" is (full_len - forecast_length)
    #    i.e. the last training point is full_len - forecast_length - 1
    train_end_idx = full_len - forecast_length
    if train_end_idx < 0:
        # edge case: dataset too short vs forecast_length
        train_end_idx = full_len

    # 4) we want from (train_end_idx - context_points) up to the end of the dataset
    start_idx = max(0, train_end_idx - context_points)

    # slice the data in that window
    tail_index = time_index[start_idx:]
    tail_values = values[start_idx:]
    if len(tail_values) == 0:
        print("WARNING: tail is empty!")
        return

    # 5) freq inference
    freq = tail_index.freq
    if freq is None:
        freq = pd.infer_freq(tail_index)
    if freq is None and len(tail_index) > 1:
        freq = tail_index[-1] - tail_index[-2]
    if freq is None:
        freq = pd.Timedelta("1D")

    # 6) last training timestamp
    #    it's here that the forecast should start
    last_training_ts = (
        time_index[train_end_idx - 1] if train_end_idx > 0 else time_index[-1]
    )

    # 7) build future times from that point
    future_times = pd.date_range(
        start=last_training_ts + freq, periods=forecast_length, freq=freq
    )

    # 8) mean forecast if asked
    mean_forecast = None
    if plot_mean:
        # weighted mean by the probabilities
        mean_forecast = np.sum(
            hypothesis_forecasts * probabilities, axis=0
        )  # shape (forecast_length, d)

    # plot
    fig, axs = plt.subplots(rows, cols, figsize=(9 * cols, 5.5 * rows))
    axs = axs.ravel()
    num_plots = min(rows * cols, target_dim)

    for dim_idx in range(num_plots):
        ax = axs[dim_idx]

        # Entire tail of real data
        ax.plot(
            tail_index,
            tail_values[:, dim_idx],
            color="black",
            lw=2,
            label="Observations" if dim_idx == 0 else None,
        )

        for h_idx in range(len(hypothesis_forecasts)):
            prob = probabilities[h_idx][0, 0]
            alpha = (
                prob if extract_unique else 0.6
            )  # Use probability as alpha if unique forecasts
            ax.plot(
                future_times,
                hypothesis_forecasts[h_idx, :, dim_idx],
                lw=2,
                alpha=0.5 + alpha / 2,
                label=(
                    f"hyp {h_idx+1} (p={prob:.5f})"
                    if (dim_idx == 0 and extract_unique)
                    else None
                ),
            )

        # Optional mean
        if mean_forecast is not None:
            ax.plot(
                future_times,
                mean_forecast[:, dim_idx],
                color="red",
                lw=2,
                label="mean" if dim_idx == 0 else None,
            )

        ax.set_title(f"dimension {dim_idx}", fontsize=18)  # Augmente la taille du titre

        # Customize x-axis based on frequency type
        if freq_type == "H":
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))  # Every 6 hours
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))  # Format hours
            ax.xaxis.set_minor_locator(mdates.DayLocator())
            ax.xaxis.set_minor_formatter(
                mdates.DateFormatter("%d-%b")
            )  # Format date (e.g., 16-Jun)
        elif freq_type == "D":
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # Every month
            ax.xaxis.set_major_formatter(
                mdates.DateFormatter("%b-%Y")
            )  # Format: month-year (e.g., Jun-2023)

        ax.tick_params(axis="x", which="major", labelsize=11)
        ax.tick_params(axis="x", which="minor", labelsize=11, pad=15)
        ax.tick_params(
            axis="y", labelsize=15
        )  # Augmente la taille des ticks de l'axe Y
        ax.yaxis.label.set_size(18)  # Augmente la taille du label de l'axe Y

    if len(hypothesis_forecasts) <= 10:
        handles, labels = axs[0].get_legend_handles_labels()
        axs[0].legend(
            handles, labels, loc="best", fontsize=15
        )  # Augmente la taille de la légende
    plt.tight_layout()

    if fname:
        if save_path is not None:
            save_path = os.path.join(save_path, fname)
        else:
            save_path = os.path.join(os.environ["PROJECT_ROOT"], fname)
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
        print(f"figure saved as '{fname}' in {save_path}")

    plt.show()
    plt.close()
