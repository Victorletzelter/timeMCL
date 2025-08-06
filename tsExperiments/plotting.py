#%%

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
import os
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import time
import pickle
import rootutils
import sys
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
sys.path.append(os.path.dirname(os.environ["PROJECT_ROOT"]))

plt.ioff()
rc('text', usetex=True)
rc('font', family='serif')

#%%

def extract_unique_forecasts(samples, is_mcl=True):
    """Extract unique forecasts and their probabilities.
    
    Args:
        samples: Array of shape (n_hyp, forecast_length, target_dim) for TimeGrad
                or (n_hyp, forecast_length, target_dim+1) for timeMCL
        is_mcl: Whether the samples come from timeMCL model (with scores)
    """
    n_hyp = samples.shape[0]

    reshaped_forecasts = samples.reshape(n_hyp, -1)
    unique_forecasts, unique_indices = np.unique(reshaped_forecasts, axis=0, return_index=True)
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
        probabilities = np.ones_like(hypothesis_forecasts) / probabilities.sum(axis=0, keepdims=True)

    return hypothesis_forecasts, probabilities

def plot_mcl(target_df, hypothesis_forecasts, forecast_length, rows=4, cols=4, plot_mean=True, context_points=None, freq_type='H', fname='Predictions_plot.png', extract_unique=True, save_path=None, is_mcl=True, plot_p=True, main_color='lightcoral', mean_color='blue', dataset=None, axs=None, ax_cbar=None, global_min_prob=None, global_max_prob=None, dims_to_plot=None, seed=None):
    """
    Args:
        ... (existing args) ...
        is_mcl: Whether the forecasts come from timeMCL model (with scores) or TimeGrad
    """
    # Handle forecasts based on model type
    if is_mcl:
        # For timeMCL, extract unique forecasts if requested
        if extract_unique:
            hypothesis_forecasts, probabilities = extract_unique_forecasts(hypothesis_forecasts)
        else:
            # Extract scores for non-unique forecasts
            scores = hypothesis_forecasts[:,:,-1]
            hypothesis_forecasts = hypothesis_forecasts[:,:,:-1]
            probabilities = scores / scores.sum(axis=0, keepdims=True)
    else:
        # For TimeGrad, use equal probabilities for all samples
        probabilities = np.ones_like(hypothesis_forecasts) / hypothesis_forecasts.shape[0]

    # check PeriodIndex
    if isinstance(target_df.index, pd.PeriodIndex):
        target_df.index = target_df.index.to_timestamp()

    # entire data
    time_index = target_df.index
    values = target_df.values
    full_len, target_dim = values.shape

    k, fcst_len, d_ = hypothesis_forecasts.shape
    assert fcst_len == forecast_length, "forecast_length mismatch"

    # define how many context points we want before the 'end of training'
    #    if not given, default to 2 * forecast_length
    if context_points is None:
        context_points = 2 * forecast_length

    # suppose "end of training" is (full_len - forecast_length)
    #    i.e. the last training point is full_len - forecast_length - 1
    train_end_idx = full_len - forecast_length
    if train_end_idx < 0:
        # edge case: dataset too short vs forecast_length
        train_end_idx = full_len

    # we want from (train_end_idx - context_points) up to the end of the dataset
    start_idx = max(0, train_end_idx - context_points)

    # slice the data in that window
    tail_index = time_index[start_idx:]
    tail_values = values[start_idx:]
    if len(tail_values) == 0:
        print("WARNING: tail is empty!")
        return

    # freq inference
    freq = tail_index.freq
    if freq is None:
        freq = pd.infer_freq(tail_index)
    if freq is None and len(tail_index) > 1:
        freq = tail_index[-1] - tail_index[-2]
    if freq is None:
        freq = pd.Timedelta("1D")

    # last training timestamp
    last_training_ts = time_index[train_end_idx - 1] if train_end_idx > 0 else time_index[-1]

    # build future times from that point
    future_times = pd.date_range(
        start=last_training_ts + freq, 
        periods=forecast_length, 
        freq=freq
    )

    # mean forecast if asked
    mean_forecast = None
    if plot_mean:
        # weighted mean by the probabilities
        mean_forecast = np.sum(hypothesis_forecasts * probabilities, axis=0)  # shape (forecast_length, d)

    # Create figure with a special layout for the sidebar
    fig = plt.figure(figsize=(9*cols + 3, 5.5*rows))  # Added width for sidebar
    
    # Create GridSpec to manage subplot layout
    from matplotlib.gridspec import GridSpec
    width_ratios = []
    if cols > 1:
        for _ in range(cols-1):
            width_ratios.append(1)  # Original column width
            width_ratios.append(0.1)  # Blank column for spacing
    width_ratios.append(1)  # Colorbar column
    width_ratios.append(0.01) # Last blank column
    width_ratios.append(0.1)  # Colorbar column
    num_columns = cols * 2 + 1  # Original columns + blank columns + colorbar column
    gs = GridSpec(rows, num_columns, width_ratios=width_ratios, wspace=0)  # Adjust wspace as needed

    if axs is None:
        # Create main plot axes
        axs = []
        for i in range(rows):
            for j in range(cols):
                ax = fig.add_subplot(gs[i, j * 2])  # Use only the original columns
                axs.append(ax)
        axs = np.array(axs)
        

    if plot_p and ax_cbar is None:
        # Create sidebar axis
        ax_cbar= fig.add_subplot(gs[:, -1])

    # If dims_to_plot not specified, use first rows*cols dimensions
    if dims_to_plot is None:
        dims_to_plot = list(range(min(rows*cols, target_dim)))
    
    num_plots = len(dims_to_plot)
    
    # Apply style to each subplot
    for ax in axs:
        ax.set_facecolor('lightgrey')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.grid(True, which='both', linewidth=2.0, alpha=1.0, color='white')

    if plot_p and is_mcl:

        assert probabilities[0,0,0] == probabilities[0,0,1], "probabilities are not the same"
        assert probabilities[0,0,0] == probabilities[0,1,0], "probabilities are not the same"
        # print('probabilities here', probabilities.shape)
        # Use global min/max if provided, otherwise compute from current data
        if global_min_prob is not None and global_max_prob is not None:
            min_prob = global_min_prob
            max_prob = global_max_prob
        else:
            min_prob = min(probabilities[h_idx][0,0] for h_idx in range(len(hypothesis_forecasts)))
            max_prob = max(probabilities[h_idx][0,0] for h_idx in range(len(hypothesis_forecasts)))

        # Create a custom colormap that starts from a darker red
        if main_color == 'red':
            colors = plt.cm.Reds(np.linspace(0.4, 1, 256))  # Start from 0.3 instead of 0 to get darker minimum
        elif main_color == 'blue':
            colors = plt.cm.Blues(np.linspace(0.4, 1, 256))  # Start from 0.3 instead of 0 to get darker minimum
        elif main_color == 'green':
            colors = plt.cm.Greens(np.linspace(0.4, 1, 256))  # Start from 0.3 instead of 0 to get darker minimum
        custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_reds', colors)

        # Create a dummy mappable for the colorbar
        norm = mcolors.Normalize(vmin=min_prob, vmax=max_prob)
        sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
        sm.set_array([])

        # Add colorbar
        cbar = plt.colorbar(sm, cax=ax_cbar)

        cbar.ax.yaxis.set_tick_params(colors='white', labelsize=30)  # Change tick color
        cbar.ax.yaxis.set_ticks_position('right')

        # Get the width of the colorbar
        cbar_width = ax_cbar.get_window_extent().width
        
        # Calculate the padding to center the labels
        # The padding should be negative and approximately half the width of the colorbar
        offset = {
            "electricity": 19,
            "taxi": 10,
            "traffic": 17,
            "exchange": 10,
            "solar": 25,
        }
        print(dataset)
        pad = -cbar_width/2 - offset[dataset]  # Center based on actual text width

        cbar.ax.yaxis.set_tick_params(pad=pad, color='black', labelsize=30)  # Move labels inside

    prediction_start = future_times[0]  # Start of the prediction zone
    prediction_end = future_times[-1] # End of the prediction zone

    print('num_plots', num_plots)
    print('len axs', len(axs))

    for plot_idx, dim_idx in enumerate(dims_to_plot):
        is_last_row = (plot_idx % rows == rows-1)
        ax = axs[plot_idx]

        # Set the background color for the prediction zone
        ax.axvspan(prediction_start, prediction_end, color='lightyellow', alpha=0.5)  # Adjust color and alpha as needed

        # Entire tail of real data
        ax.plot(
            tail_index,
            tail_values[:, dim_idx],
            color="black",
            lw=3,
            label="Observations" if dim_idx == 0 else None
        )

        for h_idx in range(len(hypothesis_forecasts)):
            prob = probabilities[h_idx][0,0]
            if plot_p:
                # alpha = 0.8
                alpha = 0.5+prob*0.5  # Use probability as alpha if unique forecasts
                normalized_prob = (prob-min_prob)/(max_prob-min_prob)
                color = custom_cmap(normalized_prob)  # Use the same colormap as the colorbar
                ax.plot(
                future_times,
                hypothesis_forecasts[h_idx, :, dim_idx],
                lw=3,
                alpha=alpha,
                label="Predicted hypotheses" if (dim_idx == 0) and (h_idx == 0) else None,
                color=color
            )
            else:
                alpha = 0.8
                ax.plot(
                future_times,
                hypothesis_forecasts[h_idx, :, dim_idx],
                lw=3,
                alpha=alpha,
                label=f"Predicted hypotheses" if (dim_idx == 0) and (h_idx == 0) else None,
                color=main_color
            )
    
        # Optional mean
        if mean_forecast is not None:
            ax.plot(
                future_times,
                mean_forecast[:, dim_idx],
                color=mean_color,
                lw=3,
                linestyle='--',
                label="Predicted mean" if dim_idx == 0 else None
            )

        # For all rows, set the same grid pattern
        if freq_type == 'H':
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        elif freq_type == 'D':
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

        # Only show labels in the last row
        if is_last_row:
            if freq_type == 'H':
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax.xaxis.set_minor_formatter(mdates.DateFormatter('%d-%b'))
            elif freq_type == 'D':
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))

        ax.tick_params(axis="x", which="major", labelsize=20)
        ax.tick_params(axis="x", which="minor", labelsize=20, pad=15)
        ax.tick_params(axis="y", labelsize=20)
        ax.yaxis.label.set_size(25)

    handles, labels = axs[0].get_legend_handles_labels()

    if fname:
        fname = f'dims_{dims_to_plot}_seed_{seed}_{fname}'
        if save_path is not None :
            save_path = os.path.join(save_path, fname)
        else:
            save_path = os.path.join(os.environ["PROJECT_ROOT"], fname)
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
        plt.close()
        print(f"figure saved as '{fname}' in {save_path}")

    return handles, labels

def find_last_log_dir(dataset, model, num_hyps, suffix=None, seed=None):
    logdir = os.path.join(os.environ["PROJECT_ROOT"], "tsExperiments", "logs", f"visual_{dataset}", "runs")
    runs = os.listdir(logdir)
    runs = [run for run in runs if f"{model}_" in run and f"_{num_hyps}" in run]
    if seed is not None:
        runs = [run for run in runs if f"seed_{seed}" in run]
    if len(runs) == 0:
        print(f"No runs found for {dataset} {model} {num_hyps}")
        return None
    if suffix is not None:
        runs = [run for run in runs if suffix in run]
    def split_run_name(run_name):
        # Extrac
        if f'seed' in run_name:
            return time.strptime(run_name.split(f'seed')[0][:-1], '%Y-%m-%d_%H-%M-%S')
        else:
            # ignore the run name by setting an early date
            return time.strptime('1970-01-01_00-00-00', '%Y-%m-%d_%H-%M-%S')
    runs = sorted(runs, key=split_run_name)
    if len(runs) == 0:
        print(f"No runs found for {dataset} {model} {num_hyps} {suffix}")
        return None
    else:
        print(os.path.join(logdir, runs[-1]))
        return os.path.join(logdir, runs[-1])

def plot_from_logdir(logdir, fname, plot_p, main_color, mean_color, rows, cols, dataset, seed=None):

    # Load the data from pickle files
    try:    
        with open(f"{logdir}/context_points.pkl", "rb") as f:
            context_points = pickle.load(f)
    except:
        print(f"No context_points.pkl found in {logdir}")
        logdir = None
        return

    try:
        with open(f"{logdir}/forecast_length.pkl", "rb") as f:
            prediction_length = pickle.load(f)
    except:
        print(f"No forecast_length.pkl found in {logdir}")
        logdir = None
        return

    try:
        with open(f"{logdir}/target_df.pkl", "rb") as f:
            target_df = pickle.load(f)
    except:
        print(f"No target_df.pkl found in {logdir}")
        logdir = None
        return

    try:
        with open(f"{logdir}/hypothesis_forecasts.pkl", "rb") as f:
            hypothesis_forecasts = pickle.load(f)
    except:
        print(f"No hypothesis_forecasts.pkl found in {logdir}")
        logdir = None
        return

    try:
        with open(f"{logdir}/forecast_length.pkl", "rb") as f:
            forecast_length = pickle.load(f)
    except:
        print(f"No forecast_length.pkl found in {logdir}")
        logdir = None
        return

    try:
        with open(f"{logdir}/freq_type.pkl", "rb") as f:
            freq_type = pickle.load(f)
    except:
        print(f"No freq_type.pkl found in {logdir}")
        logdir = None
        return

    try:
        with open(f"{logdir}/is_mcl.pkl", "rb") as f:
            is_mcl = pickle.load(f)
    except:
        print(f"No is_mcl.pkl found in {logdir}")
        logdir = None
        return

    # Additional variables
    plot_mean = True
    extract_unique = is_mcl

    save_path = os.path.join(os.environ["PROJECT_ROOT"], "tsExperiments", "logs", "plots")
    # check if the dataset folder exists
    if not os.path.exists(os.path.join(save_path, dataset)):
        os.makedirs(os.path.join(save_path, dataset))
    save_path = os.path.join(save_path, dataset)

    if logdir is not None:  
        plot_mcl(target_df, hypothesis_forecasts, forecast_length, rows=rows, cols=1, plot_mean=plot_mean, context_points=context_points, freq_type=freq_type, extract_unique=extract_unique, save_path=save_path, is_mcl=is_mcl, plot_p=plot_p, main_color=main_color, mean_color=mean_color, dataset=dataset, seed=seed)

def plot_multiple_methods(dataset, methods, num_hyps, suffixes, rows=6, cols=3, dims_to_plot=None, seed=None):
    """
    Plot different methods side by side
    Args:
        dataset: str, name of the dataset
        methods: list of str, names of methods to plot (e.g. ['timeMCL', 'timeGrad', 'deepAR'])
        num_hyps: int, number of hypotheses
        suffixes: dict, mapping method names to their suffixes (e.g. {'timeMCL': ['amcl', 'relaxed']})
        rows: int, number of rows
        cols: int, number of columns (should match total number of methods including variants)
    """
    # First pass: find global min and max probabilities for MCL methods
    global_min_prob = float('inf')
    global_max_prob = float('-inf')

    for method in methods:
        if 'MCL' in method:
            print('method', method)
            if method in suffixes:
                method_suffixes = suffixes[method]
            else:
                method_suffixes = [None]
                
            for suffix in method_suffixes:
                logdir = find_last_log_dir(dataset, method, num_hyps, suffix, seed)
                if logdir is not None:
                    with open(f"{logdir}/hypothesis_forecasts.pkl", "rb") as f:
                        import pickle
                        hypothesis_forecasts = pickle.load(f)

                    hypothesis_forecasts, probabilities = extract_unique_forecasts(hypothesis_forecasts)

                    print('probabilities', probabilities.shape)
                    # Update global min and max
                    min_prob = min(probabilities[h_idx][0,0] for h_idx in range(len(hypothesis_forecasts)))
                    max_prob = max(probabilities[h_idx][0,0] for h_idx in range(len(hypothesis_forecasts)))
                    global_min_prob = min(global_min_prob, min_prob)
                    global_max_prob = max(global_max_prob, max_prob)

    # Create figure with a special layout for the sidebar
    fig = plt.figure(figsize=(9*cols + 3, 5.5*rows))
    
    # Create GridSpec to manage subplot layout
    from matplotlib.gridspec import GridSpec
    width_ratios = []
    for _ in range(cols-1):
        width_ratios.append(1)  # Original column width
        width_ratios.append(0.1)  # Blank column for spacing
    width_ratios.append(1)  # Last column
    width_ratios.append(0.01)  # Last blank column
    width_ratios.append(0.1)  # Colorbar column
    
    num_columns = cols * 2 + 1  # Original columns + blank columns + colorbar column
    gs = GridSpec(rows+1, num_columns, width_ratios=width_ratios, wspace=0, height_ratios=[0.05] + [1]*rows)

    # Create main plot axes
    axs = []
    for i in range(rows):
        for j in range(cols):
            ax = fig.add_subplot(gs[i+1, j * 2])
            axs.append(ax)
    axs = np.array(axs)

    # Create title axes
    title_axes = []
    for j in range(cols):
        ax_title = fig.add_subplot(gs[0, j * 2])
        ax_title.axis('off')  # Hide the axis
        title_axes.append(ax_title)

    def extract_column(axs, col_idx):
        indexes = np.arange(col_idx, axs.shape[0], cols)
        return axs[indexes]

    # Apply style to each subplot
    for ax in axs:
        ax.set_facecolor('lightgrey')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.grid(True, which='both', linewidth=2.0, alpha=1.0, color='white')
        ax.set_xticks([])
        ax.set_xticklabels([])

    # Store all legend handles and labels
    all_handles = []
    all_labels = []

    # Plot each method
    col_idx = 0
    for method in methods:
        if method in suffixes:
            method_suffixes = suffixes[method]
        else:
            method_suffixes = [None]
            
        for suffix in method_suffixes:

             # Add title for this column
            if 'deepVAR' in method:
                title = r"$\texttt{DeepAR}$"
            elif 'timeMCL' in method:
                title = r"$\texttt{TimeMCL}$"
            elif 'timeGrad' in method:
                title = r"$\texttt{TimeGrad}$"
            elif 'tempflow' in method:
                title = r"$\texttt{TempFlow}$"
            elif 'tactis2' in method:
                title = r"$\texttt{Tactis2}$"
            else:
                title = r"$\texttt{" + method + "}$"
            if suffix:
                if suffix == 'amcl':
                    # suffix_to_plot = 'annealed'
                    suffix_to_plot = r"$\texttt{annealed}$"
                elif suffix == 'relaxed':
                    suffix_to_plot = r"$\texttt{relaxed}$"
                else:
                    suffix_to_plot = suffix
                # title += f"\n{suffix_to_plot}"
                title += f" ({suffix_to_plot})"
            print('col_idx', col_idx)
            title_axes[col_idx].text(0.5, 0.5, title, 
                                   horizontalalignment='center',
                                   verticalalignment='center',
                                   fontsize=65,
                                   fontweight='bold')

            print('method', method)

            # Set colors based on method
            if 'MCL' in method:
                main_color = 'blue'
                mean_color = 'blue'
                is_mcl = True
            elif 'timeGrad' in method:
                main_color = 'lightcoral'
                mean_color = 'red'
                is_mcl = False
            elif 'deep' in method:
                main_color = 'lightgreen'
                mean_color = 'green'
                is_mcl = False
            elif 'tempflow' in method:
                # main_color = 'navajowhite'
                main_color = '#FCD299'
                mean_color = 'orange'
                is_mcl = False
            elif 'tactis2' in method:
                main_color = '#C5B4E3'
                mean_color = 'purple'
                is_mcl = False
            # Find logdir
            logdir = find_last_log_dir(dataset, method, num_hyps, suffix, seed)

            print("logdir", logdir)
            
            import pickle

            if logdir is not None:
                # Load data
                with open(f"{logdir}/target_df.pkl", "rb") as f:
                    target_df = pickle.load(f)
                with open(f"{logdir}/hypothesis_forecasts.pkl", "rb") as f:
                    hypothesis_forecasts = pickle.load(f)
                with open(f"{logdir}/forecast_length.pkl", "rb") as f:
                    forecast_length = pickle.load(f)
                with open(f"{logdir}/context_points.pkl", "rb") as f:
                    context_points = pickle.load(f)
                with open(f"{logdir}/freq_type.pkl", "rb") as f:
                    freq_type = pickle.load(f)

                ax_cbar = None
                if method=='timeMCL':
                    ax_cbar = fig.add_subplot(gs[1:, -1])

                # Plot for this method using the corresponding column's axes
                axs_col = extract_column(axs, col_idx)
                handles, labels = plot_method_column(
                    axs=axs_col,
                    target_df=target_df,
                    hypothesis_forecasts=hypothesis_forecasts,
                    forecast_length=forecast_length,
                    context_points=context_points,
                    freq_type=freq_type,
                    is_mcl=is_mcl,
                    main_color=main_color,
                    mean_color=mean_color,
                    plot_p=is_mcl,
                    ax_cbar=ax_cbar if method=='timeMCL' else None,
                    global_min_prob=global_min_prob if is_mcl else None,
                    global_max_prob=global_max_prob if is_mcl else None,
                    rows=rows,
                    cols=cols,
                    dims_to_plot=dims_to_plot
                )

                # Add method name to labels if needed

                if 'deepVAR' in method:
                    title = r"$\texttt{DeepAR}$"
                elif 'timeMCL' in method:
                    title = r"$\texttt{TimeMCL}$"
                elif 'timeGrad' in method:
                    title = r"$\texttt{TimeGrad}$"
                elif 'tempflow' in method:
                    title = r"$\texttt{TempFlow}$"
                elif 'tactis2' in method:
                    title = r"$\texttt{Tactis2}$"
                else:
                    title = f"{method}"

                new_labels = []
                for label in labels:
                    if "Observations" in label:
                        new_labels.append(label)
                    else:
                        if 'Predicted' in label:
                            label = label.replace('Predicted ','')
                        new_labels.append(f"{title} {label}")

                # Remove duplicates from new_labels
                new_labels = list(dict.fromkeys(new_labels))
                labels = new_labels
                all_handles.extend(handles)
                all_labels.extend(labels)

            col_idx += 1

    # Remove duplicates from all_labels, and update all_handles
    # Extract indicices associated to each unique label
    unique_labels = list(dict.fromkeys(all_labels))
    unique_indices = [all_labels.index(label) for label in unique_labels]
    all_handles = [all_handles[i] for i in unique_indices]
    all_labels = [all_labels[i] for i in unique_indices]
    # Create a single legend at the top of the figure
    if len(dims_to_plot) == 3:
        bbox_to_anchor=(0.5, 1.08)
    elif len(dims_to_plot) > 3:
        bbox_to_anchor=(0.5, 1.05)

    ncol = max(1, len(all_handles))
    fig.legend(all_handles, all_labels, 
              loc='upper center', 
              bbox_to_anchor=bbox_to_anchor,
              ncol=ncol,  # Adjust number of columns as needed
            fontsize=27
              )
    
    plt.tight_layout()

    # Save the combined figure
    save_path = f"{os.environ['PROJECT_ROOT']}/tsExperiments/logs/plots/{dataset}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fig.savefig(os.path.join(save_path, f'combined_plot_{dataset}_{num_hyps}_rows_{rows}_cols_{cols}_dims_{dims_to_plot}_seed_{seed}.png'), 
                bbox_inches="tight", 
                pad_inches=0.05)
    print('Figure saved in ', os.path.join(save_path, f'combined_plot_{dataset}_{num_hyps}_rows_{rows}_cols_{cols}_dims_{dims_to_plot}_seed_{seed}.png'))
    plt.close()

def plot_method_column(axs, target_df, hypothesis_forecasts, forecast_length, context_points, 
                      freq_type, is_mcl, main_color, mean_color, plot_p=False, ax_cbar=None, extract_unique=True, global_min_prob=None, global_max_prob=None, rows=None, cols=None, dims_to_plot=None):
    """
    Plot a single method's results in one column
    """
    save_path = f"{os.environ['PROJECT_ROOT']}/tsExperiments/logs/plots/{dataset}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    handles, labels = plot_mcl(target_df, hypothesis_forecasts, forecast_length, rows=rows, cols=1, plot_mean=True, context_points=context_points, freq_type=freq_type, extract_unique=extract_unique, save_path=save_path, is_mcl=is_mcl, plot_p=plot_p, main_color=main_color, mean_color=mean_color, dataset=dataset, axs=axs, ax_cbar=ax_cbar, global_min_prob=global_min_prob, global_max_prob=global_max_prob, dims_to_plot=dims_to_plot)

    return handles, labels

# Usage example:
methods = ['tactis2', 'timeGrad', 'tempflow','timeMCL']
suffixes = {'timeMCL': ['amcl', 'relaxed']}  # Only timeMCL has suffixes

num_hyps = 8

dim_to_plot_to_plot = {
    'solar': [0,2,5],
    'traffic': [0,4,8,13],
    'electricity': [0,3,4,5],
}

suffixes_to_plot = {
    'solar': {'timeMCL': ['amcl', 'relaxed']},
    'traffic': {'timeMCL': ['relaxed']},
    'electricity': {'timeMCL': ['relaxed']},
}

num_hyps_to_plot = {
    'solar': 16,
    'traffic': 8,
    'electricity': 8,
}

seed=42

for dataset in ['solar', 'traffic', 'electricity']:

    num_hyps = num_hyps_to_plot[dataset]
    suffixes = suffixes_to_plot[dataset]
    dims_to_plot = dim_to_plot_to_plot[dataset]

    rows = len(dims_to_plot)
    cols = len(methods) if dataset != 'solar' else len(methods) + 1

    plot_multiple_methods(dataset, methods, num_hyps, suffixes, rows=rows, cols=cols, dims_to_plot=dims_to_plot, seed=seed)