# %% Imports and Setup
import os
import pandas as pd
import numpy as np
import sys
import warnings
import rootutils

# Display and warning settings
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
np.seterr(all="ignore")
# Add the project root to the path
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
sys.path.append(os.path.dirname(os.environ["PROJECT_ROOT"]))

# %% Configuration
datasets_list = ["electricity", "exchange", "solar", "traffic", "taxi", "wiki"]
# datasets_list = ["solar"]
num_hypothesis_list = ["1", "2", "3", "4", "5", "8", "16"]
config_name_list = [
    "timeGrad",
    "deepAR",
    "tempflow",
    "transformer_tempflow",
    "tactis2",
    "MCLrelaxed0.1mean",
    "aMCL0.95mean",
]
# Metric names
metric_name_rmse = "m_sum_RMSE"
metric_name_risk = "Distorsion"
metric_name_CRPS_sum = "m_sum_mean_wQuantileLoss"
metric_name_tv = "total_variation"
show_std = True

# %% Helper Functions
def configuration_to_extract(
    csv_file, config_name, num_hypothesis, keep_single_seed=False
):
    """Filter and extract rows from csv_file based on config and hypothesis count."""

    csv_file = csv_file[
        (csv_file["model/params/num_hypotheses"] == float(num_hypothesis))
        | (csv_file["model/params/num_hypotheses"] == num_hypothesis)
    ]

    # Check seeds
    seed_mask = csv_file["Name"].str.contains(f"seed_{keep_single_seed}_", na=False)

    # Combine both masks
    if keep_single_seed is not None:
        csv_file = csv_file[seed_mask]

    if "MCL" in config_name:
        csv_file = csv_file[
            (csv_file["model/params/score_loss_weight"] == 0.5)
            | (csv_file["model/params/score_loss_weight"] == "0.5")
        ]

    if config_name == "MCLrelaxed0.1mean":
        csv_file = csv_file[csv_file["model/name"] == "timeMCL"]
        csv_file = csv_file[csv_file["model/params/scaler_type"] == "mean"]
        csv_file = csv_file[csv_file["model/params/wta_mode"] == "relaxed-wta"]

    elif config_name == "aMCL0.95mean":
        csv_file = csv_file[csv_file["model/name"] == "timeMCL"]
        csv_file = csv_file[csv_file["model/params/wta_mode"] == "awta"]

    elif config_name == "timeGrad":
        csv_file = csv_file[csv_file["model/name"] == "timeGrad"]

    elif config_name == "deepAR":
        csv_file = csv_file[csv_file["model/name"] == "deepAR"]

    elif config_name == "tempflow":
        csv_file = csv_file[csv_file["model/name"] == "tempflow"]

    elif config_name == "transformer_tempflow":
        csv_file = csv_file[csv_file["model/name"] == "transformer_tempflow"]

    elif config_name == "tactis2":
        csv_file = csv_file[csv_file["model/name"] == "tactis2"]

    elif config_name == "ETS":
        csv_file = csv_file[csv_file["model/name"] == "ETS"]

    else:
        raise ValueError("Invalid config name")

    return csv_file


def look_exact_path(path):
    """Find the exact file path in the directory matching the given path pattern."""
    csv_dir = "saved_csv"
    Root_DIR = os.path.dirname(path)
    for file in os.listdir(Root_DIR):
        if file.split("_id")[0] == path.split("/" + csv_dir + "/")[-1].split(".csv")[0]:
            return os.path.join(Root_DIR, file)


def set_lowest_and_second_lowest_in_bold_and_underline(df, lower_is_better=True):
    """Format DataFrame to bold the best and underline the second best values per row."""
    for index, row in df.iterrows():
        values = []
        method_value_map = {}
        for col in df.columns:
            cell = row[col]
            if isinstance(cell, str):
                try:
                    value = float(cell.split(" ")[0])
                    values.append(value)
                    method_value_map[col] = value
                except ValueError:
                    continue

        if not values:
            continue

        sorted_values = sorted(set(values), reverse=not lower_is_better)
        best_val = sorted_values[0]
        second_best_val = sorted_values[1] if len(sorted_values) > 1 else None

        for col in df.columns:
            cell = row[col]
            if isinstance(cell, str):
                try:
                    value = float(cell.split(" ")[0])
                    if value == best_val:
                        df.at[index, col] = f"\\textbf{{{cell}}}"
                    elif second_best_val is not None and value == second_best_val:
                        df.at[index, col] = f"\\underline{{{cell}}}"
                except ValueError:
                    continue

    return df


def resize_table(str_table, dataset_name, metric_name, caption):
    """Wrap a LaTeX table string with formatting, caption, and label."""
    mapping_metric_name = {
        metric_name_CRPS_sum: "CRPS-Sum",
        metric_name_rmse: "RMSE-Sum",
        metric_name_risk: "Distortion Risk",
        metric_name_tv: "TV Score",
    }
    metric_name = mapping_metric_name[metric_name]

    if dataset_name is None:
        label = metric_name
    elif metric_name is None:
        label = dataset_name
    else:
        label = f"{dataset_name} - {metric_name}"

    # Add a textbf around the caption
    caption = f"\\textbf{{{caption}}}"

    latex_wrapper = f"""\\begin{{table}}
    \\begin{{center}}
    \\caption{{{caption}}}
    \\label{{tabapx:{label}}}
    \\resizebox{{\\columnwidth}}{{!}}{{{str_table}}}
    \\end{{center}}
    \\end{{table}}"""

    return latex_wrapper


def create_fixed_hypothesis_metric_table(
    metric_df_dict,
    metric_name,
    fixed_num_hypothesis,
    datasets,
    methods,
    insert_vertical_line=False,
    lower_is_better=False,
    gray_first_two_columns=False,
):
    """Generate a LaTeX table for a metric with a fixed number of hypotheses across datasets and methods."""
    dataset_to_render = {
        "electricity": "\\textsc{Elec.}",
        "exchange": "\\textsc{Exch.}",
        "solar": "\\textsc{Solar}",
        "traffic": "\\textsc{Traffic}",
        "taxi": "\\textsc{Taxi}",
        "wiki": "\\textsc{Wiki}",
    }
    datasets_to_render_dict = {
        "electricity": "\\textsc{Elec.}",
        "exchange": "\\textsc{Exch.}",
        "solar": "\\textsc{Solar}",
        "traffic": "\\textsc{Traffic}",
        "taxi": "\\textsc{Taxi}",
        "wiki": "\\textsc{Wiki}",
    }
    methods_to_render_dict = {
        "ETS": "\\textbf{\\texttt{ETS}}",
        "timeGrad": "\\textbf{\\texttt{TimeGrad}}",
        "deepAR": "\\textbf{\\texttt{DeepAR}}",
        "tempflow": "\\textbf{\\texttt{TempFlow}}",
        "transformer_tempflow": "\\textbf{\\texttt{Trf.TempFlow}}",
        "tactis2": "\\textbf{\\texttt{Tactis2}}",
        "MCLrelaxed0.1mean": "\\textbf{\\texttt{TimeMCL(R.)}}",
        "aMCL0.95mean": "\\textbf{\\texttt{TimeMCL(A.)}}",
        "timeMCL_relaxed-wta": "\\textbf{\\texttt{TimeMCL(R.)}}",
        "timeMCL_awta": "\\textbf{\\texttt{TimeMCL(A.)}}",
    }
    datasets_to_render = [dataset_to_render[dataset] for dataset in datasets]
    methods_to_render = [methods_to_render_dict[method] for method in methods]

    table_data = pd.DataFrame(index=datasets_to_render, columns=methods_to_render)

    for dataset in datasets:
        for method in methods:
            try:
                value = metric_df_dict[dataset].loc[fixed_num_hypothesis, method]
                if type(value) == str and "nan" in value.split(" $\pm$")[0]:
                    table_data.at[
                        datasets_to_render_dict[dataset], methods_to_render_dict[method]
                    ] = "N/A"
                else:
                    table_data.at[
                        datasets_to_render_dict[dataset], methods_to_render_dict[method]
                    ] = value
            except KeyError:
                table_data.at[
                    datasets_to_render_dict[dataset], methods_to_render_dict[method]
                ] = "N/A"

    table_data = set_lowest_and_second_lowest_in_bold_and_underline(
        table_data, lower_is_better=lower_is_better
    )
    latex_table = table_data.to_latex(escape=False, column_format="c" * (len(methods)))

    if insert_vertical_line:
        my_method = "MCLrelaxed0.1mean"
        if my_method in methods:
            method_index = methods.index(my_method)
            col_format = (
                "l" + "c" * (method_index) + "||" + "c" * (len(methods) - method_index)
            )
            # Match the exact pattern including newline and replace with new format
            latex_table = latex_table.replace(
                "\\begin{tabular}{" + "c" * len(methods) + "}\n",
                "\\begin{tabular}{" + col_format + "}\n",
            )

    # Add gray coloring to first two columns if requested
    if gray_first_two_columns:
        # Split the table into lines
        lines = latex_table.split("\n")
        for i, line in enumerate(lines):
            if "&" in line:  # Only process data rows
                parts = line.split("&")
                # Wrap first two columns in \textcolor{gray}{}
                parts[1] = f"\\textcolor{{gray}}{{{parts[1].strip()}}}"
                parts[2] = f"\\textcolor{{gray}}{{{parts[2].strip()}}}"
                parts[3] = f"\\textcolor{{gray}}{{{parts[3].strip()}}}"
                lines[i] = " & ".join(parts)
        latex_table = "\n".join(lines)

    # Add arrow to caption based on lower_is_better
    arrow = "($\\downarrow$)" if lower_is_better else "($\\uparrow$)"
    caption = (
        f"{metric_name} {arrow} comparison for $K = {fixed_num_hypothesis}$ hypotheses"
    )

    return resize_table(
        latex_table, dataset_name=None, metric_name=metric_name, caption=caption
    )


# %% Initialize DataFrames
def initialize_dataframes(datasets_list, config_name_list, num_hypothesis_list):
    """Initialize DataFrames for the metrics."""
    datasets = {}
    df_rmse, df_risk, df_CRPS_sum, df_tv = {}, {}, {}, {}

    for dataset_name in datasets_list:
        df_rmse[dataset_name] = pd.DataFrame(
            index=num_hypothesis_list, columns=config_name_list
        )
        df_risk[dataset_name] = pd.DataFrame(
            index=num_hypothesis_list, columns=config_name_list
        )
        df_CRPS_sum[dataset_name] = pd.DataFrame(
            index=num_hypothesis_list, columns=config_name_list
        )
        df_tv[dataset_name] = pd.DataFrame(
            index=num_hypothesis_list, columns=config_name_list
        )

    return df_rmse, df_risk, df_CRPS_sum, df_tv, datasets

df_rmse, df_risk, df_CRPS_sum, df_tv, datasets = initialize_dataframes(datasets_list, config_name_list, num_hypothesis_list)

def BASE_PATH_generator(dataset_name, config_name):
    """Generate the base path for results CSV based on dataset and config."""
    return [
        f"{os.environ['PROJECT_ROOT']}/tsExperiments/results/saved_csv/eval_{dataset_name}_200.csv"
    ]

def create_single_dataset_table(
    metric_df_dict,
    metric_name,
    dataset_name,
    methods,
    num_hypotheses_list,
    insert_vertical_line=False,
    gray_first_two_columns=False,
):
    """Generate a LaTeX table for a single dataset with varying numbers of hypotheses."""
    methods_to_render_dict = {
        "ETS": "\\textbf{\\texttt{ETS}}",
        "timeGrad": "\\textbf{\\texttt{TimeGrad}}",
        "deepAR": "\\textbf{\\texttt{DeepAR}}",
        "tempflow": "\\textbf{\\texttt{TempFlow}}",
        "transformer_tempflow": "\\textbf{\\texttt{Trf.TempFlow}}",
        "tactis2": "\\textbf{\\texttt{Tactis2}}",
        "MCLrelaxed0.1mean": "\\textbf{\\texttt{TimeMCL (R.)}}",
        "aMCL0.95mean": "\\textbf{\\texttt{TimeMCL (A.)}}",
    }

    methods_to_render = [methods_to_render_dict[method] for method in methods]

    # Create DataFrame with hypotheses as index and methods as columns
    table_data = pd.DataFrame(index=num_hypotheses_list, columns=methods_to_render)

    # Fill the table with values
    for num_hyp in num_hypotheses_list:
        for method in methods:
            try:
                value = metric_df_dict[dataset_name].loc[num_hyp, method]
                if type(value) == str and "nan" in value.split(" $\pm$")[0]:
                    table_data.at[num_hyp, methods_to_render_dict[method]] = "N/A"
                else:
                    table_data.at[num_hyp, methods_to_render_dict[method]] = value
            except KeyError:
                table_data.at[num_hyp, methods_to_render_dict[method]] = "N/A"

    # Format the table
    table_data = set_lowest_and_second_lowest_in_bold_and_underline(table_data)
    latex_table = table_data.to_latex(escape=False)

    if insert_vertical_line:
        my_method = "MCLrelaxed0.1mean"
        if my_method in methods:
            method_index = methods.index(my_method) + 1
            col_format = (
                "l"
                + "c" * (method_index - 1)
                + "|"
                + "c" * (len(methods) - method_index)
            )
            latex_table = latex_table.replace(
                r"\begin{tabular}{", r"\begin{tabular}{" + col_format + "}"
            )

    caption = f"{metric_name} Comparison for {dataset_name} Dataset"

    if gray_first_two_columns:
        # Split the table into lines
        lines = latex_table.split("\n")
        for i, line in enumerate(lines):
            if "&" in line:  # Only process data rows
                parts = line.split("&")
                # Wrap first two columns in \textcolor{gray}{}
                parts[1] = f"\\textcolor{{gray}}{{{parts[1].strip()}}}"
                parts[2] = f"\\textcolor{{gray}}{{{parts[2].strip()}}}"
                parts[3] = f"\\textcolor{{gray}}{{{parts[3].strip()}}}"
                lines[i] = " & ".join(parts)
        latex_table = "\n".join(lines)

    return resize_table(
        latex_table, dataset_name=dataset_name, metric_name=metric_name, caption=caption
    )


def extract_data(
    datasets,
    df_rmse,
    df_risk,
    df_CRPS_sum,
    df_tv,
    small_std_font,
    show_std,
    keep_single_seed,
    datasets_list,
    config_name_list,
    num_hypothesis_list,
    metric_name_rmse,
    metric_name_risk,
    metric_name_CRPS_sum,
    metric_name_tv,
):
    """Extract the data from the csv files and store it in the DataFrames."""
    round_per_dataset_tv = True

    for dataset_name in datasets_list:
        for config_name in config_name_list:
            for num_hypothesis in num_hypothesis_list:

                BASE_PATHS = BASE_PATH_generator(dataset_name, config_name)

                csv_file = pd.DataFrame()
                for BASE_PATH in BASE_PATHS:
                    BASE_PATH = look_exact_path(BASE_PATH)
                    csv_file_path = pd.read_csv(BASE_PATH)
                    csv_file = pd.concat([csv_file, csv_file_path])

                csv_file_num_hyps = configuration_to_extract(
                    csv_file, config_name, num_hypothesis, keep_single_seed
                )

                # Check for missing columns
                required_columns = [
                    metric_name_rmse,
                    metric_name_risk,
                    metric_name_CRPS_sum,
                ]
                missing_columns = [
                    col
                    for col in required_columns
                    if col not in csv_file_num_hyps.columns
                ]
                if missing_columns:
                    continue  # Skip this configuration if columns are missing

                if len(csv_file_num_hyps) == 0:
                    continue

                # Keep only one run per seed, i.e., keep only runs with distinct csv_file_num_hyps[['Name']] (e.g, seed_3143_exchange_tactis2_16, seed_3142_exchange_tactis2_16).
                # For a given 'Name', keep the most recent run (based on csv_file_num_hyps[['_start_time']], like 2025-04-07 21:07:56.873)
                csv_file_num_hyps = csv_file_num_hyps.sort_values(
                    by="_start_time", ascending=False
                )
                csv_file_num_hyps = csv_file_num_hyps.drop_duplicates(subset=["Name"])

                csv_file_num_hyps["ext_seed"] = csv_file_num_hyps["Name"].str.extract(
                    r"seed_(\d+)"
                )
                # Keep only one run per seed (when the rest of the config is fixed)
                # For each seed (and config), keep if there is more than one run, if yes, keep the most recent run
                csv_file_num_hyps = (
                    csv_file_num_hyps.groupby("ext_seed")
                    .apply(
                        lambda x: x.sort_values(by="_start_time", ascending=False).head(
                            1
                        )
                    )
                    .reset_index(drop=True)
                )

                agg_results = csv_file_num_hyps.groupby(by=["model/name"]).agg(
                    {
                        metric_name_rmse: ["mean", "std", "count"],
                        metric_name_risk: ["mean", "std"],
                        metric_name_CRPS_sum: ["mean", "std"],
                        metric_name_tv: ["mean", "std"],
                    }
                )

                datasets[dataset_name] = agg_results

                # Check if the number of runs for the std (to make sure we have 4 seeds for the std computation)
                if (
                    agg_results[(metric_name_rmse, "count")].values[0] != 0
                    and agg_results[(metric_name_rmse, "count")].values[0] != 4
                ):
                    print(csv_file_num_hyps[["Name"]])
                    print(
                        "Invalid number of runs for the std with {}, {}, {}: number of runs is {}".format(
                            config_name,
                            dataset_name,
                            num_hypothesis,
                            agg_results[(metric_name_rmse, "count")].values[0],
                        )
                    )

                # Extract the mean and std of the metrics
                mean_rmse = agg_results[(metric_name_rmse, "mean")].values[0]
                mean_risk = agg_results[(metric_name_risk, "mean")].values[0]
                mean_CRPS_sum = agg_results[(metric_name_CRPS_sum, "mean")].values[0]
                mean_tv = agg_results[(metric_name_tv, "mean")].values[0]

                std_rmse = agg_results[(metric_name_rmse, "std")].values[0]
                std_risk = agg_results[(metric_name_risk, "std")].values[0]
                std_CRPS_sum = agg_results[(metric_name_CRPS_sum, "std")].values[0]
                std_tv = agg_results[(metric_name_tv, "std")].values[0]

                # Round the metrics to the appropriate number of decimal places for better readability in the tables
                if round_per_dataset_tv is True:
                    if (
                        dataset_name == "electricity"
                        or dataset_name == "solar"
                        or dataset_name == "wiki"
                    ):
                        mean_tv = mean_tv.astype(int)
                        std_tv = std_tv.astype(int)
                    elif dataset_name == "exchange":
                        mean_tv = mean_tv.round(3)
                        std_tv = std_tv.round(4)
                    elif dataset_name == "traffic":
                        mean_tv = mean_tv.round(3)
                        std_tv = std_tv.round(3)
                    elif dataset_name == "taxi":
                        mean_tv = mean_tv.round(2)
                        std_tv = std_tv.round(2)

                if type(mean_CRPS_sum) == np.float64 and not np.isnan(mean_CRPS_sum):
                    mean_CRPS_sum = mean_CRPS_sum.round(4)
                if type(std_CRPS_sum) == np.float64 and not np.isnan(std_CRPS_sum):
                    std_CRPS_sum = std_CRPS_sum.round(4)
                if type(mean_tv) == np.float64 and not np.isnan(mean_tv):
                    mean_tv = mean_tv.round(4)
                if type(std_tv) == np.float64 and not np.isnan(std_tv):
                    std_tv = std_tv.round(4)

                if dataset_name == "exchange":
                    mean_risk, mean_rmse = mean_risk.round(3), mean_rmse.round(3)
                    if type(std_risk) == np.float64 and not np.isnan(std_risk):
                        std_risk, std_rmse = std_risk.round(3), std_rmse.round(3)
                elif dataset_name == "wiki" or dataset_name == "electricity":
                    if type(mean_risk) == np.float64 and not np.isnan(mean_risk):
                        mean_risk, mean_rmse = mean_risk.round(0).astype(
                            int
                        ), mean_rmse.round(0).astype(int)
                    if type(std_risk) == np.float64 and not np.isnan(std_risk):
                        std_risk, std_rmse = std_risk.round(0).astype(
                            int
                        ), std_rmse.round(0).astype(int)
                else:
                    if type(mean_risk) == np.float64 and not np.isnan(mean_risk):
                        mean_risk, mean_rmse = mean_risk.round(2), mean_rmse.round(2)

                    if type(std_risk) == np.float64 and not np.isnan(std_risk):
                        std_risk, std_rmse = std_risk.round(2), std_rmse.round(2)

                # Format the metrics for the table (depending on the show_std, and the small_std_font option)
                if show_std is True:
                    if small_std_font:
                        rmse_result = (
                            f"{mean_rmse}"
                            + " \\scriptsize{{$\\pm$ {}}}".format(std_rmse)
                        )
                        risk_result = (
                            f"{mean_risk}"
                            + " \\scriptsize{{$\\pm$ {}}}".format(std_risk)
                        )
                        CRPS_sum_result = (
                            f"{mean_CRPS_sum}"
                            + " \\scriptsize{{$\\pm$ {}}}".format(std_CRPS_sum)
                        )
                        tv_result = f"{mean_tv}" + " \\scriptsize{{$\\pm$ {}}}".format(
                            std_tv
                        )
                    else:
                        rmse_result = f"{mean_rmse}" + " $\\pm$ {}".format(std_rmse)
                        risk_result = f"{mean_risk}" + " $\\pm$ {}".format(std_risk)
                        CRPS_sum_result = f"{mean_CRPS_sum}" + " $\\pm$ {}".format(
                            std_CRPS_sum
                        )
                        tv_result = f"{mean_tv}" + " $\\pm$ {}".format(std_tv)
                else:
                    rmse_result = f"{mean_rmse}"
                    risk_result = f"{mean_risk}"
                    CRPS_sum_result = f"{mean_CRPS_sum}"
                    tv_result = f"{mean_tv}"

                # Store the results in the corresponding DataFrames
                df_rmse[dataset_name].loc[num_hypothesis, config_name] = rmse_result
                df_risk[dataset_name].loc[num_hypothesis, config_name] = risk_result
                df_CRPS_sum[dataset_name].loc[
                    num_hypothesis, config_name
                ] = CRPS_sum_result
                df_tv[dataset_name].loc[num_hypothesis, config_name] = tv_result

    return df_rmse, df_risk, df_CRPS_sum, df_tv


def reproduce_tables(plot_additional_tables=False):
    """Create LaTeX tables from the results as csv files, and save them in a txt file."""
    show_std = True  # Show the std in the table
    keep_single_seed = None  # Set the string of the seed to keep only one run per seed
    datasets_list = ["electricity", "exchange", "solar", "traffic", "taxi", "wiki"]
    num_hypothesis_list = ["16"]
    config_name_list = [
        "ETS",
        "transformer_tempflow",
        "tactis2",
        "timeGrad",
        "deepAR",
        "tempflow",
        "MCLrelaxed0.1mean",
        "aMCL0.95mean",
    ]
    # Metric names
    metric_name_rmse = "m_sum_RMSE"
    metric_name_risk = "Distorsion"
    metric_name_CRPS_sum = "m_sum_mean_wQuantileLoss"
    metric_name_tv = "total_variation"
    show_std = True
    lower_is_better_dict = {
        metric_name_risk: True,
        metric_name_rmse: True,
        metric_name_CRPS_sum: True,
        metric_name_tv: True,
    }

    # Open output file for writing all tables
    with open("latex_tables_output.txt", "w") as f:
        ### Show to the user the tables that are being generated
        print("Generating tables...")
        # Table 1: Distortion Comparison (16 hyps)
        fixed_num_hypothesis = "16"
        fixed_metric = metric_name_risk
        df_rmse, df_risk, df_CRPS_sum, df_tv, datasets = initialize_dataframes(datasets_list, config_name_list, num_hypothesis_list)
        df_rmse, df_risk, df_CRPS_sum, df_tv = extract_data(
            datasets=datasets,
            df_rmse=df_rmse,
            df_risk=df_risk,
            df_CRPS_sum=df_CRPS_sum,
            df_tv=df_tv,
            small_std_font=False,
            keep_single_seed=keep_single_seed,
            datasets_list=datasets_list,
            show_std=show_std,
            config_name_list=config_name_list,
            num_hypothesis_list=num_hypothesis_list,
            metric_name_rmse=metric_name_rmse,
            metric_name_risk=metric_name_risk,
            metric_name_CRPS_sum=metric_name_CRPS_sum,
            metric_name_tv=metric_name_tv,
        )
        df = {
            metric_name_risk: df_risk,
            metric_name_rmse: df_rmse,
            metric_name_CRPS_sum: df_CRPS_sum,
            metric_name_tv: df_tv,
        }

        latex_distortion_table = create_fixed_hypothesis_metric_table(
            metric_df_dict=df[fixed_metric],
            metric_name=fixed_metric,
            fixed_num_hypothesis=fixed_num_hypothesis,
            datasets=datasets_list,
            methods=config_name_list,
            insert_vertical_line=True,
            lower_is_better=lower_is_better_dict[fixed_metric],
            gray_first_two_columns=True,
        )
        f.write(latex_distortion_table + "\n\n")

        # Table 3: Total Variation Comparison (16 hyps)
        config_name_list = [
            "tactis2",
            "timeGrad",
            "tempflow",
            "MCLrelaxed0.1mean",
            "aMCL0.95mean",
        ]
        fixed_metric = metric_name_tv
        df_rmse, df_risk, df_CRPS_sum, df_tv, datasets = initialize_dataframes(datasets_list, config_name_list, num_hypothesis_list)
        df_rmse, df_risk, df_CRPS_sum, df_tv = extract_data(
            datasets=datasets,
            df_rmse=df_rmse,
            df_risk=df_risk,
            df_CRPS_sum=df_CRPS_sum,
            df_tv=df_tv,
            small_std_font=True,
            keep_single_seed=keep_single_seed,
            datasets_list=datasets_list,
            show_std=show_std,
            config_name_list=config_name_list,
            num_hypothesis_list=num_hypothesis_list,
            metric_name_rmse=metric_name_rmse,
            metric_name_risk=metric_name_risk,
            metric_name_CRPS_sum=metric_name_CRPS_sum,
            metric_name_tv=metric_name_tv,
        )
        df = {
            metric_name_risk: df_risk,
            metric_name_rmse: df_rmse,
            metric_name_CRPS_sum: df_CRPS_sum,
            metric_name_tv: df_tv,
        }

        latex_tv_table = create_fixed_hypothesis_metric_table(
            metric_df_dict=df[fixed_metric],
            metric_name=fixed_metric,
            fixed_num_hypothesis=fixed_num_hypothesis,
            datasets=datasets_list,
            methods=config_name_list,
            insert_vertical_line=True,
            lower_is_better=lower_is_better_dict[fixed_metric],
            gray_first_two_columns=False,
        )
        f.write(latex_tv_table + "\n\n")

        # Table 4: Distortion against number of hypotheses (Solar)
        num_hypothesis_list = ["1", "2", "3", "4", "5", "8", "16"]
        datasets_list = ["solar"]
        fixed_metric = metric_name_risk
        config_name_list = [
            "tactis2",
            "timeGrad",
            "tempflow",
            "MCLrelaxed0.1mean",
            "aMCL0.95mean",
        ]
        df_rmse, df_risk, df_CRPS_sum, df_tv, datasets = initialize_dataframes(datasets_list, config_name_list, num_hypothesis_list)
        df_rmse, df_risk, df_CRPS_sum, df_tv = extract_data(
            datasets=datasets,
            df_rmse=df_rmse,
            df_risk=df_risk,
            df_CRPS_sum=df_CRPS_sum,
            df_tv=df_tv,
            small_std_font=True,
            keep_single_seed=keep_single_seed,
            datasets_list=datasets_list,
            show_std=show_std,
            config_name_list=config_name_list,
            num_hypothesis_list=num_hypothesis_list,
            metric_name_rmse=metric_name_rmse,
            metric_name_risk=metric_name_risk,
            metric_name_CRPS_sum=metric_name_CRPS_sum,
            metric_name_tv=metric_name_tv,
        )
        df = {
            metric_name_risk: df_risk,
            metric_name_rmse: df_rmse,
            metric_name_CRPS_sum: df_CRPS_sum,
            metric_name_tv: df_tv,
        }
        num_hypothesis_list = ["1", "2", "3", "4", "5", "8", "16"]
        dataset_name = "solar"
        metric = metric_name_risk
        latex_table = create_single_dataset_table(
            metric_df_dict=df[metric],
            metric_name=metric,
            dataset_name=dataset_name,
            methods=config_name_list,
            num_hypotheses_list=num_hypothesis_list,
            insert_vertical_line=False,
            gray_first_two_columns=False,
        )
        f.write(latex_table + "\n\n")

        # Table 6: Distortion comparison (8 hyps)
        datasets_list = ["electricity", "exchange", "solar", "traffic", "taxi", "wiki"]
        num_hypothesis_list = ["8"]
        fixed_num_hypothesis = "8"
        fixed_metric = metric_name_risk
        config_name_list = [
            "ETS",
            "transformer_tempflow",
            "tactis2",
            "timeGrad",
            "deepAR",
            "tempflow",
            "MCLrelaxed0.1mean",
            "aMCL0.95mean",
        ]
        df_rmse, df_risk, df_CRPS_sum, df_tv, datasets = initialize_dataframes(datasets_list, config_name_list, num_hypothesis_list)
        df_rmse, df_risk, df_CRPS_sum, df_tv = extract_data(
            datasets=datasets,
            df_rmse=df_rmse,
            df_risk=df_risk,
            df_CRPS_sum=df_CRPS_sum,
            df_tv=df_tv,
            small_std_font=False,
            keep_single_seed=keep_single_seed,
            datasets_list=datasets_list,
            show_std=show_std,
            config_name_list=config_name_list,
            num_hypothesis_list=num_hypothesis_list,
            metric_name_rmse=metric_name_rmse,
            metric_name_risk=metric_name_risk,
            metric_name_CRPS_sum=metric_name_CRPS_sum,
            metric_name_tv=metric_name_tv,
        )
        df = {
            metric_name_risk: df_risk,
            metric_name_rmse: df_rmse,
            metric_name_CRPS_sum: df_CRPS_sum,
            metric_name_tv: df_tv,
        }
        latex_distortion_table = create_fixed_hypothesis_metric_table(
            metric_df_dict=df[fixed_metric],
            metric_name=fixed_metric,
            fixed_num_hypothesis=fixed_num_hypothesis,
            datasets=datasets_list,
            methods=config_name_list,
            insert_vertical_line=True,
            lower_is_better=lower_is_better_dict[fixed_metric],
            gray_first_two_columns=True,
        )
        f.write(latex_distortion_table + "\n\n")

        # Table 7: RMSE comparison (16 hyps)
        datasets_list = ["electricity", "exchange", "solar", "traffic", "taxi", "wiki"]
        num_hypothesis_list = ["16"]
        fixed_num_hypothesis = "16"
        fixed_metric = metric_name_rmse
        config_name_list = [
            "ETS",
            "transformer_tempflow",
            "tactis2",
            "timeGrad",
            "deepAR",
            "tempflow",
            "MCLrelaxed0.1mean",
            "aMCL0.95mean",
        ]
        df_rmse, df_risk, df_CRPS_sum, df_tv, datasets = initialize_dataframes(datasets_list, config_name_list, num_hypothesis_list)
        df_rmse, df_risk, df_CRPS_sum, df_tv = extract_data(
            datasets=datasets,
            df_rmse=df_rmse,
            df_risk=df_risk,
            df_CRPS_sum=df_CRPS_sum,
            df_tv=df_tv,
            small_std_font=False,
            keep_single_seed=keep_single_seed,
            datasets_list=datasets_list,
            show_std=show_std,
            config_name_list=config_name_list,
            num_hypothesis_list=num_hypothesis_list,
            metric_name_rmse=metric_name_rmse,
            metric_name_risk=metric_name_risk,
            metric_name_CRPS_sum=metric_name_CRPS_sum,
            metric_name_tv=metric_name_tv,
        )
        df = {
            metric_name_risk: df_risk,
            metric_name_rmse: df_rmse,
            metric_name_CRPS_sum: df_CRPS_sum,
            metric_name_tv: df_tv,
        }
        latex_distortion_table = create_fixed_hypothesis_metric_table(
            metric_df_dict=df[fixed_metric],
            metric_name=fixed_metric,
            fixed_num_hypothesis=fixed_num_hypothesis,
            datasets=datasets_list,
            methods=config_name_list,
            insert_vertical_line=True,
            lower_is_better=lower_is_better_dict[fixed_metric],
            gray_first_two_columns=True,
        )
        f.write(latex_distortion_table + "\n\n")

        # Table 8: CRPS comparison (16 hyps)
        datasets_list = ["electricity", "exchange", "solar", "traffic", "taxi", "wiki"]
        num_hypothesis_list = ["16"]
        fixed_num_hypothesis = "16"
        fixed_metric = metric_name_CRPS_sum
        config_name_list = [
            "ETS",
            "transformer_tempflow",
            "tactis2",
            "timeGrad",
            "deepAR",
            "tempflow",
            "MCLrelaxed0.1mean",
            "aMCL0.95mean",
        ]
        df_rmse, df_risk, df_CRPS_sum, df_tv, datasets = initialize_dataframes(datasets_list, config_name_list, num_hypothesis_list)
        df_rmse, df_risk, df_CRPS_sum, df_tv = extract_data(
            datasets=datasets,
            df_rmse=df_rmse,
            df_risk=df_risk,
            df_CRPS_sum=df_CRPS_sum,
            df_tv=df_tv,
            small_std_font=False,
            keep_single_seed=keep_single_seed,
            datasets_list=datasets_list,
            show_std=show_std,
            config_name_list=config_name_list,
            num_hypothesis_list=num_hypothesis_list,
            metric_name_rmse=metric_name_rmse,
            metric_name_risk=metric_name_risk,
            metric_name_CRPS_sum=metric_name_CRPS_sum,
            metric_name_tv=metric_name_tv,
        )
        df = {
            metric_name_risk: df_risk,
            metric_name_rmse: df_rmse,
            metric_name_CRPS_sum: df_CRPS_sum,
            metric_name_tv: df_tv,
        }
        latex_distortion_table = create_fixed_hypothesis_metric_table(
            metric_df_dict=df[fixed_metric],
            metric_name=fixed_metric,
            fixed_num_hypothesis=fixed_num_hypothesis,
            datasets=datasets_list,
            methods=config_name_list,
            insert_vertical_line=True,
            lower_is_better=lower_is_better_dict[fixed_metric],
            gray_first_two_columns=True,
        )

        f.write(latex_distortion_table + "\n\n")

        # Table 9: Distortion against number of hypotheses (Solar)
        datasets_list = ["electricity", "exchange", "solar", "traffic", "taxi", "wiki"]
        num_hypothesis_list = ["1", "2", "3", "4", "5", "8", "16"]
        fixed_metric = metric_name_risk
        config_name_list = [
            "ETS",
            "transformer_tempflow",
            "tactis2",
            "timeGrad",
            "deepAR",
            "tempflow",
            "MCLrelaxed0.1mean",
            "aMCL0.95mean",
        ]
        df_rmse, df_risk, df_CRPS_sum, df_tv, datasets = initialize_dataframes(datasets_list, config_name_list, num_hypothesis_list)
        df_rmse, df_risk, df_CRPS_sum, df_tv = extract_data(
            datasets=datasets,
            df_rmse=df_rmse,
            df_risk=df_risk,
            df_CRPS_sum=df_CRPS_sum,
            df_tv=df_tv,
            small_std_font=False,
            keep_single_seed=keep_single_seed,
            datasets_list=datasets_list,
            show_std=show_std,
            config_name_list=config_name_list,
            num_hypothesis_list=num_hypothesis_list,
            metric_name_rmse=metric_name_rmse,
            metric_name_risk=metric_name_risk,
            metric_name_CRPS_sum=metric_name_CRPS_sum,
            metric_name_tv=metric_name_tv,
        )
        df = {
            metric_name_risk: df_risk,
            metric_name_rmse: df_rmse,
            metric_name_CRPS_sum: df_CRPS_sum,
            metric_name_tv: df_tv,
        }
        num_hypothesis_list = ["1", "2", "3", "4", "5", "8", "16"]
        dataset_name = "solar"
        metric = metric_name_risk

        latex_table = create_single_dataset_table(
            metric_df_dict=df[metric],
            metric_name=metric,
            dataset_name=dataset_name,
            methods=config_name_list,
            num_hypotheses_list=num_hypothesis_list,
            insert_vertical_line=False,
            gray_first_two_columns=True,
        )
        f.write(latex_table + "\n\n")

        print("Tables generated successfully in latex_tables_output.txt!")

        # Table 11 Total Variation full scores
        config_name_list = [
            "ETS",
            "transformer_tempflow",
            "tactis2",
            "timeGrad",
            "deepAR",
            "tempflow",
            "MCLrelaxed0.1mean",
            "aMCL0.95mean",
        ]
        fixed_metric = metric_name_tv
        df_rmse, df_risk, df_CRPS_sum, df_tv, datasets = initialize_dataframes(datasets_list, config_name_list, num_hypothesis_list)
        df_rmse, df_risk, df_CRPS_sum, df_tv = extract_data(
            datasets=datasets,
            df_rmse=df_rmse,
            df_risk=df_risk,
            df_CRPS_sum=df_CRPS_sum,
            df_tv=df_tv,
            small_std_font=False,
            keep_single_seed=keep_single_seed,
            datasets_list=datasets_list,
            show_std=show_std,
            config_name_list=config_name_list,
            num_hypothesis_list=num_hypothesis_list,
            metric_name_rmse=metric_name_rmse,
            metric_name_risk=metric_name_risk,
            metric_name_CRPS_sum=metric_name_CRPS_sum,
            metric_name_tv=metric_name_tv,
        )
        df = {
            metric_name_risk: df_risk,
            metric_name_rmse: df_rmse,
            metric_name_CRPS_sum: df_CRPS_sum,
            metric_name_tv: df_tv,
        }

        latex_tv_table = create_fixed_hypothesis_metric_table(
            metric_df_dict=df[fixed_metric],
            metric_name=fixed_metric,
            fixed_num_hypothesis=fixed_num_hypothesis,
            datasets=datasets_list,
            methods=config_name_list,
            insert_vertical_line=True,
            lower_is_better=lower_is_better_dict[fixed_metric],
            gray_first_two_columns=True,
        )
        f.write(latex_tv_table + "\n\n")

    if plot_additional_tables:
        # Table 11: Total Variation (4 hyps)
        config_name_list = [
            "ETS",
            "transformer_tempflow",
            "tactis2",
            "timeGrad",
            "tempflow",
            "MCLrelaxed0.1mean",
            "aMCL0.95mean",
        ]
        fixed_metric = metric_name_tv
        num_hypothesis_list = ["4"]
        datasets_list = ["crypt"]
        df_rmse, df_risk, df_CRPS_sum, df_tv, datasets = initialize_dataframes(datasets_list, config_name_list, num_hypothesis_list)
        df_rmse, df_risk, df_CRPS_sum, df_tv = extract_data(
            datasets=datasets,
            df_rmse=df_rmse,
            df_risk=df_risk,
            df_CRPS_sum=df_CRPS_sum,
            df_tv=df_tv,
            small_std_font=False,
            keep_single_seed=keep_single_seed,
            datasets_list=datasets_list,
            show_std=show_std,
            config_name_list=config_name_list,
            num_hypothesis_list=num_hypothesis_list,
            metric_name_rmse=metric_name_rmse,
            metric_name_risk=metric_name_risk,
            metric_name_CRPS_sum=metric_name_CRPS_sum,
            metric_name_tv=metric_name_tv,
        )
        df = {
            metric_name_risk: df_risk,
            metric_name_rmse: df_rmse,
            metric_name_CRPS_sum: df_CRPS_sum,
            metric_name_tv: df_tv,
        }

        latex_tv_table = create_fixed_hypothesis_metric_table(
            metric_df_dict=df[fixed_metric],
            metric_name=fixed_metric,
            fixed_num_hypothesis=4,
            datasets=datasets_list,
            methods=config_name_list,
            insert_vertical_line=True,
            lower_is_better=lower_is_better_dict[fixed_metric],
            gray_first_two_columns=True,
        )

# %%

reproduce_tables()

# %% Process Data

small_std_font = False  # Use small font (or not) for the std
show_std = True  # Show the std in the table (or not)
keep_single_seed = (
    None  # Set the string of the seed to keep only one run per seed (or not)
)

df_rmse, df_risk, df_CRPS_sum, df_tv = extract_data(
    datasets=datasets,
    df_rmse=df_rmse,
    df_risk=df_risk,
    df_CRPS_sum=df_CRPS_sum,
    df_tv=df_tv,
    small_std_font=small_std_font,
    keep_single_seed=keep_single_seed,
    datasets_list=datasets_list,
    show_std=show_std,
    config_name_list=config_name_list,
    num_hypothesis_list=num_hypothesis_list,
    metric_name_rmse=metric_name_rmse,
    metric_name_risk=metric_name_risk,
    metric_name_CRPS_sum=metric_name_CRPS_sum,
    metric_name_tv=metric_name_tv,
)

# %%
generate_tables_fix_num_hypotheses = (
    False  # Generate tables for a fixed number of hypotheses
)
generate_tables_single_dataset = False  # Generate tables for a single dataset

# %% Generate Tables

if generate_tables_fix_num_hypotheses:

    fixed_num_hypothesis = "8"  # Change this to the number of hypotheses you want to generate the table for
    fixed_metric = (
        metric_name_risk  # Change this to the metric you want to generate the table for
    )
    # fixed_metric = metric_name_tv
    # fixed_metric = metric_name_rmse
    # fixed_metric = metric_name_CRPS_sum
    config_name_list = [
        "ETS",
        "transformer_tempflow",
        "tactis2",
        "timeGrad",
        "deepAR",
        "tempflow",
        "MCLrelaxed0.1mean",
        "aMCL0.95mean",
    ]  # Change this to the models you want to generate the table for

    df = {
        metric_name_risk: df_risk,
        metric_name_rmse: df_rmse,
        metric_name_CRPS_sum: df_CRPS_sum,
        metric_name_tv: df_tv,
    }

    lower_is_better_dict = {
        metric_name_risk: True,
        metric_name_rmse: True,
        metric_name_CRPS_sum: True,
        metric_name_tv: True,
    }

    for metric in [fixed_metric]:
        latex_distortion_table = create_fixed_hypothesis_metric_table(
            metric_df_dict=df[metric],
            metric_name=metric,
            fixed_num_hypothesis=fixed_num_hypothesis,
            datasets=datasets_list,
            methods=config_name_list,
            insert_vertical_line=True,
            lower_is_better=lower_is_better_dict[metric],
            gray_first_two_columns=True,
        )

        print(latex_distortion_table)

# # %%
if generate_tables_single_dataset:
    # Example usage for single dataset tables

    dataset_name = "solar"  # Change this to the dataset you want
    metric = metric_name_risk
    config_name_list = [
        "ETS",
        "transformer_tempflow",
        "tactis2",
        "timeGrad",
        "deepAR",
        "tempflow",
        "MCLrelaxed0.1mean",
        "aMCL0.95mean",
    ]

    df = {
        metric_name_risk: df_risk,
        metric_name_rmse: df_rmse,
        metric_name_CRPS_sum: df_CRPS_sum,
    }

    latex_table = create_single_dataset_table(
        metric_df_dict=df[metric],
        metric_name=metric,
        dataset_name=dataset_name,
        methods=config_name_list,
        num_hypotheses_list=num_hypothesis_list,
        insert_vertical_line=False,
        gray_first_two_columns=True,
    )
    print(latex_table)
    print("\n")

################## Extract Tables Crypt

#%%

csv_files = os.listdir(os.path.join(os.environ['PROJECT_ROOT'], 'tsExperiments', 'results', 'saved_csv'))
csv_path = [e for e in csv_files if 'crypt_101' in e]
assert len(csv_path) == 1, "There should be only one crypt_101 file"
csv_path = os.path.join(os.environ['PROJECT_ROOT'], 'tsExperiments', 'results', 'saved_csv', csv_path[0])

#%%

df = pd.read_csv(csv_path)

### For the row, keep only those where model/params/scaler_type is mean_std
df = df[~((df['model/name'].str.contains('timeGrad')) & (df['model/params/scaler_type'] != 'mean_std'))]

# Same for tempflow
df = df[~((df['model/name'].str.contains('tempflow')) & (df['model/params/scaler_type'] != 'mean_std'))]

# Opposite for deepAR
df = df[~((df['model/name'].str.contains('deepAR')) & (df['model/params/scaler_type'] == 'mean_std'))]

# Opposite for transformer_tempflow
df = df[~((df['model/name'].str.contains('transformer_tempflow')) & (df['model/params/scaler_type'] != 'mean_std'))]

#%%

df = df.sort_values(
                    by="_start_time", ascending=False
                )

            
df = df.drop_duplicates(subset=["Name"])

#%%

def set_lowest_and_second_lowest_in_bold_and_underline(df, exclude_methods=[]):
    """Format lowest and second lowest values in DataFrame"""
    for index, row in df.iterrows():
        values = []
        method_value_map = {}
        # Process non-excluded methods for min/second min
        for col in [c for c in df.columns if c not in exclude_methods]:
            cell = row[col]
            if isinstance(cell, str):
                try:
                    value = float(cell.split(' ')[0])
                    values.append(value)
                    method_value_map[col] = value
                except ValueError:
                    continue

        if not values:
            continue

        sorted_values = sorted(set(values))
        min_val = sorted_values[0]
        second_min_val = sorted_values[1] if len(sorted_values) > 1 else None

        # Process non-excluded methods for bold/underline
        for col in [c for c in df.columns if c not in exclude_methods]:
            cell = row[col]
            if isinstance(cell, str):
                try:
                    value = float(cell.split(' ')[0])
                    if value == min_val:
                        df.at[index, col] = f"\\textbf{{{cell}}}"
                    elif second_min_val is not None and value == second_min_val:
                        df.at[index, col] = f"\\underline{{{cell}}}"
                except ValueError:
                    continue

        # Process excluded methods to be grey
        for col in exclude_methods:
            cell = row[col]
            if isinstance(cell, str):
                df.at[index, col] = f"\\textcolor{{gray}}{{{cell}}}"

    return df

result = df.groupby(['model/name']).agg(
    distortion_mean=('Distorsion', 'mean'),
    distortion_std=('Distorsion', 'std'),
    total_variation_mean=('total_variation', 'mean'),
    total_variation_std=('total_variation', 'std'),
    crps_mean=('m_sum_mean_wQuantileLoss', 'mean'),
    crps_std=('m_sum_mean_wQuantileLoss', 'std'),
    RMSE_mean=('m_sum_RMSE', 'mean'),
    RMSE_std=('m_sum_RMSE', 'std'),
).reset_index()

#%%

# exclude wta_mode
result = result[~(result['model/name'].str.contains('timeMCL'))]

result_awta = df.groupby(['model/name', 'model/params/wta_mode',"model/params/scaler_type"]).agg(
    distortion_mean=('Distorsion', 'mean'),
    distortion_std=('Distorsion', 'std'),
    total_variation_mean=('total_variation', 'mean'),
    total_variation_std=('total_variation', 'std'),
    crps_mean=('m_sum_mean_wQuantileLoss', 'mean'),
    crps_std=('m_sum_mean_wQuantileLoss', 'std'),
    RMSE_mean=('m_sum_RMSE', 'mean'),
    RMSE_std=('m_sum_RMSE', 'std'),
).reset_index()
result_awta = result_awta[result_awta['model/params/scaler_type'] == 'mean_std']
result_awta['model/name'] = result_awta['model/name'] + result_awta['model/params/wta_mode']
# exclude relaxed mode
result_awta = result_awta[result_awta['model/params/wta_mode'] != 'relaxed-wta']
#%%
# Same for result_awta
result_awta = result_awta.drop(columns=['model/params/wta_mode', 'model/params/scaler_type'])

# Concatenate the two DataFrames, and reindex
result = pd.concat([result, result_awta])
result = result.reset_index(drop=True)

#%%

# Create result columns with mean ± std format
result['distortion'] = result['distortion_mean'].round(3).astype(str) + ' $\\pm$ ' + result['distortion_std'].round(3).astype(str)
result['total_variation'] = result['total_variation_mean'].round(3).astype(str) + ' $\\pm$ ' + result['total_variation_std'].round(3).astype(str)
result['crps'] = result['crps_mean'].round(3).astype(str) + ' $\\pm$ ' + result['crps_std'].round(3).astype(str)
result['RMSE'] = result['RMSE_mean'].round(3).astype(str) + ' $\\pm$ ' + result['RMSE_std'].round(3).astype(str)

# Define metrics and their display names
metrics = {
    'distortion': 'Distortion',
    'total_variation': 'Total Variation',
    'crps': 'CRPS',
    'RMSE': 'RMSE'
}

# Create a single table with metrics as rows and methods as columns
table_data = pd.DataFrame(index=metrics.values(), columns=result['model/name'].unique())

# Fill the table with values
for model in table_data.columns:
    mask = (result['model/name'] == model)
    if mask.any():
        for metric_key, metric_name in metrics.items():
            value = result.loc[mask, metric_key].values[0]
            table_data.at[metric_name, model] = value

# Format the table with bold and underline for lowest and second lowest values
table_data = set_lowest_and_second_lowest_in_bold_and_underline(
    table_data,
)

# Change the method names using methods_to_render_dict

methods_to_render_dict = {
    'timeGrad': '\\textbf{\\texttt{TimeGrad}}',
    'tempflow': '\\textbf{\\texttt{TempFlow}}',
    'transformer_tempflow': '\\textbf{\\texttt{Trf.TempFlow}}',
    'tactis2': '\\textbf{\\texttt{Tactis2}}',
    'timeMCL': '\\textbf{\\texttt{Time MCL}}',
    'deepAR': '\\textbf{\\texttt{DeepAR}}',
    'ETS': '\\textbf{\\texttt{ETS}}',
    "timeMCLrelaxed-wta_0.1": "\\textbf{\\texttt{Time MCL (Rel.)}}",
    "timeMCLawta": "\\textbf{\\texttt{Time MCL (Ann.)}}"}

# Before renaming, print the current column names to debug
print("Current column names:", table_data.columns.tolist())

desired_order = [
    'transformer_tempflow',
    'tactis2',
    'timeGrad',
    'deepAR',
    'tempflow',
    'timeMCLawta',
]

# Filter out any columns that don't exist in the data
desired_order = [col for col in desired_order if col in table_data.columns]

# Reorder the columns
table_data = table_data[desired_order]

# Rename the columns using the mapping
table_data.columns = table_data.columns.map(methods_to_render_dict)

# Print the new column names to verify
print("New column names:", table_data.columns.tolist())

#%%

# Convert to LaTeX
str_table = table_data.to_latex(
    escape=False,
    column_format='l|' + 'c' * (len(table_data.columns)-1) + '|c',  # left align metric names, center values
    index=True,
    index_names=False
)

# Create the complete LaTeX table with caption and label
caption = "Results for cryptocurrency dataset"
label = "results_crypto"

latex_table = f"""\\begin{{table}}
\\begin{{center}}
\\caption{{{caption}}}
\\label{{tab:{label}}}
\\resizebox{{\\columnwidth}}{{!}}{{
{str_table}
}}
\\end{{center}}
\\end{{table}}"""

print("\nResults Table:")
print(latex_table)
print("\n" + "="*80 + "\n")

#%%

gray_first_two_columns = True

# Add gray coloring to first two columns if requested
if gray_first_two_columns:
    # Split the table into lines
    lines = latex_table.split('\n')
    for i, line in enumerate(lines):
        if '&' in line:  # Only process data rows
            parts = line.split('&')
            # Wrap first two columns in \textcolor{gray}{}
            parts[1] = f"\\textcolor{{gray}}{{{parts[1].strip()}}}"
            parts[2] = f"\\textcolor{{gray}}{{{parts[2].strip()}}}"
            lines[i] = ' & '.join(parts)
    latex_table = '\n'.join(lines)

num_hypotheses_list = result['nb_hyp'].unique().astype(int)

# Define the order you want
methods_order = [
    'timeGrad',
    'deepAR',
    'tempflow',
    'transformer_tempflow',
    'tactis2',
    'timeMCL',
]

methods_to_render_dict = {
    'timeGrad': '\\textbf{\\texttt{TimeGrad}}',
    'tempflow': '\\textbf{\\texttt{TempFlow}}',
    'transformer_tempflow': '\\textbf{\\texttt{Transf. TempFlow}}',
    'tactis2': '\\textbf{\\texttt{Tactis2}}',
    'timeMCL': '\\textbf{\\texttt{Time MCL}}',
    'deepAR': '\\textbf{\\texttt{DeepAR}}',
    'ETS': '\\textbf{\\texttt{ETS}}',
}

methods = [m for m in methods_order if m in result['model'].unique()]
# methods = result['model'].unique()
methods_to_render = [methods_to_render_dict[method] for method in methods]

#%%

# Create a dictionary to store the formatted time values for each dataset
metric_df_dict = {}

for dataset_name in result['dataset_name'].unique():
    dataset_results = result[result['dataset_name'] == dataset_name]
    
    # Create a DataFrame to store formatted time values
    metric_df = pd.DataFrame(index=num_hypotheses_list, columns=methods)
    
    for num_hyp in num_hypotheses_list:
        for method in methods:
            mask = (dataset_results['nb_hyp'] == num_hyp) & (dataset_results['model'] == method)
            if mask.any():
                mean = dataset_results.loc[mask, 'time_mean'].values[0]
                std = dataset_results.loc[mask, 'time_std'].values[0]
                # Format with mean ± std, rounded to 2 decimal places
                metric_df.at[num_hyp, method] = f"{mean:.2f} $\\pm$ {std:.2f}"
            else:
                metric_df.at[num_hyp, method] = "N/A"
    
    metric_df_dict[dataset_name] = metric_df

# Generate tables for each dataset
for dataset_name in metric_df_dict:
    # Create the table data
    table_data = pd.DataFrame(index=num_hypotheses_list, columns=methods_to_render)
    
    for num_hyp in num_hypotheses_list:
        for method in methods:
            try:
                value = metric_df_dict[dataset_name].loc[num_hyp, method]
                if type(value) == str and 'nan' in value.split(' $\\pm$')[0]:
                    table_data.at[num_hyp, methods_to_render_dict[method]] = 'N/A'
                else:
                    table_data.at[num_hyp, methods_to_render_dict[method]] = value
            except KeyError:
                table_data.at[num_hyp, methods_to_render_dict[method]] = 'N/A'

    # Format the table with bold and underline for lowest and second lowest values
    table_data = set_lowest_and_second_lowest_in_bold_and_underline(table_data, exclude_methods=['\\textbf{\\texttt{ETS}}'])
    
    # Convert to LaTeX
    str_table = table_data.to_latex(escape=False,column_format='c' * len(table_data.columns))

    # Add vertical bar before the last column
    # First, find where the column specification starts and ends
    first_brace = str_table.find('{')
    col_start = str_table.find('{', first_brace + 1) + 1  # Find second occurrence
    col_end = str_table.find('}', col_start)
    col_spec = str_table[col_start:col_end]
    
    # Count the number of columns (number of 'l' characters)
    num_cols = len(col_spec)
    
    # Insert the vertical bar before the last column
    new_col_spec = col_spec[:-1] + '||' + col_spec[-1]
    
    # Replace the original column specification
    str_table = str_table.replace(f'{{{col_spec}}}', f'{{{new_col_spec}}}')

    # Create the complete LaTeX table with caption and label
    caption = f"Computation time (in seconds) for {dataset_name} dataset"
    label = f"computation_time_{dataset_name.lower()}"
    
    latex_table = f"""\\begin{{table}}
    \\begin{{center}}
    \\caption{{{caption}}}
    \\label{{tab:{label}}}
    \\resizebox{{\\columnwidth}}{{!}}{{
    {str_table}
    }}
    \\end{{center}}
    \\end{{table}}"""

    # print(latex_table)

    gray_first_two_columns = True

    # Add gray coloring to first two columns if requested
    if gray_first_two_columns:
        # Split the table into lines
        lines = latex_table.split('\n')
        for i, line in enumerate(lines):
            if '&' in line:  # Only process data rows
                parts = line.split('&')
                # Wrap first two columns in \textcolor{gray}{}
                parts[1] = f"\\textcolor{{gray}}{{{parts[1].strip()}}}"
                parts[2] = f"\\textcolor{{gray}}{{{parts[2].strip()}}}"
                parts[3] = f"\\textcolor{{gray}}{{{parts[3].strip()}}}"
                lines[i] = ' & '.join(parts)
        latex_table = '\n'.join(lines)
# %%

import pandas as pd

csv_files = os.listdir(os.path.join(os.environ['PROJECT_ROOT'], 'tsExperiments', 'results', 'saved_csv'))
csv_path = [e for e in csv_files if 'crypt_101' in e]
csv_file = os.path.join(os.environ['PROJECT_ROOT'], 'tsExperiments', 'results', 'saved_csv', csv_path[0])

df_flops = pd.read_csv(csv_file)

df_flops = df_flops[['model/name', 'prediction_flops']]

df_flops = df_flops.T

df_flops.columns = df_flops.iloc[0]
# Remove the first row
df_flops = df_flops.iloc[1:]

desired_order = [
    'ETS',
    'transformer_tempflow',
    'tactis2',
    'timeGrad',
    'deepAR',
    'tempflow',
    'timeMCL',
]

# Filter out any columns that don't exist in the data
desired_order = [col for col in desired_order if col in df_flops.columns]

# Reorder the columns
df_flops = df_flops[desired_order]

# Rename the columns using the mapping
df_flops.columns = df_flops.columns.map(methods_to_render_dict)

# Write the flops in scientific notation \times 10^
df_flops = df_flops.applymap(lambda x: f"{x:.2e}")

# Transform the e+0x to \times 10^x
df_flops = df_flops.applymap(lambda x: x.replace('e+0', ' \\times 10^{'))
df_flops = df_flops.applymap(lambda x: x + '}')

a = df_flops.to_latex(escape=False)
# %%
