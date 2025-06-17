# %%

import pandas as pd
import os
import sys
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
sys.path.append(os.path.dirname(os.environ["PROJECT_ROOT"]))

df = pd.read_csv(
    os.path.join(
        os.environ["PROJECT_ROOT"],
        "tsExperiments",
        "computation_time",
        "results",
        "time_of_runs_exchange_rate_nips_nbsimu_16.csv",
    )
)
df = df[df["simu_number"].astype(int) > 0]

# %%


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
                    value = float(cell.split(" ")[0])
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
                    value = float(cell.split(" ")[0])
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


result = (
    df.groupby(["dataset_name", "model", "nb_hyp"])
    .agg(time_mean=("time", "mean"), time_std=("time", "std"))
    .reset_index()
)


# %%

num_hypotheses_list = result["nb_hyp"].unique().astype(int)

# Define the order you want
methods_order = [
    "ETS",
    "transformer_tempflow",
    "tactis2",
    "timeGrad",
    "DeepVar",
    "deepVAR3",
    "tempflow",
    "timeMCL",
]

methods_to_render_dict = {
    "ETS": "\\textbf{\\texttt{ETS}}",
    "timeGrad": "\\textbf{\\texttt{TimeGrad}}",
    "DeepVar": "\\textbf{\\texttt{DeepAR}}",
    "tempflow": "\\textbf{\\texttt{TempFlow}}",
    "transformer_tempflow": "\\textbf{\\texttt{Transf. TempFlow}}",
    "tactis2": "\\textbf{\\texttt{Tactis2}}",
    "timeMCL": "\\textbf{\\texttt{Time MCL}}",
    "deepVAR3": "\\textbf{\\texttt{DeepAR}}",
}

methods = [m for m in methods_order if m in result["model"].unique()]
# methods = result['model'].unique()
methods_to_render = [methods_to_render_dict[method] for method in methods]


# %%

import pandas as pd

# Create a dictionary to store the formatted time values for each dataset
metric_df_dict = {}

for dataset_name in result["dataset_name"].unique():
    dataset_results = result[result["dataset_name"] == dataset_name]

    # Create a DataFrame to store formatted time values
    metric_df = pd.DataFrame(index=num_hypotheses_list, columns=methods)

    for num_hyp in num_hypotheses_list:
        for method in methods:
            mask = (dataset_results["nb_hyp"] == num_hyp) & (
                dataset_results["model"] == method
            )
            if mask.any():
                mean = dataset_results.loc[mask, "time_mean"].values[0]
                std = dataset_results.loc[mask, "time_std"].values[0]
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
                if type(value) == str and "nan" in value.split(" $\\pm$")[0]:
                    table_data.at[num_hyp, methods_to_render_dict[method]] = "N/A"
                else:
                    table_data.at[num_hyp, methods_to_render_dict[method]] = value
            except KeyError:
                table_data.at[num_hyp, methods_to_render_dict[method]] = "N/A"

    # Format the table with bold and underline for lowest and second lowest values
    table_data = set_lowest_and_second_lowest_in_bold_and_underline(
        table_data, exclude_methods=["\\textbf{\\texttt{ETS}}"]
    )
    # table_data = set_lowest_and_second_lowest_in_bold_and_underline(table_data)

    # Convert to LaTeX
    str_table = table_data.to_latex(
        escape=False, column_format="c" * len(table_data.columns)
    )

    # Add vertical bar before the last column
    # First, find where the column specification starts and ends
    first_brace = str_table.find("{")
    col_start = str_table.find("{", first_brace + 1) + 1  # Find second occurrence
    col_end = str_table.find("}", col_start)
    col_spec = str_table[col_start:col_end]

    # Count the number of columns (number of 'l' characters)
    num_cols = len(col_spec)

    # Insert the vertical bar before the last column
    new_col_spec = col_spec[:-1] + "||" + col_spec[-1]

    # Replace the original column specification
    str_table = str_table.replace(f"{{{col_spec}}}", f"{{{new_col_spec}}}")

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

    print(latex_table)
# %%
