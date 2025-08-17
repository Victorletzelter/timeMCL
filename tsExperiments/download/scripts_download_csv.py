import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import time
import os
import argparse
import numpy as np
from datetime import datetime
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

MY_HOME = os.getenv("PROJECT_ROOT")

# Parse arguments from the command line
parser = argparse.ArgumentParser(
    description="Extract experiment runs into a DataFrame."
)
parser.add_argument(
    "--experiment_name",
    type=str,
    required=True,
    help="Name of the experiment to extract.",
)
parser.add_argument(
    "--exp_id", type=str, default=None, help="Name of the experiment to extract."
)
parser.add_argument(
    "--save_to_csv",
    type=bool,
    default=True,
    help="Save the extracted DataFrame to a CSV file.",
)
parser.add_argument(
    "--override_saved",
    type=bool,
    default=True,
    help="Override the saved CSV file if it exists.",
)
parser.add_argument(
    "--save_dir",
    type=str,
    default=f"{os.getenv('PROJECT_ROOT')}/results/saved_csv",
    help="Directory to save the CSV file.",
)
args = parser.parse_args()

# Correctly set the tracking URI
mlflow.set_tracking_uri(
    f"file://{os.getenv('PROJECT_ROOT')}/tsExperiments/logs/mlflow/mlruns"
)

# Initialize the MLflow client
client = MlflowClient()

# List experiments and select one to search for runs
experiments = client.search_experiments()
for exp in experiments:
    print(f"Experiment ID: {exp.experiment_id}, Name: {exp.name}")

mapping_id_name = {}

for exp in experiments:
    mapping_id_name[exp.name] = exp.experiment_id

mapping_name_id = {}

for exp in experiments:
    mapping_name_id[exp.experiment_id] = exp.name


def convert_to_datetime(date):
    # Convert the timestamp (assumed to be in milliseconds) to seconds
    return datetime.fromtimestamp(date / 1000)


def convert_list_dates(dates):
    # Convert only the non-None values in the list, keeping the same list length
    return [convert_to_datetime(date) if date is not None else None for date in dates]


def calculate_duration(row):
    start = row["start_times"]
    end = row["end_times"]

    # Convert start and end times to datetime, considering None values
    start_dt = convert_to_datetime(start)
    end_dt = convert_to_datetime(end)

    if start_dt is None or end_dt is None:
        return "Unknown"
    else:
        duration_minutes = (end_dt - start_dt).total_seconds() / 60
        return f"{duration_minutes:.1f}min"


def extract_dataframe(
    experiment_name,
    exp_id_provided=None,
    save_to_csv=True,
    override_saved=False,
    save_dir=None,
):
    if exp_id_provided is not None:
        assert (
            mapping_name_id[exp_id_provided] == experiment_name
        ), "The provided experiment id does not match the provided experiment name"
        exp_id = exp_id_provided
    else:
        exp_id = mapping_id_name[experiment_name]

    # We check if the experiment is already saved
    if override_saved is False:
        for file in os.listdir(f"{os.getenv('PROJECT_ROOT')}/results/saved_csv"):
            if file == "{}_id_{}.csv".format(experiment_name, exp_id):
                print("The experiment is already saved")
                return pd.read_csv(
                    f"{os.getenv('PROJECT_ROOT')}/results/saved_csv/{experiment_name}_id_{exp_id}.csv"
                )

    runs = client.search_runs(experiment_ids=[exp_id])

    # Initialize lists to hold metrics and parameters
    runs_data = []

    for run in runs:
        run_metrics = run.data.metrics
        run_params = run.data.params
        run_info = run.info
        run_info_dict = run_info.__dict__
        new_order = [
            "_start_time",
            "_end_time",
            "duration",
            "Name",
            "_run_id",
            "_experiment_id",
            "_status",
        ]
        run_info_dict["Name"] = run.info.run_name

        if (
            not (run_info_dict["_start_time"] is None)
            and np.isnan(run_info_dict["_start_time"]) == False
        ):
            run_info_dict["_start_time"] = convert_to_datetime(
                run_info_dict["_start_time"]
            )
        else:
            run_info_dict["_start_time"] = None

        if (
            not (run_info_dict["_end_time"] is None)
            and np.isnan(run_info_dict["_end_time"]) == False
        ):
            run_info_dict["_end_time"] = convert_to_datetime(run_info_dict["_end_time"])
        else:
            run_info_dict["_end_time"] = None

        if not (run_info_dict["_start_time"] is None) and not (
            run_info_dict["_end_time"] is None
        ):
            run_info_dict["duration"] = (
                run_info_dict["_end_time"] - run_info_dict["_start_time"]
            )
            # Calculate the duration in minutes
            duration_minutes = run_info_dict["duration"].total_seconds() / 60
            # Format the duration as a string with 1 decimal place (e.g., '58.5min')
            duration_str = f"{duration_minutes:.1f}min"
            # Assigning the formatted string back to your dictionary
            run_info_dict["duration"] = duration_str
        else:
            run_info_dict["duration"] = None

        run_info_dict = {key: run_info_dict[key] for key in new_order}
        # Combine metrics and parameters into a single dictionary
        run_data = {**run_info_dict, **run_metrics, **run_params}
        # set the name
        runs_data.append(run_data)

    if save_to_csv is True:
        # Convert the combined data into a DataFrame
        df = pd.DataFrame(runs_data)
        df.to_csv(os.path.join(save_dir, f"{experiment_name}_id_{exp_id}.csv"))

    # Convert the combined data into a DataFrame
    return pd.DataFrame(runs_data)


if __name__ == "__main__":
    df_runs = extract_dataframe(
        experiment_name=args.experiment_name,
        exp_id_provided=args.exp_id,
        save_to_csv=args.save_to_csv,
        override_saved=args.override_saved,
        save_dir=args.save_dir,
    )
