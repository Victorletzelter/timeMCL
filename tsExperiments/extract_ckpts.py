"""
Checkpoint Extraction Script

This script extracts and organizes model checkpoints from a log directory. It processes folders
with names in the format: 'YYYY-MM-DD_HH-MM-SS_dataset_model_num_hypotheses'
and creates a structured JSON file organizing checkpoints by dataset, model, and number of hypotheses.

The script keeps only the most recent run for each unique combination of
(dataset_name, model_name, num_hypotheses).

Example folder structure:
logs/
    2025-01-15_13-21-31_electricity_timeMCL_4/
        checkpoints/
            epoch=10.ckpt
    2025-01-15_14-45-22_electricity_timeMCL_4/
        checkpoints/
            epoch=15.ckpt

Example output JSON structure:
{
    "electricity": {
        "timeMCL": {
            "4": ["/path/to/checkpoint1.ckpt", "/path/to/checkpoint2.ckpt"]
        }
    }
}
"""

import sys, os, rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
sys.path.append(os.path.dirname(os.environ["PROJECT_ROOT"]))

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse


def parse_folder_name(
    folder_name: str,
) -> Optional[Tuple[str, str, str, str, Optional[str]]]:
    """
    Parse a folder name to extract its components.

    Args:
        folder_name: String in format:
            'YYYY-MM-DD_HH-MM-SS_dataset_model_num_hypotheses' or
            'YYYY-MM-DD_HH-MM-SS_seed_${seed}_dataset_model_num_hypotheses_modelspecificities_fromckpt'

    Returns:
        Tuple containing (datetime_str, dataset_name, model_name, num_hypotheses, model_specificities)
        where model_specificities is None if not present
        Returns None if the folder name doesn't match expected format
    """
    try:
        parts = folder_name.split("_")
        if len(parts) < 5:  # Minimum required parts
            return None

        # Extract date and time parts
        date_str = parts[0]
        time_str = parts[1]
        # Combine date and time
        datetime_str = f"{date_str} {time_str.replace('-', ':')}"

        # Handle both formats
        seed = parts[3]
        dataset_name = parts[4]
        if ("transformer" in parts and "tempflow" in parts) or (
            "transformertempflow" in parts
        ):
            model_name = "transformer_tempflow"
            num_hypotheses = parts[7]
        else:
            model_name = parts[5]
            num_hypotheses = parts[6]
        # Combine remaining parts except 'fromckpt' for model specificities
        if model_name == "timeMCL":
            remaining_parts = parts[7:]
        else:
            remaining_parts = None
        model_specificities = (
            "_".join(remaining_parts) if remaining_parts is not None else None
        )

        return (
            datetime_str,
            seed,
            dataset_name,
            model_name,
            num_hypotheses,
            model_specificities,
        )
    except (ValueError, IndexError):
        return None


def get_checkpoint_paths(folder_path: Path, model_name: str) -> List[str]:
    """
    Recursively find all checkpoint files in a folder.

    Args:
        folder_path: Path to the folder to search in

    Returns:
        List of absolute paths to checkpoint files

    Note:
        - Excludes checkpoints with 'last' in their name
        - Only includes files ending with '.ckpt'
        - Returns absolute paths to checkpoints
    """
    if model_name == "tactis2":
        checkpoint_paths = {}
        # Special handling for tactis2 with two checkpoint directories
        for phase in ["1", "2"]:
            ckpt_dir = folder_path / f"checkpoints_{phase}"
            if ckpt_dir.is_dir():
                for root, _, files in os.walk(ckpt_dir):
                    for file in files:
                        if file.endswith(".ckpt") and "last" not in file:
                            new_filename = file.replace("=", "_")
                            old_path = Path(root) / file
                            new_path = Path(root) / new_filename
                            # Rename the file
                            old_path.rename(new_path)
                            checkpoint_paths[phase] = str(new_path.absolute())
        return checkpoint_paths
    else:
        checkpoint_paths = []
        # Original behavior for other models
        for root, _, files in os.walk(folder_path):
            for file in files:
                if "crypt" in folder_path.name:
                    ### For the crypto currency dataset, we use the last checkpoint, because the validation loss was found to be too noisy and 
                    ### we did not observe overfitting (validation loss was not increasing).
                    ### This could be solved in the future by increasing the validation set size.
                    if file.endswith(".ckpt") and "last" in file:
                        checkpoint_path = Path(root) / file
                        checkpoint_paths.append(str(checkpoint_path.absolute()))
                else:
                    if file.endswith(".ckpt") and "last" not in file:
                        checkpoint_path = Path(root) / file
                        checkpoint_paths.append(str(checkpoint_path.absolute()))
        return checkpoint_paths

def build_checkpoint_structure(
    base_log_dir: Path, start_date: datetime, datasets: List[str]
) -> Dict:
    """
    Build a dictionary structure of checkpoints from specified dataset directories.

    Args:
        base_log_dir: Base path containing dataset-specific log directories
        start_date: Datetime object representing the earliest date to consider
        datasets: List of dataset names to process

    Returns:
        Dictionary with structure:
        {
            "dataset_model_num_hypotheses[_modelspecificities]": [list of checkpoint paths]
        }
    """
    # First, collect all folders with their dates
    model_folders = {}

    # Iterate through specified datasets
    for dataset in datasets:
        dataset_log_dir = base_log_dir / dataset
        if not dataset_log_dir.is_dir():
            print(
                f"Warning: Directory for dataset '{dataset}' not found at {dataset_log_dir}"
            )
            continue

        if "runs" in os.listdir(dataset_log_dir):
            dataset_log_dir = dataset_log_dir / "runs"
        else:
            dataset_log_dir = dataset_log_dir

        for folder in os.listdir(dataset_log_dir):
            folder_path = dataset_log_dir / folder
            if not folder_path.is_dir():
                continue

            parsed = parse_folder_name(folder)
            if parsed is None:
                continue

            (
                datetime_str,
                seed,
                dataset_name,
                model_name,
                num_hypotheses,
                model_specificities,
            ) = parsed

            try:
                folder_datetime = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
                if folder_datetime < start_date:
                    continue
            except ValueError:
                continue

            # Create combined key with optional specificities
            key = f"seed_{seed}_{dataset_name}_{model_name}_{num_hypotheses}"
            if model_specificities:
                key = f"{key}_{model_specificities}"

            if key not in model_folders:
                model_folders[key] = []
            model_folders[key].append((folder_datetime, folder_path, model_name))

    # Build structure keeping only most recent folders
    structure = {}
    for key, folders in model_folders.items():
        # Sort folders by date (most recent first)
        model_name = folders[0][-1]
        folders.sort(key=lambda x: x[0], reverse=True)

        # Take only the most recent folder
        checkpoint_paths = None
        most_recent_folder = folders[0][1]

        checkpoint_paths = get_checkpoint_paths(most_recent_folder, model_name)

        # Skip if no checkpoints found in any folder
        if not checkpoint_paths:
            continue

        if type(checkpoint_paths) == dict:
            for phase in checkpoint_paths:
                structure[f"{key}_phase_{phase}"] = checkpoint_paths[phase]
        elif type(checkpoint_paths) == list:
            structure[key] = checkpoint_paths[0]

    return structure


def main():
    """
    Main function to run the checkpoint extraction script.

    Command line arguments:
        --log_dir: Base path containing dataset-specific log directories
                  (default: './tsExperiments/logs/')
        --datasets: Comma-separated list of dataset names (default: 'electricity,exchange,solar')
        --start_date: Start date in YYYYMMDD format (default: '20250115')
        --output_file: Output JSON file path (default: 'ckpts.json')

    Example usage:
        python extract_ckpts.py --log_dir ./logs --datasets electricity,exchange,solar --start_date 20240101 --output_file checkpoints.json
    """
    parser = argparse.ArgumentParser(
        description="Extract checkpoints from log directory"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=f'{os.environ["PROJECT_ROOT"]}/tsExperiments/logs',
        help="Base path containing dataset-specific log directories",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="electricity_200,exchange_200,solar_200,taxi_200,traffic_200,wiki_200,crypt_101",
        help="Comma-separated list of dataset names",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default="20250115",
        help="Start date in YYYYMMDD format",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=f'{os.environ["PROJECT_ROOT"]}/tsExperiments/ckpts.json',
        help="Output JSON file path",
    )

    args = parser.parse_args()

    # Convert start date string to datetime
    start_date = datetime.strptime(args.start_date, "%Y%m%d")

    # Parse datasets list
    datasets = [d.strip() for d in args.datasets.split(",")]

    # Build checkpoint structure
    log_dir = Path(args.log_dir)
    structure = build_checkpoint_structure(log_dir, start_date, datasets)

    # Write to JSON file
    with open(args.output_file, "w") as f:
        json.dump(structure, f, indent=4)


if __name__ == "__main__":
    main()
