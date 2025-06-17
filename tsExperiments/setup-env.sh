#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "=== Python Virtual Environment Configuration ==="

# Check if Python3 is installed
if ! command -v python3 &> /dev/null
then
    echo "Python3 is not installed. Installing..."
    sudo apt update
    sudo apt install -y python3 python3-venv python3-pip
else
    echo "Python3 is already installed."
fi

# Set the base directory for virtual environments
BASE_VENV_DIR="$HOME/.venvs"

# Create the base directory if it doesn't exist
if [ ! -d "$BASE_VENV_DIR" ]; then
    echo "Creating base virtual environments directory at '$BASE_VENV_DIR'..."
    mkdir -p "$BASE_VENV_DIR"
    echo "Base virtual environments directory created."
else
    echo "Base virtual environments directory '$BASE_VENV_DIR' already exists."
fi

# Set the name and path of the virtual environment
ENV_NAME="research_time_series"
ENV_DIR="$BASE_VENV_DIR/$ENV_NAME"

# Create the virtual environment if it does not exist
if [ ! -d "$ENV_DIR" ]; then
    echo "Creating virtual environment '$ENV_NAME' at '$ENV_DIR'..."
    python3 -m venv "$ENV_DIR"
    echo "Virtual environment '$ENV_NAME' created at '$ENV_DIR'."
else
    echo "Virtual environment '$ENV_NAME' already exists at '$ENV_DIR'."
fi

# Activate the virtual environment
echo "Activating the virtual environment..."
source "$ENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies if requirements.txt exists in the project directory
PROJECT_DIR="$(pwd)"
REQUIREMENTS_FILE="$PROJECT_DIR/requirements.txt"

if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r "$REQUIREMENTS_FILE"
else
    echo "No requirements.txt file found in '$PROJECT_DIR'. You can manually install packages with 'pip install <package>'."
fi

# Deactivate the virtual environment after installation
deactivate

echo "=== Configuration Complete ==="
echo "To activate the virtual environment, use: source $ENV_DIR/bin/activate"
