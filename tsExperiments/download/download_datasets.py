from gluonts.dataset.repository import get_dataset
import gdown
import os
import sys
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
sys.path.append(os.path.dirname(os.environ["PROJECT_ROOT"]))

# Gluonts datasets

for dataset in ['electricity_nips', 'exchange_rate_nips', 'solar_nips', 'taxi_30min', 'traffic_nips', 'wiki-rolling_nips']:
  _ = get_dataset(dataset, regenerate=True)

# Crypto data

url= "https://drive.google.com/file/d/1GUsRSb1P4GMPJ1F3LxUfXhpVdDvShKCH/view?usp=sharing"

# Convert to downloadable format
file_id = url.split("/d/")[1].split("/")[0]
download_url = f"https://drive.google.com/uc?id={file_id}"

dataset_path = os.path.join(os.environ["PROJECT_ROOT"], "tsExperiments", "data", "crypto_data.csv")

if not os.path.exists(dataset_path):
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

# Download
gdown.download(download_url, output=dataset_path, quiet=False)