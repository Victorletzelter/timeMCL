import gdown
import os
import sys
import rootutils
import zipfile

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
sys.path.append(os.path.dirname(os.environ["PROJECT_ROOT"]))

url= "https://drive.google.com/file/d/1WUEbXuto-rgUU1BDoA_rrwV5YANK1cBE/view?usp=sharing"

# Convert to downloadable format
file_id = url.split("/d/")[1].split("/")[0]
download_url = f"https://drive.google.com/uc?id={file_id}"

dataset_path = os.path.join(os.environ["PROJECT_ROOT"], "tsExperiments", "logs_timeMCL")

### Create the directory if it doesn't exist

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path, exist_ok=True)

# Download
gdown.download(download_url, output=dataset_path, quiet=False)

# Extract the path of the downloaded file
downloaded_file = os.path.join(dataset_path, os.listdir(dataset_path)[0])

# Unzip the file
with zipfile.ZipFile(downloaded_file, 'r') as zip_ref:
    zip_ref.extractall(dataset_path)

