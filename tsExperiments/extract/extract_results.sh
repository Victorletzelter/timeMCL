#!/bin/bash

declare -a datasets=('electricity' 'exchange' 'solar' 'traffic' 'taxi' 'wiki')

PROJECT_ROOT=$(python set_project_root.py)
export PROJECT_ROOT

cd ${PROJECT_ROOT}/tsExperiments/results

# Delete the saved_csv directory if it exists
if [ -d "saved_csv" ]; then
    rm -r saved_csv
fi

mkdir saved_csv

cd ${PROJECT_ROOT}/tsExperiments/download

for dataset_name in "${datasets[@]}"
do
    python scripts_download_csv.py --experiment_name=eval_${dataset_name}_200 --save_dir=${PROJECT_ROOT}/tsExperiments/results/saved_csv
done

python scripts_download_csv.py --experiment_name=eval_crypt_101 --save_dir=${PROJECT_ROOT}/tsExperiments/results/saved_csv