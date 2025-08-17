#!/bin/bash

declare -a datasets=('crypt')
declare -a models=('tactis2')
num_hyps_ckpt_path=16

if [ $# -ge 1 ]; then
    seed=$1
else
    seed=3142
fi

if [ $# -ge 2 ]; then
    num_hyp=$2
else
    # Raise error if no num_hyp is provided
    echo "No num_hyp provided"
    exit 1
fi

cd ..

num_hyp_ckpt=1
CKPT_JSON='../ckpts.json'
CKPT_PATHS=$(cat ${CKPT_JSON})
dataset='crypt'
model='tactis2'
seed=3142
key=seed_${seed}_${dataset}_${model}_${num_hyp_ckpt}_phase_1
ckpt_path_phase1=$(echo ${CKPT_PATHS} | jq -r --arg key "${key}" '.[$key]')
key=seed_${seed}_${dataset}_${model}_${num_hyp_ckpt}_phase_2
ckpt_path_phase2=$(echo ${CKPT_PATHS} | jq -r --arg key "${key}" '.[$key]')

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        # for num_hyp in "${num_hyps[@]}"; do
        python train.py ckpt_path_phase1=${path_phase1} ckpt_path_phase2=${path_phase2} experiment=${dataset}.yaml model=${model}.yaml run_name=seed_${seed}_${dataset}_${model}_${num_hyp} model.params.num_hypotheses=${num_hyp} logger.mlflow.experiment_name=setup_viz task_name=setup_viz trainer.max_epochs=${max_epochs} seed=${seed} train=False test=False model.plot_forecasts=True visualize_specific_date=True
        # done
    done
done