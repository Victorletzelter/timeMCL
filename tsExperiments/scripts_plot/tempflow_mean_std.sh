#!/bin/bash

declare -a datasets=('crypt')
declare -a models=('tempflow')
declare -a num_hyps=('4')

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

if [ $# -ge 3 ]; then
    max_epochs=$3
else
    max_epochs=200
fi

num_hyp_ckpt=1
CKPT_JSON='../ckpts.json'
CKPT_PATHS=$(cat ${CKPT_JSON})
dataset='crypt'
model='tempflow'
seed=3142
key=seed_${seed}_${dataset}_${model}_${num_hyp_ckpt}
ckpt_path=$(echo ${CKPT_PATHS} | jq -r --arg key "${key}" '.[$key]')

cd ..

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        # for num_hyp in "${num_hyps[@]}"; do
        python train.py ckpt_path=${path} experiment=${dataset}.yaml model=${model}.yaml run_name=seed_${seed}_${dataset}_${model}_${num_hyp}_meanstd model.params.num_hypotheses=${num_hyp} logger.mlflow.experiment_name=setup_viz task_name=setup_viz trainer.max_epochs=${max_epochs} seed=${seed} train=False test=False model.compute_flops=False model.plot_forecasts=True model.params.scaler_type="mean_std" model.params.div_by_std=True visualize_specific_date=True
        # done
    done
done