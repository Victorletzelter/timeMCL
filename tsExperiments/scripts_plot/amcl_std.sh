#!/bin/bash

declare -a datasets=('crypt')
declare -a specificities=('temp_ini_10_decay_0.95_scaler_mean')
declare -a models=('timeMCL')
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

wta_mode_params_temperature_ini=10
wta_mode_params_scheduler_mode="exponential"
wta_mode_params_temperature_decay=0.95
wta_mode_params_temperature_lim=5e-4
wta_mode_params_wta_after_temperature_lim=True

num_hyp=4
CKPT_JSON='../ckpts.json'
CKPT_PATHS=$(cat ${CKPT_JSON})
dataset='crypt'
model='timeMCL'
seed=3142
key=seed_${seed}_${dataset}_${model}_${num_hyp}_awta_temp_ini_10_decay_0.95_scaler_mean_std
ckpt_path=$(echo ${CKPT_PATHS} | jq -r --arg key "${key}" '.[$key]')

cd ..

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        # for num_hyp in "${num_hyps[@]}"; do
        for specificity in "${specificities[@]}"; do
            python train.py ckpt_path=${path} data=${dataset} experiment=${dataset}.yaml model=${model}.yaml run_name=seed_${seed}_${dataset}_${model}_${num_hyp}_awta_${specificity}_meanstd model.params.num_hypotheses=${num_hyp} logger.mlflow.experiment_name=setup_viz train=False test=False task_name=setup_viz model.params.wta_mode=awta model.compute_flops=False model.params.wta_mode_params.temperature_ini=${wta_mode_params_temperature_ini} model.params.wta_mode_params.scheduler_mode=${wta_mode_params_scheduler_mode} model.params.wta_mode_params.temperature_decay=${wta_mode_params_temperature_decay} model.params.wta_mode_params.temperature_lim=${wta_mode_params_temperature_lim} model.params.wta_mode_params.wta_after_temperature_lim=${wta_mode_params_wta_after_temperature_lim} seed=${seed} trainer.max_epochs=1 model.params.scaler_type=mean_std model.params.div_by_std=True visualize_specific_date=True
        done
        # done
    done
done
