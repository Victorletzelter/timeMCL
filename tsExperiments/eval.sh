#!/bin/bash

declare -a datasets=('electricity' 'exchange' 'solar' 'taxi' 'traffic')
max_epochs=200
num_hyp_ckpt=1
CKPT_JSON='ckpts.json'
CKPT_PATHS=$(cat ${CKPT_JSON})

if [ $# -ge 1 ]; then
    seed=$1
else
    # Error if no seed is provided
    echo "Error: No seed provided"
    exit 1
fi

if [ $# -ge 2 ]; then
    datasets=$2
else
    # Error if no dataset is provided
    echo "Error: No datasets provided"
    exit 1
fi

if [ $datasets == "all" ]; then
    datasets=('electricity' 'exchange' 'solar' 'taxi' 'traffic' 'wiki')
else
    datasets=($datasets)
fi

if [ $# -ge 3 ]; then
    model=$3 # models=('tempflow' 'tactis2' 'timeGrad' 'deepAR' 'transformer_tempflow')
else
    # Error if no model is provided
    echo "Error: No model provided"
    exit 1
fi

if [ $# -ge 4 ]; then
    num_hyp=$4
else
    echo "No num_hyp provided, setting to 1"
    num_hyp=1
fi

if [ $# -ge 5 ]; then
    wta_mode=$5
else
    echo "No wta_mode provided, setting to Null"
    wta_mode=None # Applicable only for timeMCL
fi

for dataset in "${datasets[@]}"; do

    if [ $dataset == "taxi" ]; then
        batch_size=32 # Reduce batch size for taxi dataset due to (possible) memory constraints
    else
        batch_size=200 # Default batch size
    fi

    if [ $model == "tactis2" ]; then
        key=seed_${seed}_${dataset}_${model}_${num_hyp_ckpt}_phase_1
        ckpt_path_phase1=$(echo ${CKPT_PATHS} | jq -r --arg key "${key}" '.[$key]')
        key=seed_${seed}_${dataset}_${model}_${num_hyp_ckpt}_phase_2
        ckpt_path_phase2=$(echo ${CKPT_PATHS} | jq -r --arg key "${key}" '.[$key]')
        if [ -n "${ckpt_path_phase1}" ] && [ "${ckpt_path_phase1}" != "null" ]; then
            echo "Evaluating ${ckpt_path_phase1}" and ${ckpt_path_phase2}
            python train.py ckpt_path_phase1=${ckpt_path_phase1} ckpt_path_phase2=${ckpt_path_phase2} experiment=${dataset}.yaml model=${model}.yaml run_name=seed_${seed}_${dataset}_${model}_${num_hyp}_fromckpt model.params.num_hypotheses=${num_hyp} logger.mlflow.experiment_name=eval_${dataset}_${max_epochs} task_name=eval_${dataset}_${max_epochs} trainer.max_epochs=${max_epochs} seed=3141 train=False test=True data.batch_size=${batch_size}
        else
            echo "ckpt_path is None for ${key}"
        fi
    
    elif [ $model == "timeMCL" ]; then
        if [ $wta_mode == "awta" ]; then
            key=seed_${seed}_${dataset}_${model}_${num_hyp}_awta_temp_ini_10_decay_0.95_scaler_mean
        elif [ $wta_mode == "relaxed-wta" ]; then
            key=seed_${seed}_${dataset}_${model}_${num_hyp}_relaxed-wta_epsilon_0.1_scaler_mean
        fi
        ckpt_path=$(echo ${CKPT_PATHS} | jq -r --arg key "${key}" '.[$key]')
        if [ -n "${ckpt_path}" ] && [ "${ckpt_path}" != "null" ]; then
        echo "Evaluating ${ckpt_path}"
            if [ $wta_mode == "awta" ]; then
                python train.py ckpt_path=${ckpt_path} experiment=${dataset}.yaml model=${model}.yaml run_name=seed_${seed}_${dataset}_${model}_${wta_mode}_${num_hyp}_fromckpt model.params.num_hypotheses=${num_hyp} logger.mlflow.experiment_name=eval_${dataset}_${max_epochs} task_name=eval_${dataset}_${max_epochs} trainer.max_epochs=${max_epochs} seed=3141 train=False test=True model.compute_flops=False model.params.wta_mode=$wta_mode data.batch_size=${batch_size}
            elif [ $wta_mode == "relaxed-wta" ]; then            
                python train.py ckpt_path=${ckpt_path} experiment=${dataset}.yaml model=${model}.yaml run_name=seed_${seed}_${dataset}_${model}_${wta_mode}_${num_hyp}_fromckpt model.params.num_hypotheses=${num_hyp} logger.mlflow.experiment_name=eval_${dataset}_${max_epochs} task_name=eval_${dataset}_${max_epochs} trainer.max_epochs=${max_epochs} seed=3141 train=False test=True model.compute_flops=False model.params.wta_mode=$wta_mode data.batch_size=${batch_size}
            fi
        else
            echo "ckpt_path is None for ${key}"
        fi
    
    else
        key=seed_${seed}_${dataset}_${model}_${num_hyp_ckpt}
        ckpt_path=$(echo ${CKPT_PATHS} | jq -r --arg key "${key}" '.[$key]')
        if [ -n "${ckpt_path}" ] && [ "${ckpt_path}" != "null" ]; then
            echo "Evaluating ${ckpt_path}"
            python train.py ckpt_path=${ckpt_path} experiment=${dataset}.yaml model=${model}.yaml run_name=seed_${seed}_${dataset}_${model}_${num_hyp}_fromckpt model.params.num_hypotheses=${num_hyp} logger.mlflow.experiment_name=eval_${dataset}_${max_epochs} task_name=eval_${dataset}_${max_epochs} trainer.max_epochs=${max_epochs} seed=3141 train=False test=True model.compute_flops=False data.batch_size=${batch_size}
        else
            echo "ckpt_path is None for ${key}"
        fi
    fi
done