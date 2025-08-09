#!/bin/bash

declare -a datasets=('electricity' 'exchange' 'solar' 'taxi' 'traffic' 'wiki')
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

if [ $seed == "all_seeds" ]; then
    seeds=('42' '3141' '3142' '3143')
else
    seeds=($seed)
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

if [ $# -ge 6 ]; then
    echo "Using provided scaler_type"
    declare -A scaler_type_dict
    scaler_type_dict["crypt"]=$6
    scaler_type_dict["electricity"]=$6
    scaler_type_dict["exchange"]=$6
    scaler_type_dict["solar"]=$6
    scaler_type_dict["taxi"]=$6
    scaler_type_dict["traffic"]=$6
    scaler_type_dict["wiki"]=$6
else
    echo "No scaler_type provided, setting to the default scaler_type"
    declare -A scaler_type_dict
    scaler_type_dict["crypt"]="mean_std"
    scaler_type_dict["electricity"]="mean"
    scaler_type_dict["exchange"]="mean"
    scaler_type_dict["solar"]="mean"
    scaler_type_dict["taxi"]="mean"
    scaler_type_dict["traffic"]="mean"
    scaler_type_dict["wiki"]="mean"
    if [ $model == "deepAR" ]; then
        scaler_type_dict["crypt"]="mean" #For deepAR, we retained mean scaler on crypt because it performs better than mean_std scaler.
    fi
fi

declare -A max_epochs_dict
max_epochs_dict["crypt"]=101
max_epochs_dict["electricity"]=200
max_epochs_dict["exchange"]=200
max_epochs_dict["solar"]=200
max_epochs_dict["taxi"]=200
max_epochs_dict["traffic"]=200
max_epochs_dict["wiki"]=200

for dataset in "${datasets[@]}"; do
    max_epochs=${max_epochs_dict[$dataset]}

    scaler_type=${scaler_type_dict[$dataset]}

    if [ $dataset == "taxi" ]; then
        batch_size=32 # Reduce batch size for taxi dataset due to (possible) memory constraints
    elif [ $dataset == "crypt" ]; then
        batch_size=64
    else
        batch_size=200 # Default batch size
    fi

    for seed in "${seeds[@]}"; do

        if [ $model == "tactis2" ]; then
            key=seed_${seed}_${dataset}_${model}_${num_hyp_ckpt}_phase_1
            ckpt_path_phase1=$(echo ${CKPT_PATHS} | jq -r --arg key "${key}" '.[$key]')
            key=seed_${seed}_${dataset}_${model}_${num_hyp_ckpt}_phase_2
            ckpt_path_phase2=$(echo ${CKPT_PATHS} | jq -r --arg key "${key}" '.[$key]')
            if [ -n "${ckpt_path_phase1}" ] && [ "${ckpt_path_phase1}" != "null" ]; then
                echo "Evaluating ${ckpt_path_phase1}" and ${ckpt_path_phase2}
                python train.py data=${dataset}.yaml ckpt_path_phase1=${ckpt_path_phase1} ckpt_path_phase2=${ckpt_path_phase2} experiment=${dataset}.yaml model=${model}.yaml run_name=seed_${seed}_${dataset}_${model}_${num_hyp}_fromckpt model.params.num_hypotheses=${num_hyp} logger.mlflow.experiment_name=eval_${dataset}_${max_epochs} task_name=eval_${dataset}_${max_epochs} trainer.max_epochs=${max_epochs} seed=3141 train=False test=True data.batch_size=${batch_size}
            else
                echo "ckpt_path is None for ${key}"
            fi
        
        elif [ $model == "timeMCL" ]; then
            if [ $wta_mode == "awta" ]; then
                if [ $scaler_type == "mean" ]; then
                    key=seed_${seed}_${dataset}_${model}_${num_hyp}_awta_temp_ini_10_decay_0.95_scaler_mean
                else
                    key=seed_${seed}_${dataset}_${model}_${num_hyp}_awta_temp_ini_10_decay_0.95_scaler_${scaler_type}
                fi
            elif [ $wta_mode == "relaxed-wta" ]; then
                if [ $scaler_type == "mean_std" ]; then
                    key=seed_${seed}_${dataset}_${model}_${num_hyp}_relaxed-wta_epsilon_0.1_scaler_mean
                else
                    key=seed_${seed}_${dataset}_${model}_${num_hyp}_relaxed-wta_epsilon_0.1_scaler_${scaler_type}
                fi
            fi
            ckpt_path=$(echo ${CKPT_PATHS} | jq -r --arg key "${key}" '.[$key]')
            if [ -n "${ckpt_path}" ] && [ "${ckpt_path}" != "null" ]; then
            echo "Evaluating ${ckpt_path}"
                if [ $wta_mode == "awta" ]; then
                    python train.py data=${dataset}.yaml ckpt_path=${ckpt_path} experiment=${dataset}.yaml model=${model}.yaml run_name=seed_${seed}_${dataset}_${model}_${wta_mode}_${num_hyp}_fromckpt model.params.num_hypotheses=${num_hyp} logger.mlflow.experiment_name=eval_${dataset}_${max_epochs} task_name=eval_${dataset}_${max_epochs} trainer.max_epochs=${max_epochs} seed=3141 train=False test=True model.compute_flops=False model.params.wta_mode=$wta_mode data.batch_size=${batch_size} model.params.scaler_type=${scaler_type}
                elif [ $wta_mode == "relaxed-wta" ]; then            
                    python train.py data=${dataset}.yaml ckpt_path=${ckpt_path} experiment=${dataset}.yaml model=${model}.yaml run_name=seed_${seed}_${dataset}_${model}_${wta_mode}_${num_hyp}_fromckpt model.params.num_hypotheses=${num_hyp} logger.mlflow.experiment_name=eval_${dataset}_${max_epochs} task_name=eval_${dataset}_${max_epochs} trainer.max_epochs=${max_epochs} seed=3141 train=False test=True model.compute_flops=False model.params.wta_mode=$wta_mode data.batch_size=${batch_size} model.params.scaler_type=${scaler_type}
                fi
            else
                echo "ckpt_path is None for ${key}"
            fi
        
        else
            key=seed_${seed}_${dataset}_${model}_${num_hyp_ckpt}
            ckpt_path=$(echo ${CKPT_PATHS} | jq -r --arg key "${key}" '.[$key]')
            if [ -n "${ckpt_path}" ] && [ "${ckpt_path}" != "null" ]; then
                echo "Evaluating ${ckpt_path}"
                python train.py data=${dataset}.yaml ckpt_path=${ckpt_path} experiment=${dataset}.yaml model=${model}.yaml run_name=seed_${seed}_${dataset}_${model}_${num_hyp}_fromckpt model.params.num_hypotheses=${num_hyp} logger.mlflow.experiment_name=eval_${dataset}_${max_epochs} task_name=eval_${dataset}_${max_epochs} trainer.max_epochs=${max_epochs} seed=3141 train=False test=True model.compute_flops=False data.batch_size=${batch_size} model.params.scaler_type=${scaler_type}
            else
                echo "ckpt_path is None for ${key}"
            fi
        fi
    done
done