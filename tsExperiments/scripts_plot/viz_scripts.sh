#!/bin/bash

declare -a datasets=('electricity' 'solar' 'traffic')
declare -a num_hyps=('8' '16')
num_hyps_ckpt_path=1
seed=$1
model=$2

# Check if there is a third argument
if [ -n "$3" ]; then
    wta_mode=$3
else
    wta_mode=None
fi

cd ..

CKPT_JSON='ckpts.json'
CKPT_PATHS=$(cat ${CKPT_JSON})

for dataset in "${datasets[@]}"; do
    for num_hyp in "${num_hyps[@]}"; do
        batch_size=200
        if [ "${dataset}" == "taxi" ]; then
            batch_size=32
        fi

        if [ "${model}" == "timeMCL" ]; then
            if [ "${wta_mode}" == "awta" ]; then
                key=seed_${seed}_${dataset}_${model}_${num_hyps}_awta_temp_ini_10_decay_0.95_scaler_mean
            else
                key=seed_${seed}_${dataset}_${model}_${num_hyps}_relaxed-wta_epsilon_0.1_scaler_mean
            fi
        elif [ "${model}" == "tactis2" ]; then
            key_phase_1=seed_${seed}_${dataset}_${model}_${num_hyps_ckpt_path}_phase_1
            key_phase_2=seed_${seed}_${dataset}_${model}_${num_hyps_ckpt_path}_phase_2
        else
            key=seed_${seed}_${dataset}_${model}_${num_hyps_ckpt_path}
        fi

        if [ "${model}" == "tactis2" ]; then
            matching_count=$(echo ${CKPT_PATHS} | jq -r "to_entries | map(select(.key | contains(\"${key_phase_1}\"))) | length")
            
            if [ "${matching_count}" -eq 1 ]; then
                # ckpt_path=$(echo ${CKPT_PATHS} | jq -r "to_entries | .[] | select(.key | contains(\"${key}\")) | .value" | head -n 1)
                ckpt_path_phase_1=$(echo ${CKPT_PATHS} | jq -r "to_entries | .[] | select(.key | contains(\"${key_phase_1}\")) | .value" | head -n 1)
                ckpt_path_phase_2=$(echo ${CKPT_PATHS} | jq -r "to_entries | .[] | select(.key | contains(\"${key_phase_2}\")) | .value" | head -n 1)
                echo "ckpt_path_phase_1: ${ckpt_path_phase_1}"
                echo "ckpt_path_phase_2: ${ckpt_path_phase_2}"
                # check if the ckpt_path is not None
                if [ -n "${ckpt_path_phase_1}" ] && [ "${ckpt_path_phase_1}" != "null" ] && [ -n "${ckpt_path_phase_2}" ] && [ "${ckpt_path_phase_2}" != "null" ]; then
                    python train.py ckpt_path_phase1=${ckpt_path_phase_1} ckpt_path_phase2=${ckpt_path_phase_2} experiment=${dataset}.yaml model=${model}.yaml run_name=seed_${seed}_${dataset}_${model}_${num_hyp} model.params.num_hypotheses=${num_hyp} logger.mlflow.experiment_name=visual_${dataset} task_name=visual_${dataset} train=False test=False model.compute_flops=False model.plot_forecasts=True data.batch_size=${batch_size}
                else
                    echo "ckpt_path is None for ${key}"
                fi
            else
                echo "Found ${matching_count} matches for key '${key}'. Expected exactly 1 match."
                echo "Matching keys:"
                echo ${CKPT_PATHS} | jq -r "to_entries | map(select(.key | contains(\"${key}\"))) | .[].key"
            fi
        else       
            # Count matching keys
            matching_count=$(echo ${CKPT_PATHS} | jq -r "to_entries | map(select(.key | contains(\"${key}\"))) | length")
            
            if [ "${matching_count}" -eq 1 ]; then
                ckpt_path=$(echo ${CKPT_PATHS} | jq -r "to_entries | .[] | select(.key | contains(\"${key}\")) | .value" | head -n 1)
                # check if the ckpt_path is not None
                if [ -n "${ckpt_path}" ] && [ "${ckpt_path}" != "null" ]; then
                    python train.py ckpt_path=${ckpt_path} experiment=${dataset}.yaml model=${model}.yaml run_name=seed_${seed}_${dataset}_${model}_${num_hyp}_amcl model.params.num_hypotheses=${num_hyp} logger.mlflow.experiment_name=visual_${dataset} task_name=visual_${dataset} train=False test=False model.compute_flops=False trainer.validation_only=True model.plot_forecasts=True data.batch_size=${batch_size}
                else
                    echo "ckpt_path is None for ${key}"
                fi
            else
                echo "Found ${matching_count} matches for key '${key}'. Expected exactly 1 match."
                echo "Matching keys:"
                echo ${CKPT_PATHS} | jq -r "to_entries | map(select(.key | contains(\"${key}\"))) | .[].key"
            fi
        fi
    done
done

