#!/bin/bash
declare -a num_hyps=('1')
scaler_type="mean"

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
    model=$3
    # models=('tempflow' 'tactis2' 'timeGrad' 'ETS' 'deepAR' 'transformer_tempflow')
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

declare -A max_epochs_dict
max_epochs_dict["crypt"]=101
max_epochs_dict["electricity"]=200
max_epochs_dict["exchange"]=200
max_epochs_dict["solar"]=200
max_epochs_dict["taxi"]=200
max_epochs_dict["traffic"]=200
max_epochs_dict["wiki"]=200

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

if [ $model == "timeMCL" ]; then
    if [ $wta_mode == "awta" ]; then

        wta_mode_params_epsilon=0
        wta_mode_params_temperature_ini=10
        wta_mode_params_scheduler_mode="exponential"
        wta_mode_params_temperature_decay=0.95
        wta_mode_params_temperature_lim=5e-4
        wta_mode_params_wta_after_temperature_lim=True

        for dataset in "${datasets[@]}"; do
            max_epochs=${max_epochs_dict[$dataset]}
            python train.py data=${dataset}.yaml experiment=${dataset}.yaml model=${model}.yaml run_name=seed_${seed}_${dataset}_${model}_${num_hyp}_${wta_mode}_temp_ini_${wta_mode_params_temperature_ini}_decay_${wta_mode_params_temperature_decay}_scaler_${scaler_type} model.params.num_hypotheses=${num_hyp} logger.mlflow.experiment_name=${dataset}_${max_epochs} task_name=${dataset}_${max_epochs} model.params.wta_mode=${wta_mode} model.params.wta_mode_params.epsilon=${wta_mode_params_epsilon} model.params.wta_mode_params.temperature_ini=${wta_mode_params_temperature_ini} model.params.wta_mode_params.scheduler_mode=${wta_mode_params_scheduler_mode} model.params.wta_mode_params.temperature_decay=${wta_mode_params_temperature_decay} model.params.wta_mode_params.temperature_lim=${wta_mode_params_temperature_lim} model.params.wta_mode_params.wta_after_temperature_lim=${wta_mode_params_wta_after_temperature_lim} model.params.scaler_type=${scaler_type} trainer.max_epochs=${max_epochs} test=False model.params.scaler_type=${scaler_type_dict[$dataset]}
        done

    elif [ $wta_mode == "relaxed-wta" ]; then

        wta_mode_params_epsilon=0.1

        for dataset in "${datasets[@]}"; do
            max_epochs=${max_epochs_dict[$dataset]}
            python train.py data=${dataset}.yaml experiment=${dataset}.yaml model=${model}.yaml run_name=seed_${seed}_${dataset}_${model}_${num_hyp}_${wta_mode}_epsilon_${wta_mode_params_epsilon}_scaler_${scaler_type} model.params.num_hypotheses=${num_hyp} logger.mlflow.experiment_name=${dataset}_${max_epochs} task_name=${dataset}_${max_epochs} model.params.wta_mode=${wta_mode} model.params.wta_mode_params.epsilon=${wta_mode_params_epsilon} model.params.scaler_type=${scaler_type} trainer.max_epochs=${max_epochs} test=False
        done

    else
        # Invalid wta_mode
        echo "Error: Invalid wta_mode"
        exit 1
    fi
elif [ $model == "ETS" ]; then
    for dataset in "${datasets[@]}"; do
        max_epochs=${max_epochs_dict[$dataset]}
        python train.py data=${dataset}.yaml experiment=${dataset}.yaml model=${model}.yaml run_name=seed_${seed}_${dataset}_${model}_${num_hyp} model.params.num_hypotheses=${num_hyp} logger.mlflow.experiment_name=eval_${dataset}_${max_epochs} task_name=eval_${dataset}_${max_epochs} trainer.max_epochs=${max_epochs} seed=${seed} test=True
    done
elif [ $model == "tactis2" ]; then
    for dataset in "${datasets[@]}"; do
        max_epochs=${max_epochs_dict[$dataset]}
        python train.py data=${dataset}.yaml experiment=${dataset}.yaml model=${model}.yaml run_name=seed_${seed}_${dataset}_${model}_${num_hyp} model.params.num_hypotheses=${num_hyp} logger.mlflow.experiment_name=train_${dataset}_${max_epochs} task_name=${dataset}_${max_epochs} trainer.max_epochs=${max_epochs} seed=${seed} test=False
    done
else
    for dataset in "${datasets[@]}"; do
        max_epochs=${max_epochs_dict[$dataset]}
        python train.py data=${dataset}.yaml experiment=${dataset}.yaml model=${model}.yaml run_name=seed_${seed}_${dataset}_${model}_${num_hyp} model.params.num_hypotheses=${num_hyp} logger.mlflow.experiment_name=train_${dataset}_${max_epochs} task_name=${dataset}_${max_epochs} trainer.max_epochs=${max_epochs} seed=${seed} test=False model.params.scaler_type=${scaler_type_dict[$dataset]}
    done
fi