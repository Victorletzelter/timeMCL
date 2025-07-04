#!/bin/bash

declare -a models=('tactis2')
declare -a num_hyps=('16')
max_epochs=1
num_hyp_ckpt=16
dataset='exchange'

if [ $# -ge 1 ]; then
    seed=$1
else
    seed=3142
fi

for num_hyp in "${num_hyps[@]}"; do
    for model in "${models[@]}"; do
        # Tactis
        python train.py experiment=${dataset}.yaml model=${model}.yaml run_name=seed_${seed}_${dataset}_${model}_${num_hyp}_compute model.params.num_hypotheses=${num_hyp} logger.mlflow.experiment_name=compute_${dataset}_1 task_name=compute_${dataset} trainer.max_epochs=${max_epochs} seed=${seed} train=True test=False model.compute_flops=True model.plot_forecasts=False model.params.num_parallel_samples=${num_hyp} data.num_batches_per_epoch=1 data.num_batches_val_per_epoch=1
        # TempFlow
        python train.py experiment=${dataset}.yaml model=${model}.yaml run_name=seed_${seed}_${dataset}_${model}_${num_hyp}_compute model.params.num_hypotheses=${num_hyp} logger.mlflow.experiment_name=compute_${dataset}_1 task_name=compute_${dataset} trainer.max_epochs=${max_epochs} seed=3141 train=True test=False model.compute_flops=True model.params.num_parallel_samples=${num_hyp} data.num_batches_per_epoch=1 data.num_batches_val_per_epoch=1
        # DeepAR
        python train.py experiment=${dataset}.yaml model=${model}.yaml run_name=seed_${seed}_${dataset}_${model}_${num_hyp}_dist_type_${dist_type}_compute model.params.num_hypotheses=${num_hyp} logger.mlflow.experiment_name=compute_${dataset}_1 task_name=compute_${dataset} model.params.dist_type=${dist_type} train=True test=False trainer.validation_only=True trainer.max_epochs=1 data.num_batches_per_epoch=1 data.num_batches_val_per_epoch=1 model.compute_flops=True model.plot_forecasts=False model.params.num_parallel_samples=${num_hyp}
        # timeMCL
        python train.py experiment=${dataset}.yaml model=${model}.yaml run_name=seed_${seed}_${dataset}_${model}_${num_hyp}_${wta_mode}_scaler_${scaler_type}_compute model.params.num_hypotheses=${num_hyp} logger.mlflow.experiment_name=compute_${dataset}_1 task_name=compute_${dataset} model.params.wta_mode=${wta_mode} model.params.scaler_type=${scaler_type} train=True test=False trainer.max_epochs=1 data.num_batches_per_epoch=1 data.num_batches_val_per_epoch=1 trainer.validation_only=True model.compute_flops=True model.params.num_parallel_samples=1 
        # TimeGrad
        python train.py experiment=${dataset}.yaml model=${model}.yaml run_name=seed_${seed}_${dataset}_${model}_${num_hyp}_compute model.params.num_hypotheses=${num_hyp} logger.mlflow.experiment_name=compute_${dataset}_1 task_name=compute_${dataset} train=True test=False trainer.max_epochs=1 data.num_batches_per_epoch=1 data.num_batches_val_per_epoch=1 trainer.validation_only=True model.compute_flops=True model.params.num_parallel_samples=${num_hyp}
        # TransformerTempFlow
        python train.py experiment=${dataset}.yaml model=${model}.yaml run_name=seed_${seed}_${dataset}_${model}_${num_hyp}_compute model.params.num_hypotheses=${num_hyp} logger.mlflow.experiment_name=compute_${dataset}_1 task_name=compute_${dataset} trainer.max_epochs=${max_epochs} seed=3141 train=True test=False model.compute_flops=True model.params.num_parallel_samples=${num_hyp} trainer.max_epochs=1 data.num_batches_per_epoch=1 data.num_batches_val_per_epoch=1 trainer.validation_only=True
    done
done