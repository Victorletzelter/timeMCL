#!/bin/bash
declare -a models=('tempflow')
declare -a num_hyps=('1')
max_epochs=200

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

cd tsExperiments

for dataset in "${datasets[@]}"; do
    python train.py experiment=${dataset}.yaml model=${model}.yaml run_name=seed_${seed}_${dataset}_${model}_${num_hyp}_embed_dim_0 model.params.num_hypotheses=${num_hyp} logger.mlflow.experiment_name=setup task_name=setup model.params.embedding_dimension=0 trainer.max_epochs=${max_epochs} seed=${seed}
done