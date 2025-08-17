# Winner-Takes-All for Multivariate Probabilistic Time Series Forecasting (ICML 2025)

This repository contains the source code associated with the publication *Winner-Takes-All for Multivariate Probabilistic Time Series Forecasting* (ICML 2025). 

<div align="center"> <h3> Abstract </h3>  </div>
<div align="justify">

We introduce **TimeMCL**, a method leveraging the Multiple Choice Learning (MCL) paradigm to forecast multiple plausible time series futures. Our approach employs a neural network with multiple heads and utilizes the Winner-Takes-All (WTA) loss to promote diversity among predictions. MCL has recently gained attention due to its simplicity and ability to address ill-posed and ambiguous tasks. We propose an adaptation of this framework for time-series forecasting, presenting it as an efficient method to predict diverse futures, which we relate to its implicit *quantization* objective. We provide insights into our approach using synthetic data and evaluate it on real-world time series, demonstrating its promising performance at a light computational cost.

</br>

## 📁 Repository Structure

```shell
└── demo # Quick start notebooks
└── toy # Synthetic data experiments
└── tsExperiments # Real-world dataset
```

## 🚀 Fast Demo 

For quick prototyping, we provide a notebook in [TimeMCL-Solar.ipynb](demo/TimeMCL-Solar.ipynb) for training, inference, evaluation, and plotting on real-world time series data. A similar demonstration for synthetic data experiments is available in [toy.ipynb](demo/toy.ipynb).

## 🔥 Synthetic Data Experiments

This part of the code focuses on **toy experiments** with synthetic data. These toy experiments help illustrate the *TimeMCL* model's underlying theory and demonstrate, using controlled examples, how *TimeMCL* acts as a functional quantizer for stochastic processes.

### 🗃️ Datasets

The synthetic experiments use three types of datasets:
- **ARp**: Autoregressive process of order p
- **Brownian Motion**: Standard Brownian motion process
- **Brownian Bridge**: Brownian bridge process

### 🔨 Setup

If you have conda, you can create an environment with:

```shell
conda create -n synth_env -y python=3.10.15
```

Then, close and re-open your shell, and activate your environment:

```shell
conda activate synth_env
```

Install the required dependencies:

```shell
cd toy
pip install -r requirements.txt
```

LaTeX can optionally be used for plot rendering. Install it with: `sudo apt-get install -y dvipng texlive-latex-extra texlive-fonts-recommended cm-super`.

### 🔄 Training and Inference

For training TimeMCL on the synthetic datasets, run:

```shell
python train.py ARp.yaml
python train.py brownian_motion.yaml
python train.py brownian_bridge.yaml
```

Checkpoints will be stored in `toy/logs/trained_timeMCL_{dataset_name}.pth`.

For inference and visualization, please run:

```shell
python plot.py
```

The figure will be stored in `toy/figures/toy_figure.png`.

![Conditional Quantization of Stochastic Processes with TimeMCL.](./toy/figures/toy_figure.png)

## 🔥 Real Datasets Experiments

### 🔨 Setup

To reproduce the experiments on real-world datasets, you can set up an environment as follows. This setup assumes you have Python3 installed (we used Python 3.10.15).

```shell
cd tsExperiments
bash setup-env.sh
```

The environment variable ENV_DIR should then be set. After running the setup script, the environment variable ENV_DIR will be set. Activate the environment with:

```shell
source $ENV_DIR/bin/activate
```

The gluonts datasets (`electricity`, `exchange`, `solar`, `taxi`, `traffic`, `wiki`) will be downloaded automatically under `~/.gluonts/datasets` when calling for the first time the `get_dataset` function from gluonts. These datasets, along with the hourly crypto-currency dataset used in Section 6.4 and Appendix C.4 can be downloaded with: 
```bash
python download_datasets.py
```

### 🔄 Training

To train timeMCL with 16 hypotheses on the datasets (`electricity`, `exchange`, `solar`, `taxi`, `traffic`, `wiki`) using seed 3141, and with annealed and relaxed variants (with default parameters). You can set `num_hyps=16`, `seed=3141`, `datasets=all` and run the following commands:

```shell
bash train.sh $seed $datasets timeMCL $num_hyps awta # For the annealed variant
bash train.sh $seed $datasets timeMCL $num_hyps relaxed-wta # For the relaxed variant
```

To train the different baselines, on all the datasets use the following commands:

```shell
bash train.sh $seed $datasets tempflow
bash train.sh $seed $datasets tactis2
bash train.sh $seed $datasets timeGrad
bash train.sh $seed $datasets ETS
bash train.sh $seed $datasets deepAR
bash train.sh $seed $datasets transformer_tempflow
```

The experiment on the crypto-currency dataset can be run with the above commands by setting `num_hyps=4` and `datasets=crypt`.

If you have the resources, you can run the above trainings with several seeds, to be able to compute standard deviations on the scores of each baseline.

When launching the above trainings, the logs will be saved in `tsExperiments/logs` following the [Hydra](https://github.com/facebookresearch/hydra) template, that is organized as follows:

```shell
└── tsExperiments 
  └── logs
    └── <experiment_name> # By default: <dataset_name>_<num_epochs>
      └── runs
        └── <run_folder_name> # By Default: <start_run_time>_<dataset_name>_<model>_<num_hypotheses>_<model_specificities>, where start_run_time is in the form %Y-%m-%d_%H-%M-%S and <model_specificities> applicable only for the timeMCL runs. 
          ├── Prediction_plot.png # Visualisations of the predictions on the test set (if enabled).
          ├── {context_points,forecast_length,freq_type,hypothesis_forecasts,is_mcl,target_df}.pkl # Raw data to reproduce the plot if needed.
          └── .hydra # Folder to save the config yaml files associated with the run
          └── checkpoints # Folder where the checkpoints are saved. By default, it contains epoch_{best_epoch_number}.ckpt and last.ckpt, where the best epoch number is based on the validation loss. 
          └── tensorboard # Folder that contains tensorboard event files. 
```

### 🔄 Inference and evaluation

#### 📥 Trained models checkpoints path extraction 

If you performed the training above, the checkpoints should be stored in `tsExperiments/logs`. In case you just want to launch evaluation with our models, the later can be downloaded with `python tsExperiments/download_ckpts.py`.

The general command to evaluate a model with a given checkpoint path, the command takes this form (except for tactis2):

```shell
python train.py ckpt_path=${ckpt_path} experiment=${dataset}.yaml model=${model}.yaml run_name=seed_${training_seed}_${dataset}_${model}_${num_hyp} model.params.num_hypotheses=${num_hyp} logger.mlflow.experiment_name=eval_${dataset} task_name=eval_${dataset} seed=${inference_seed} train=False test=True
```
where 
* `model` is the model to be evaluated, following the file names in `configs/model`.
* `training_seed` and `inference_seed` refer respectively to the seed that was used to train the model, and the seed that is used for evaluation.
* `dataset` is the dataset used for evaluation.
* `num_hyp` is the number of hypotheses (or samples) used for inference.

For Tactis2, instead of `ckpt_path`, set `ckpt_path_phase1` and `ckpt_path_phase2` to the paths of the models for phase 1 and phase 2, respectively.

To avoid the burden of extracting each checkpoint path by hand, we provide `extract_ckpts.py`, a python script to automate checkpoint path extraction. It can be executed by running:

```shell
cd extract
python extract_ckpts.py --log_dir=tsExperiments/logs
```

where `--log_dir` specifices the logging directories. Then, a json file named `ckpts.json` and containing the checkpoint paths will be created in the folder `tsExperiments/`.

#### 📊 Inference, metrics computation and results extraction

In this case, the full evaluation can be performed by first installing `jq`, e.g., with `sudo apt-get update ; sudo apt-get install jq -y --fix-missing`. Then, having `seed`, `num_hyps` and `datasets` defined the evaluation scripts can be launched
```shell
bash eval.sh $seed $datasets $timeMCL $num_hyps awta
bash eval.sh $seed $datasets $timeMCL $num_hyps relaxed-wta
```
and for the baselines:
```shell
bash eval.sh $seed $datasets $tempflow $num_hyps
bash eval.sh $seed $datasets $tactis2 $num_hyps
bash eval.sh $seed $datasets $timeGrad $num_hyps
bash eval.sh $seed $datasets $deepAR $num_hyps
bash eval.sh $seed $datasets $transformer_tempflow $num_hyps
```
To launch those scripts with the four random states provided in the checkpoints by settings `seed=all_seeds` (e.g., with `bash eval.sh all_seeds all timeMCL 16 awta`). Update the `all_seeds` arrays in `eval.sh` accordingly if you ran the trainings by yourself.

The results can then be visualized with the integrated MLFLow logger. To do so, please move to the created MLFLow dir with `cd tsExperiments/logs/mlflow`. To do so, please define a port number, e.g., `PORT=5066`. Then, run 
```shell
GUNICORN_CMD_ARGS="--timeout 0" mlflow ui -p $PORT
``` 
The results can then be analyzed in `http://localhost:{PORT}/`.

The full results will be downloaded as csv files (one for each dataset) in `tsExperiments/results/saved_csv` by running 
```shell
cd extract
bash extract_results.sh
```
To generate LaTeX tables from these results, run:
```shell
python extract_tables.py
``` 
The LaTeX tables will then be generated in `latex_tables_output.txt`.

#### 📈 Visualisation

To reproduce visualisations from Figures 3, 7 and 8, first, you need to have run inference with the arg `model.plot_forecasts=True`. We provide a script to run the required inference (without computing the metrics) for each baseline, that can be run by first setting a seed number to plot (e.g., seed=3141), and run it with:
```bash
cd tsExperiments/scripts_plot
bash viz_scripts.sh $seed timeMCL awta
bash viz_scripts.sh $seed timeMCL relaxed-wta
bash viz_scripts.sh $seed tempflow
bash viz_scripts.sh $seed tactis2
bash viz_scripts.sh $seed timeGrad
bash viz_scripts.sh $seed transformer_tempflow
```

Then run 
```bash
cd tsExperiments/scripts_plot
python plotting.py
```
The Figures will be saved in `logs/plots/{dataset_name}`.

To reproduce the Figures 4 and 6 from the crypto-currency dataset, run:
```shell
cd tsExperiments/scripts_plot
bash scripts.sh
```
Then run: 
```bash
python plot_crypt.py 
python plot_crypt_grid.py
```

#### ⚡ Computational cost evaluation

We provide a dedicated script, `flops.sh` in `tsExperiments/computation_flops` to compute floating point operations (with randomly initialized models). It can be executed as `cd tsExperiments/computation_flops ; bash flops.sh`.
We performed runtime evaluation on a single NVIDIA GeForce RTX 2080 Ti. To evaluate runtime with your own machine, please execute the following script:

```shell
cd tsExperiments/computation_time ; python evaluate_time.py
```

The run-time results will be stored in `tsExperiments/computation_time/results/` and can be turned into a table by following the instructions in the `tsExperiments/computation_time/extract_table.py` file.

### 👍 Acknowledgments

This work was funded by the French Association for Technological Research (ANRT CIFRE contract 2022-1854) and the LISTEN Laboratory of Télécom Paris. It also benefited from access to the HPC resources of IDRIS (allocation 2024-AD011014345) by GENCI. We are grateful to the reviewers for their insightful comments.

This repository contains source code adapted from the following Github repositories, for which we greatly thank the authors:

[pytorch-ts](https://github.com/zalandoresearch/pytorch-ts) (under MIT License)

[tactis](https://github.com/servicenow/tactis) (under Apache License 2.0)

[fvcore](https://github.com/facebookresearch/fvcore) (under Apache License 2.0)

[gluonts](https://github.com/awslabs/gluonts) (under Apache License 2.0)

[statsmodels](https://github.com/statsmodels/statsmodels) (under BSD 3-Clause "New" or "Revised" License)

[pytorch-lightning](https://github.com/Lightning-AI/pytorch-lightning/blob/master/LICENSE) (under Apache 2.0 License)

[Hydra](https://github.com/facebookresearch/hydra) (under MIT License)

### 🤝 Contribution

We welcome contributions! Please feel free to:
- Submit issues for bugs or difficulties
- Create pull requests for improvements
- Suggest better organization or efficiency improvements

### ✏️ Citation

If our work helped in your research, feel free to give us a star ⭐ or to cite us with the following bibtex code:

```bibtex
@inproceedings{timemcl,
  title={Winner-takes-all for Multivariate Probabilistic Time Series Forecasting},
  author={Cort{\'e}s, Adrien and Rehm, R{\'e}mi and Letzelter, Victor},
  booktitle={International Conference on Machine Learning},
  year={2025}
}
```
