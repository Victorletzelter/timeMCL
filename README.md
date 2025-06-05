# Code for *Winner-Takes-All for Multivariate Probabilistic Time Series Forecasting* (ICML 2024)

This repository contains the source code associated to the publication *Winner-Takes-All for Multivariate Probabilistic Time Series Forecasting*, to appear at ICML 2024. 

<div align="center"> <h3> Abstract </h3>  </div>
<div align="justify">

We introduce \texttt{TimeMCL}, a method leveraging the Multiple Choice Learning (MCL) paradigm to forecast multiple plausible time series futures. Our approach employs a neural network with multiple heads and utilizes the Winner-Takes-All (WTA) loss to promote diversity among predictions. MCL has recently gained attention due to its simplicity and ability to address ill-posed and ambiguous tasks. We propose an adaptation of this framework for time-series forecasting, presenting it as an efficient method to predict diverse futures, which we relate to its implicit \textit{quantization} objective. We provide insights into our approach using synthetic data and evaluate it on real-world time series, demonstrating its promising performance at a light computational cost.

</br>

## Repository Structure

In this first code release, we provide code for reproducing the experiments with synthetic data. Code for reproducing the experiments with real-world data will be released soon.

```shell
├── requirements.txt 
├── toy.py # Source file related to synthetic data experiments
├── train.py # Training script
├── plot.py # Script for visualization and inference
├── config/ # Yaml config files containing configuration for training models
├── logs/ # Logs files (including model checkpoints)
├── figures/ # Directory containing generated figures
```

## Synthetic data experiments

This part of the code focuses on **toy experiments** with synthetic data. These toy experiments help illustrate the *TimeMCL* model's underlying theory and demonstrate, on controlled examples, how *TimeMCL* acts as a functional quantizer of stochastic processes. The full code that includes on real-world time-series forecasting will be released soon.

### Datasets

The synthetic experiments use three types of datasets:
- **ARp**: Autoregressive process of order p
- **Brownian Motion**: Standard Brownian motion process
- **Brownian Bridge**: Brownian bridge process

### Setup

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
pip install -r requirements.txt
```

### Training and inference

For training TimeMCL on the synthetic datasets, run:

```shell
python train.py --config=config/ARp.yaml
python train.py --config=config/brownian_motion.yaml
python train.py --config=config/brownian_bridge.yaml
```

Checkpoints will be stored in `logs/trained_timeMCL_{dataset_name}.pth`.

For inference and visualization, please run:

```shell
python plot.py
```

The figure will be stored in `figures/toy_figure.png`.

![Conditional Quantization of Stochastic Processes with TimeMCL.](figures/toy_figure.png)

### Contribution

We welcome contributions! Please feel free to:
- Submit issues for bugs or difficulties
- Create pull requests for improvements
- Suggest better organization or efficiency improvements

### Citation

If our work helped in your research, feel free to give us a star ⭐ or to cite us with the following bibtex code:

```bibtex
@article{timemcl,
  title={Winner-Takes-All for Multivariate Probabilistic Time Series Forecasting},
  author={Cortes, Adrien and Rehm, Remi and Letzelter, Victor},
  journal={International Conference on Machine Learning},
  year={2025}
}
```
