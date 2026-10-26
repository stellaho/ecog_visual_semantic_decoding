# ECoG Visual Semantic Decoding

This repository contains the code used for the experiments reported in our paper:

**[Visual Semantic Decoding of Electrocorticography from Video Stimuli using End-to-End Deep Learning](https://arxiv.org/abs/2607.18923)**

The study investigates visual semantic decoding from intracranial ECoG recordings using several neural network architectures and training strategies.

## Repository structure

- `model/` – neural network encoders and classifiers
- `meta_learning/` – ProtoNet models and episodic sampling
- `prep_data/` and `sig_prep/` – signal preprocessing and data preparation
- `dataset/` – dataset loading
- `scripts/` – training and evaluation code
- `sweep/` and `config/` – hyperparameter search and experiment configurations
- `run_sweep.py` – hyperparameter tuning
- `run_evaluate.py` – model training and evaluation
- `run_evaluate_protonet.py` – ProtoNet training and evaluation

## Running experiments

Install the required packages with:

```bash
python -m pip install -r requirements.txt
```

Weights & Biases is used for hyperparameter tuning:

```bash
wandb login
```

The main experiment entry points are:

```bash
python run_sweep.py --help
python run_evaluate.py --help
python run_evaluate_protonet.py --help
```

Experiment settings and hyperparameter search spaces are provided under `config/`.

## Data availability

The ECoG dataset used in this study was originally collected and described in **[Voluntary control of semantic neural representations by imagery with conflicting visual stimulation](https://www.nature.com/articles/s42003-022-03137-x)**. Please contact the authors of the original study for data access.