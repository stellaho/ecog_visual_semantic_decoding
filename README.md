# ECoG Visual Semantic Decoding

This repository contains the code used for the experiments reported in our paper:

**[Visual Semantic Decoding of Electrocorticography from Video Stimuli using End-to-End Deep Learning](https://arxiv.org/abs/2607.18923)**

The study investigates visual semantic decoding from intracranial ECoG recordings using multiple neural network architectures and learning strategies.

## Models and learning strategies

The implemented neural network architectures are:

- ResNet1D
- LSNet
- TCN
- EEGNet
- BiLSTM
- Transformer

The learning strategies are:

- Baseline supervised learning
- Mixup
- SimCLR self-supervised learning
- ProtoNet few-shot learning

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

The model architecture can be selected with:

```bash
--encoder {resnet,lsnet,tcn,eegnet,lstm,transformer}
```

For hyperparameter tuning, `run_sweep.py` supports:

```bash
--mode {baseline,mixup,simclr,protonet}
```

For training and evaluation, `run_evaluate.py` supports:

```bash
--mode {baseline,mixup,simclr}
```

ProtoNet uses the separate `run_evaluate_protonet.py` entry point.

For example, a Transformer Mixup sweep can be run with:

```bash
python run_sweep.py \
  --patient_id E02 \
  --encoder transformer \
  --mode mixup \
  --loro_test_id Trn06
```

The corresponding model can then be evaluated with:

```bash
python run_evaluate.py \
  --patient_id E02 \
  --encoder transformer \
  --mode mixup \
  --config_mode per_fold \
  --loro_recordings Trn06
```

Additional options are available through:

```bash
python run_sweep.py --help
python run_evaluate.py --help
python run_evaluate_protonet.py --help
```

Hyperparameter search spaces are provided under `config/visual_decoding/sweeps/`. Selected configurations are saved under `config/visual_decoding/subjects/`.

## Data availability

The ECoG dataset used in this study was originally collected and described in **[Voluntary control of semantic neural representations by imagery with conflicting visual stimulation](https://www.nature.com/articles/s42003-022-03137-x)**. Please contact the authors of the original study for data access.