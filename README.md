# Weight-Space-Learning

This repository studies weight-space learning by representing trained neural
network weights as image-like tensors.

The first pipeline creates a meta-dataset where each meta-datapoint contains:

1. one sampled synthetic dataset,
2. one target neural network trained on that dataset,
3. the trained target-network weights,
4. one deterministic weight-image representation,
5. metadata plus train/test metrics.

The professor's `Rügamer code` folder was used only as context for the original
weight-to-image idea. This repository is rebuilt from scratch with a clean,
modular PyTorch structure.

## Repository Structure

```text
config/                 YAML experiment configs
data/                   generated artifacts, ignored by Git except .gitkeep files
src/dataset_gen/        synthetic dataset generators and collection builder
src/network_learning/   target-network models and training
src/image_gen/          weight extraction and image layout code
src/cond_AE/            minimal AE/CAE scaffold for later work
src/evaluation/         metrics and plotting helpers
src/utils/              config, IO, paths, and seed helpers
scripts/                runnable pipeline entry points
tests/                  minimal smoke tests
```

## Setup

Use a Python environment with PyTorch installed, then run:

```bash
pip install -r requirements.txt
```

## Run The First Pipeline

Run all commands from the repository root:

```bash
python scripts/01_generate_datasets.py --config config/default.yaml
python scripts/02_train_target_networks.py --config config/default.yaml
python scripts/03_generate_weight_images.py --config config/default.yaml
```

Generated artifacts are written under:

```text
data/processed/       generated datasets and dataset metadata
data/model_zoo/       trained model checkpoints, metrics, and metadata
data/weight_images/   raw extracted weights, image tensors, and layout metadata
```

These generated files are ignored by Git. The tracked `.gitkeep` files preserve
the empty data-folder structure.

## Tests

```bash
python -m pytest
```

## Configuration

The main config is `config/default.yaml`. It controls:

- number of meta-datapoints `M`,
- train/test points per dataset,
- synthetic generator, currently `moons` or `blobs`,
- generator parameters such as `noise`, `centers`, and `cluster_std`,
- target MLP hidden layers, activation, epochs, batch size, learning rate,
- weight-image representation,
- whether to include the original training inputs in the image,
- per-image normalization.

The first image representation is `block_matrix`: each weight matrix or bias
vector is placed as a structured 2D block in deterministic layer order. Layout
metadata records the exact region for every block so the image remains
interpretable and reversible in principle.

