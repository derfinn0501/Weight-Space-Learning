# Architecture

## Project Goal

Weight-Space-Learning builds meta-datasets for studying trained neural-network weights. Each meta-datapoint starts from one synthetic dataset, trains one target model, converts the trained parameters into a structured image representation, and then learns representations of those weight images with autoencoders.

The current AE/CAE stage reconstructs existing weight images. It is not yet a conditional model that predicts target-network weights from a dataset.

## Current Pipeline

Run scripts from the repository root in this order:

```bash
python scripts/01_generate_datasets.py --config config/default.yaml
python scripts/02_train_target_networks.py --config config/default.yaml
python scripts/03_generate_weight_images.py --config config/default.yaml
python scripts/04_train_autoencoder.py --config config/default.yaml
python scripts/05_evaluate.py --config config/default.yaml
```

The pipeline is:

1. Generate synthetic classification datasets.
2. Train one configurable target MLP per dataset.
3. Extract trained weights and biases in deterministic order.
4. Convert parameters into structured weight-image tensors.
5. Train an AE or CAE to reconstruct those images.
6. Evaluate reconstruction quality, save plots, and optionally save latent embeddings.

## Top-Level Folders

- `config/`: YAML files controlling data generation, target-network training, image generation, AE/CAE training, evaluation, and paths.
- `data/`: generated artifacts. Only `.gitkeep` files are tracked.
- `docs/`: project notes and architecture documentation.
- `scripts/`: executable pipeline entry points. Scripts orchestrate modules but should not contain core logic.
- `src/`: reusable Python modules.
- `tests/`: smoke tests for data generation, models, image generation, datasets, and metrics.

## Major Modules

- `src/dataset_gen/`: synthetic dataset generators, registry, and collection saving.
- `src/network_learning/`: configurable target MLP, model registry, single-model training, and collection training.
- `src/image_gen/`: state-dict weight extraction, deterministic image layouts, normalization, and weight-image loading.
- `src/cond_AE/`: AE/CAE dataset helpers, models, and training logic.
- `src/evaluation/`: classification and reconstruction metrics, reconstruction plots, loss curves, and latent embedding export.
- `src/utils/`: config loading, seed setting, IO, and output-directory creation.

## Generated Artifacts

- `data/processed/`: generated dataset `.pt` files and dataset metadata.
- `data/model_zoo/`: trained target-network checkpoints, metrics, and metadata.
- `data/weight_images/`: generated weight-image tensors, raw extracted parameters, and layout metadata.
- `data/results/autoencoders/`: AE/CAE checkpoints, training history, final metrics, and metadata.
- `data/results/figures/`: dataset, weight-image, reconstruction, error, and loss plots.
- `data/results/metrics/`: evaluation metrics and latent embeddings.

All generated artifacts are ignored by Git unless explicitly requested otherwise.

## Reference Code

The external `Rügamer code` folder was used only as conceptual reference for the original idea of image-like neural-network weight representations. Its structure should not be copied blindly into this repository.

