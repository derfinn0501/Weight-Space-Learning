# AGENTS.md

## Project Purpose

This repository studies weight-space learning by generating synthetic datasets, training target neural networks on them, converting trained weights into structured image representations, and learning representations of those weight-images with autoencoders.

The current project phase is:

1. Generate M synthetic datasets.
2. Train one configurable target model per dataset.
3. Convert trained model weights into deterministic weight images.
4. Train AE/CAE models on weight images.
5. Evaluate reconstruction quality and latent structure.
6. Later: add conditional dataset-to-weight-image prediction.

## Architecture Rules

- Keep the existing top-level structure stable:
  - config/
  - data/
  - scripts/
  - src/dataset_gen/
  - src/network_learning/
  - src/image_gen/
  - src/cond_AE/
  - src/evaluation/
  - src/utils/
  - tests/
- Do not move code into one large script.
- Do not duplicate logic across scripts.
- Use reusable functions inside `src/`.
- Scripts should only orchestrate pipeline steps.
- All important behavior must be controlled through YAML configs.
- Keep modules small, readable, and focused.
- Prefer explicit simple code over clever abstractions.
- Do not introduce unnecessary frameworks.

## Data And Git Rules

- The `data/` folder structure is tracked through `.gitkeep` files.
- Generated datasets, models, weight images, plots, metrics, logs, and checkpoints must stay ignored by Git.
- Do not commit generated experiment outputs unless explicitly requested.
- Raw trained model weights should be saved separately from generated weight images.
- Weight image layout metadata must be saved so representations remain interpretable.

## Coding Rules

- Use Python and PyTorch.
- Use type hints where helpful.
- Use short docstrings for important public functions/classes.
- Keep imports clean and stable.
- Scripts must run from the repository root.
- Avoid hidden global state.
- Use existing utility functions for config, paths, seeds, and IO.
- Keep naming consistent with existing modules.

## Testing Rules

Before finishing any change, run:

```bash
python -m pytest
```

For pipeline changes, also run the relevant scripts on the default config:

```bash
python scripts/01_generate_datasets.py --config config/default.yaml
python scripts/02_train_target_networks.py --config config/default.yaml
python scripts/03_generate_weight_images.py --config config/default.yaml
python scripts/04_train_autoencoder.py --config config/default.yaml
python scripts/05_evaluate.py --config config/default.yaml
```

If a command cannot be run, explain why.

