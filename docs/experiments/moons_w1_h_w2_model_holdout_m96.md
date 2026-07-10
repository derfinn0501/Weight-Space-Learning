# Moons `w1_h_w2` Model-Holdout M96

## Purpose

This experiment tests whether the autoencoder learns a general structure of
the weight-image space or mainly memorizes target-model-specific `W1` and `W2`
patterns.

The key change compared with the first pilot is the split:

```text
All images from a trained target model stay together.
Validation uses target models that are never seen during AE training.
```

## Setup

- Dataset generator: `moons`
- Target models: `M = 96`
- Train points per target model: `N_train = 64`
- Test points per target model: `N_test = 256`
- Target network: one-hidden-layer MLP with hidden layer `[32]`
- Target activation: ReLU
- Target training epochs: `50`
- Weight-image representation: `w1_h_w2`
- Weight-image count: `96 * 64 = 6144`
- Weight-image shape: `[1, 32, 7]`
- Autoencoder: fully connected AE
- Latent dimension: `16`
- AE epochs: `60`
- AE split: model-held-out split

Config:

```text
config/experiments/moons_w1_h_w2_model_holdout_m96.yaml
```

Local outputs:

```text
data/experiments/moons_w1_h_w2_model_holdout_m96/
```

MLflow run:

```text
http://192.168.178.25:5000/#/experiments/3/runs/3a9bb05bf7504502a1178761e47b443e
```

## Split Details

The grouped split worked as intended:

```text
train images:       4928
validation images:  1216
train models:       77
validation models:  19
overlap models:     0
```

## Results

Target model performance:

```text
mean test accuracy: 0.8661
min test accuracy:  0.8164
max test accuracy:  0.9258
```

Autoencoder performance:

```text
train MSE:        0.0703
train relL2:      0.2588

validation MSE:   1.5046
validation relL2: 1.2214

overall reconstruction MSE:   0.3542
overall reconstruction relL2: 0.4493
```

Runtime:

```text
about 200 seconds wall time
```

Stage times from MLflow:

```text
datasets:        0.13s
target networks: 7.66s
weight images:   3.31s
autoencoder:   129.34s
evaluation:      7.20s
```

## Interpretation

This is the most important result so far. The train reconstruction improves,
but validation on unseen target models is much worse. This means the random
image split from the first pilot was optimistic.

The current AE can reconstruct images from target-model families it has seen,
but it does not yet generalize well to completely new trained target networks.
This suggests that the representation and/or model currently captures
model-specific structure more strongly than general weight-space structure.

Scientifically, this is useful: the grouped split gives a more honest
evaluation of weight-space generalization. Future experiments should use this
split when the question is generalization to unseen trained models.

## Next Questions

Useful follow-up experiments:

- Increase AE capacity or latent dimension and keep the model-held-out split.
- Compare `latent_dim = 16` against `32` or `64`.
- Try a CAE on the same held-out split, although the current images are narrow.
- Add normalization or gauge-fixing ideas to reduce target-model-specific
  variation.
- Compare against a simpler baseline, such as reconstructing the mean image or
  PCA on flattened images.

