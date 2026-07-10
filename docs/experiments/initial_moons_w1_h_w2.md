# Initial Moons `w1_h_w2` Pilot

## Purpose

This was the first real experiment with the professor-style image idea:

```text
W1 | h(x) | W2^T
```

The goal was to check whether this representation contains enough regular
structure for an autoencoder to learn meaningful reconstructions.

## Setup

- Dataset generator: `moons`
- Target models: `M = 32`
- Train points per target model: `N_train = 64`
- Test points per target model: `N_test = 256`
- Target network: one-hidden-layer MLP with hidden layer `[32]`
- Target activation: ReLU
- Target training epochs: `50`
- Weight-image representation: `w1_h_w2`
- Weight-image count: `32 * 64 = 2048`
- Weight-image shape: `[1, 32, 7]`
- Autoencoder: fully connected AE
- Latent dimension: `16`
- AE epochs: `40`
- AE split: random image split

Config:

```text
config/experiments/initial_moons_w1_h_w2.yaml
```

Local outputs:

```text
data/experiments/initial_moons_w1_h_w2/
```

Primary MLflow run:

```text
http://192.168.178.25:5000/#/experiments/3/runs/4df9c49c19964a07be651dedf697a594
```

There is also an earlier duplicate run with the same scientific setup. That
first run uploaded many per-image layout files to MLflow and was slower, but
the clean run above is the one to use for comparison.

## Results

Target model performance:

```text
mean test accuracy: 0.8704
min test accuracy:  0.8281
max test accuracy:  0.8984
```

Autoencoder performance:

```text
train MSE:        0.0484
validation MSE:   0.0563
validation relL2: 0.2287

overall reconstruction MSE:   0.0496
overall reconstruction relL2: 0.2143
```

Runtime after reducing MLflow artifact upload:

```text
about 65 seconds wall time
```

## Interpretation

This run confirms that the `W1 | h(x) | W2^T` representation is learnable in a
basic autoencoder setting. The AE reconstructs the images much better than a
trivial failure case, and the training curve decreases smoothly.

However, the split is a random image split. Since each target model contributes
64 images with the same `W1` and `W2`, train and validation likely contain
images from the same trained target networks. The validation score therefore
answers a limited question:

```text
Can the AE reconstruct new h(x) variations for mostly known W1/W2 families?
```

It does not yet answer:

```text
Can the AE generalize to completely unseen trained target networks?
```

This is why the next experiment uses a model-held-out split.

