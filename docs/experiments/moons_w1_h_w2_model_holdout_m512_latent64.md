# Moons `w1_h_w2` Model-Holdout M512 Latent64

## Purpose

This experiment scales the model-held-out setup from `M = 96` to `M = 512`.
The goal is to test whether more trained target models give the autoencoder
enough variation to learn a more general weight-image structure.

The scientific question is:

```text
Can the AE reconstruct weight images from completely unseen trained target
networks when it has seen many more target models during training?
```

This run keeps the same basic target-network architecture and `w1_h_w2`
representation, but increases the number of target models and uses a larger
latent dimension.

## Setup

- Dataset generator: `moons`
- Target models: `M = 512`
- Train points per target model: `N_train = 64`
- Test points per target model: `N_test = 256`
- Target network: one-hidden-layer MLP with hidden layer `[32]`
- Target activation: ReLU
- Target training epochs: `50`
- Weight-image representation: `w1_h_w2`
- Weight-image count: `512 * 64 = 32768`
- Weight-image shape: `[1, 32, 7]`
- Autoencoder: fully connected AE
- Latent dimension: `64`
- AE epochs: `150`
- AE split: model-held-out split

Config:

```text
config/experiments/moons_w1_h_w2_model_holdout_m512_latent64.yaml
```

Local outputs:

```text
data/experiments/moons_w1_h_w2_model_holdout_m512_latent64/
```

MLflow run:

```text
http://192.168.178.25:5000/#/experiments/3/runs/e74681f7baba456798bfec0fc9c509c7
```

## Split Details

The grouped split worked as intended:

```text
train images:       26240
validation images:   6528
train models:         410
validation models:    102
overlap models:         0
```

## Results

Target model performance:

```text
mean test accuracy: 0.8701
min test accuracy:  0.8086
max test accuracy:  0.9180

mean train accuracy: 0.8800
mean test loss:      0.2950
```

Autoencoder final performance:

```text
train MSE:        0.1120
train relL2:      0.3325

validation MSE:   0.4707
validation relL2: 0.6750

overall reconstruction MSE:   0.1835
overall reconstruction relL2: 0.4007
```

Best validation point during AE training:

```text
best epoch:       6
best val MSE:     0.3103
train MSE then:   0.2074

final epoch:      150
final val MSE:    0.4707
final train MSE:  0.1128
```

Runtime:

```text
about 39 minutes wall time
```

Stage times from MLflow:

```text
datasets:           0.47s
target networks:   40.46s
weight images:     17.91s
autoencoder:     1992.93s
evaluation:        38.19s
```

## Interpretation

This is the strongest baseline so far because it uses many more target models
and keeps validation models completely held out. The target networks learned
the moons task well enough to produce meaningful trained weights, with mean
test accuracy around `87%`.

The larger dataset clearly improves the stability of the held-out validation
curve compared with the smaller `M = 96` run. The validation MSE is much lower
than in the small model-holdout experiment, so increasing the number of target
models helps.

However, the AE still overfits strongly. Training loss keeps decreasing until
the end, but validation is best at epoch `6` and gets worse afterwards. This
means the final checkpoint is not the best scientific model for
generalization. The important result is therefore not only the final validation
score, but the shape of the curve:

```text
more target models help, but longer AE training without early stopping hurts
model-held-out generalization.
```

The model-held-out split remains the right evaluation setup. A random image
split would again be too optimistic, because multiple images from the same
target model share the same `W1` and `W2`.

## Comparison With Earlier Runs

Compared with the initial random-split pilot, this run is harder and more
realistic. Validation images come from target models that the AE never sees
during training.

Compared with the `M = 96` model-held-out run, this run shows that scaling to
`M = 512` helps substantially. The held-out validation error is much lower and
the result is less noisy. But the overfitting pattern is now very clear, so the
next bottleneck is the AE training policy rather than the raw number of images.

## Next Questions

Useful follow-up experiments:

- Add early stopping and save the best validation checkpoint.
- Re-run this same `M = 512` setup with early stopping before increasing `M`.
- Compare latent dimensions, for example `16`, `32`, and `64`, under the same
  model-held-out split.
- Add AE regularization, such as weight decay or dropout.
- Test whether a smaller AE generalizes better than the current higher-capacity
  model.
- Try a CAE on the same split, while keeping in mind that the images are narrow
  and may not benefit much from convolution yet.

