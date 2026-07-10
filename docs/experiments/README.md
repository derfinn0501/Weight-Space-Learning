# Experiment Notes

This folder contains short human-readable notes for the MLflow runs and local
experiment outputs produced so far. The goal is to make the experiment history
understandable without opening every config, metric JSON, or artifact folder.

The current experiment sequence studies the `w1_h_w2` image representation:

```text
W1 | h(x) | W2^T
```

For one trained one-hidden-layer target MLP and one input point `x`, the image
contains the first-layer weights `W1`, the hidden activation vector `h(x)`, and
the transposed output weights `W2^T`. Therefore each trained target model
produces `N_train` weight images.

## Runs

| Run | Main question | Split | Summary |
| --- | --- | --- | --- |
| [Initial moons w1_h_w2 pilot](initial_moons_w1_h_w2.md) | Is the new image representation learnable at all? | Random image split | Yes. Reconstruction works when train/validation can share target models. |
| [Moons w1_h_w2 model holdout M96](moons_w1_h_w2_model_holdout_m96.md) | Does the AE generalize to unseen trained target models? | Model-held-out split | Not yet. Strong generalization gap appears. |
| [Moons w1_h_w2 model holdout M512 latent64](moons_w1_h_w2_model_holdout_m512_latent64.md) | Does scaling to many more target models improve held-out reconstruction? | Model-held-out split | Yes, but the AE overfits quickly. Best validation is at epoch 6, so early stopping is needed. |

## Current Interpretation

The first pilot showed that the representation has learnable structure. The
second run showed that a random image split is too optimistic, because many
images share the same `W1` and `W2`. When validation models are held out
entirely, reconstruction becomes much harder.

The larger `M = 512` run shows that scaling the number of target models helps:
held-out validation is much better than in the small `M = 96` run. At the same
time, the AE clearly overfits. The best validation point is very early, while
the final checkpoint is worse on unseen target models.

This suggests that the next useful work should keep the model-held-out split
and focus on early stopping, best-checkpoint saving, and AE regularization
before scaling to even larger experiments.
