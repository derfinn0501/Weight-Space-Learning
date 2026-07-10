# Weight-Space-Learning

This repository studies weight-space learning by representing trained neural
network weights as image-like tensors.

The pipeline creates a meta-dataset where each meta-datapoint contains:

1. one sampled synthetic dataset,
2. one target neural network trained on that dataset,
3. the trained target-network weights,
4. one deterministic weight-image representation,
5. metadata plus train/test metrics,
6. AE/CAE reconstructions and latent embeddings of the weight images,
7. an optional dataset encoder trained into the same latent space.

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
src/cond_AE/            AE/CAE models, datasets, and training
src/dataset_encoder/    dataset-to-latent encoder models and training
src/evaluation/         metrics, plotting, and latent embedding helpers
src/utils/              config, IO, paths, and seed helpers
scripts/                runnable pipeline entry points
tests/                  minimal smoke tests
```

## Setup

Use a Python environment with PyTorch installed, then run:

```bash
pip install -r requirements.txt
```

## Run The Full Pipeline

Run all commands from the repository root:

```bash
python scripts/01_generate_datasets.py --config config/default.yaml
python scripts/02_train_target_networks.py --config config/default.yaml
python scripts/03_generate_weight_images.py --config config/default.yaml
python scripts/04_train_autoencoder.py --config config/default.yaml
python scripts/06_train_dataset_encoder.py --config config/default.yaml
python scripts/05_evaluate.py --config config/default.yaml
```

Or run the same stages as one tracked MLflow experiment:

```bash
python scripts/run_experiment.py --config config/default.yaml
```

With the default team configuration, runs are logged to the shared Raspberry Pi
MLflow server:

```text
http://192.168.1.26:5000
```

The Pi must be reachable from the current machine. When working outside the
home network, connect to the WireGuard VPN first and keep using the same
internal URL.

Use this command to run the pipeline without creating an MLflow run:

```bash
python scripts/run_experiment.py --config config/default.yaml --no-mlflow
```

## Shared MLflow Server

The team MLflow server runs on the Raspberry Pi `kopi` with Podman. The server
is reachable on the local network or VPN at:

```text
http://192.168.1.26:5000
```

The Pi deployment lives outside this repository:

```text
~/mlflow-server/
  compose.yml
  Dockerfile
  .env
  postgres-data/       PostgreSQL metadata store
  mlflow-artifacts/    run artifacts served through MLflow
```

The server has two persistent stores:

- PostgreSQL stores experiment metadata, parameters, metrics, tags, and run
  state.
- `mlflow-artifacts/` stores configs, plots, checkpoints, and other files
  logged by `run_experiment.py`.

Do not delete `postgres-data/` or `mlflow-artifacts/` unless you explicitly want
to remove the experiment history.

### Using The Shared Server

1. Connect to the same network as the Pi, or connect through WireGuard VPN.
2. Open `http://192.168.1.26:5000` and confirm the MLflow UI loads.
3. Run an experiment from the repository root:

   ```bash
   python scripts/run_experiment.py --config config/default.yaml
   ```

4. Refresh the MLflow UI. The run should appear under the experiment configured
   by `mlflow.experiment_name`.

MLflow tracking is configured in `config/default.yaml` under the `mlflow`
section. The default config points to the shared Pi server so normal
`run_experiment.py` calls are logged there automatically.

### Restarting The Server

On the Raspberry Pi:

```bash
cd ~/mlflow-server

podman stop mlflow-server
podman restart mlflow-postgres
sleep 5
podman start mlflow-server
sleep 60
curl -I http://127.0.0.1:5000
```

The final `curl` should return an HTTP response such as `HTTP/1.1 200 OK`.
MLflow can take 30-60 seconds to finish booting on the Pi.

After editing `~/mlflow-server/compose.yml`, recreate the MLflow container so
the new command is applied:

```bash
cd ~/mlflow-server

podman stop mlflow-server
podman rm mlflow-server
podman compose -f compose.yml up -d mlflow
sleep 60
curl -I http://127.0.0.1:5000
```

Avoid `podman compose down -v`; volume removal can destroy persisted server
state in volume-based setups.

### Server Health Checks

On the Pi:

```bash
podman ps -a --filter name=mlflow
podman exec mlflow-postgres pg_isready -U mlflow -d mlflow
curl -I http://127.0.0.1:5000
```

From another machine on the network or VPN:

```bash
curl -I http://192.168.1.26:5000
```

If the browser UI works but some requests are blocked, check the MLflow logs:

```bash
podman logs --tail 120 mlflow-server
```

The server command in `compose.yml` should allow the host and origin used by the
team, for example `192.168.1.26:5000`.

### Local Fallback Server

For single-user local development without the Pi, start the SQLite-backed local
server from the repository root:

```bash
python scripts/start_mlflow_server.py --config config/default.yaml
```

The local server uses:

```text
data/mlflow/mlflow.db         SQLite backend store for local run metadata
data/mlflow/artifacts/        local artifact store
```

For local-only tracking, set `mlflow.tracking_uri` in the config to:

```text
http://127.0.0.1:5000
```

Generated artifacts are written under:

```text
data/processed/             generated datasets and dataset metadata
data/model_zoo/             trained model checkpoints, metrics, and metadata
data/weight_images/         raw extracted weights, image tensors, and layout metadata
data/results/autoencoders/  AE/CAE checkpoint, training history, metrics, metadata
data/results/dataset_encoders/  dataset encoder checkpoint, history, metrics, metadata
data/results/figures/       loss curves and reconstruction plots
data/results/metrics/       reconstruction metrics and latent embeddings
data/mlflow/                local MLflow SQLite DB and artifacts
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
- per-image normalization,
- AE/CAE model type, latent dimension, training hyperparameters, and split,
- dataset-encoder model type, tabular input split, architecture, and training hyperparameters,
- evaluation plot and latent-embedding outputs,
- MLflow tracking behavior, including whether tracking is enabled, the shared
  server URI, experiment name, run name, artifact logging, and run tags.

The first image representation is `block_matrix`: each weight matrix or bias
vector is placed as a structured 2D block in deterministic layer order. Layout
metadata records the exact region for every block so the image remains
interpretable and reversible in principle.

## Autoencoder Stage

`scripts/04_train_autoencoder.py` trains either:

- `autoencoder.model_type: "ae"` for a fully connected autoencoder over flattened weight images, or
- `autoencoder.model_type: "cae"` for a convolutional autoencoder over `[1, H, W]` image tensors.

The AE/CAE learns to reconstruct generated weight images. This is not yet full
conditional weight generation. Later milestones will add dataset-conditioned
prediction of weight images.

`scripts/05_evaluate.py` loads the trained checkpoint, computes reconstruction
MSE/MAE/relative-L2 metrics, saves example reconstruction plots, and optionally
saves latent embeddings.

## Dataset Encoder Stage

`scripts/06_train_dataset_encoder.py` trains `dataset_encoder.model_type:
"deepsets"` to map generated tabular datasets into the frozen autoencoder latent
space. The default `dataset_encoder.input_split: "all"` uses both train and test
rows, with labels included as row features.
