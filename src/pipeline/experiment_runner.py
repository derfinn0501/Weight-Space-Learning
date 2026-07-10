"""Full experiment orchestration with optional MLflow tracking."""

from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from src.cond_AE.train_ae import train_autoencoder
from src.dataset_gen.dataset_collection import generate_dataset_collection
from src.dataset_encoder.train_encoder import train_dataset_encoder
from src.evaluation.pipeline_evaluation import evaluate_trained_autoencoder
from src.evaluation.plots import plot_loss_curve
from src.image_gen.generate_collection import generate_weight_images
from src.network_learning.train_collection import train_model_collection
from src.utils.config import load_config
from src.utils.paths import create_output_dirs
from src.utils.seed import set_random_seed

STAGE_ORDER = (
    "datasets",
    "target_networks",
    "weight_images",
    "autoencoder",
    "dataset_encoder",
    "evaluation",
)


def _import_mlflow() -> Any:
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(
            "MLflow tracking is enabled, but the 'mlflow' package is not installed. "
            "Install it with `pip install mlflow` or run with `--no-mlflow`."
        ) from exc
    return mlflow


def _flatten_for_params(payload: Any, prefix: str = "") -> dict[str, str]:
    if isinstance(payload, dict):
        flattened: dict[str, str] = {}
        for key, value in payload.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_for_params(value, name))
        return flattened

    if isinstance(payload, (list, tuple)):
        value = json.dumps(payload)
    elif isinstance(payload, (str, int, float, bool)) or payload is None:
        value = str(payload)
    else:
        value = str(payload)

    return {prefix: value[:500]}


def _log_metric(mlflow: Any | None, name: str, value: Any, step: int | None = None) -> None:
    if mlflow is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        return
    value_float = float(value)
    if math.isfinite(value_float):
        mlflow.log_metric(name, value_float, step=step)


def _log_metrics_dict(
    mlflow: Any | None,
    prefix: str,
    metrics: dict[str, Any],
    step: int | None = None,
) -> None:
    for key, value in metrics.items():
        if isinstance(value, dict):
            _log_metrics_dict(mlflow, f"{prefix}.{key}", value, step=step)
        else:
            _log_metric(mlflow, f"{prefix}.{key}", value, step=step)


def _log_record_summary(mlflow: Any | None, prefix: str, records: Sequence[dict[str, Any]]) -> None:
    _log_metric(mlflow, f"{prefix}.count", len(records))
    if not records:
        return

    keys = sorted({key for record in records for key in record})
    for key in keys:
        values = [
            record[key]
            for record in records
            if isinstance(record.get(key), (int, float, np.integer, np.floating))
            and not isinstance(record.get(key), bool)
        ]
        if not values:
            continue
        values_np = np.asarray(values, dtype=np.float64)
        if values_np.size == 1:
            _log_metric(mlflow, f"{prefix}.{key}", values_np.item())
        else:
            _log_metric(mlflow, f"{prefix}.{key}.mean", float(values_np.mean()))
            _log_metric(mlflow, f"{prefix}.{key}.min", float(values_np.min()))
            _log_metric(mlflow, f"{prefix}.{key}.max", float(values_np.max()))


def _log_artifact_paths(
    mlflow: Any | None,
    paths: Iterable[str | Path],
    artifact_path: str,
) -> None:
    if mlflow is None:
        return
    for path_like in paths:
        path = Path(path_like)
        if not path.exists():
            continue
        if path.is_dir():
            mlflow.log_artifacts(str(path), artifact_path=artifact_path)
        else:
            mlflow.log_artifact(str(path), artifact_path=artifact_path)


def _selected_artifacts(config: dict[str, Any], stage: str) -> list[Path]:
    paths = config.get("paths", {})
    processed_dir = Path(paths.get("processed_dir", "data/processed"))
    model_zoo_dir = Path(paths.get("model_zoo_dir", "data/model_zoo"))
    weight_image_dir = Path(paths.get("weight_image_dir", "data/weight_images"))
    autoencoder_dir = Path(paths.get("autoencoder_dir", "data/results/autoencoders"))
    dataset_encoder_dir = Path(paths.get("dataset_encoder_dir", "data/results/dataset_encoders"))
    figure_dir = Path(paths.get("figure_dir", "data/results/figures"))
    metric_dir = Path(paths.get("metric_dir", "data/results/metrics"))

    if stage == "datasets":
        return [
            processed_dir / "dataset_metadata.json",
            processed_dir / "dataset_metadata.csv",
        ]
    if stage == "target_networks":
        return [
            model_zoo_dir / "collection_metadata.json",
            model_zoo_dir / "collection_metadata.csv",
            model_zoo_dir / "metrics",
        ]
    if stage == "weight_images":
        return [
            weight_image_dir / "metadata.json",
            weight_image_dir / "metadata.csv",
            weight_image_dir / "layouts",
        ]
    if stage == "autoencoder":
        return [
            autoencoder_dir / "checkpoint.pt",
            autoencoder_dir / "training_history.json",
            autoencoder_dir / "metrics.json",
            autoencoder_dir / "metadata.json",
            figure_dir / f"autoencoder_loss_curve.{config.get('evaluation', {}).get('plot_format', 'png')}",
        ]
    if stage == "dataset_encoder":
        return [
            dataset_encoder_dir / "checkpoint.pt",
            dataset_encoder_dir / "training_history.json",
            dataset_encoder_dir / "metrics.json",
            dataset_encoder_dir / "metadata.json",
            figure_dir / f"dataset_encoder_loss_curve.{config.get('evaluation', {}).get('plot_format', 'png')}",
            figure_dir / f"dataset_encoder_reconstructions.{config.get('evaluation', {}).get('plot_format', 'png')}",
        ]
    if stage == "evaluation":
        return [
            metric_dir / "reconstruction_metrics.json",
            metric_dir / "latent_embeddings.csv",
            metric_dir / "latent_embeddings.pt",
            figure_dir / f"reconstructions.{config.get('evaluation', {}).get('plot_format', 'png')}",
            figure_dir / f"reconstruction_error_heatmap.{config.get('evaluation', {}).get('plot_format', 'png')}",
        ]
    return []


def _run_stage(
    name: str,
    fn: Callable[[], Any],
    *,
    mlflow: Any | None,
    log_artifacts: bool,
    config: dict[str, Any],
) -> Any:
    start = time.perf_counter()
    try:
        result = fn()
    except Exception:
        _log_metric(mlflow, f"stage.{name}.failed", 1.0)
        raise

    duration = time.perf_counter() - start
    _log_metric(mlflow, f"stage.{name}.duration_seconds", duration)
    _log_metric(mlflow, f"stage.{name}.completed", 1.0)

    if log_artifacts:
        _log_artifact_paths(mlflow, _selected_artifacts(config, name), artifact_path=f"stages/{name}")

    return result


def _train_autoencoder_stage(config: dict[str, Any]) -> dict[str, Any]:
    result = train_autoencoder(config)
    plot_format = config.get("evaluation", {}).get("plot_format", "png")
    figure_dir = Path(config.get("paths", {}).get("figure_dir", "data/results/figures"))
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_loss_curve(
        result["history"],
        figure_dir / f"autoencoder_loss_curve.{plot_format}",
    )
    return result


def _train_dataset_encoder_stage(config: dict[str, Any]) -> dict[str, Any]:
    result = train_dataset_encoder(config)
    plot_format = config.get("evaluation", {}).get("plot_format", "png")
    figure_dir = Path(config.get("paths", {}).get("figure_dir", "data/results/figures"))
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_loss_curve(
        result["history"],
        figure_dir / f"dataset_encoder_loss_curve.{plot_format}",
        title="Dataset encoder training loss",
    )
    return result


def _normalize_stages(stages: Sequence[str] | None) -> list[str]:
    if stages is None:
        return list(STAGE_ORDER)

    unknown = sorted(set(stages) - set(STAGE_ORDER))
    if unknown:
        raise ValueError(f"Unknown stage(s): {unknown}. Valid stages: {list(STAGE_ORDER)}")
    return [stage for stage in STAGE_ORDER if stage in stages]


def _setup_mlflow(config: dict[str, Any], config_path: Path, disable_mlflow: bool) -> Any | None:
    mlflow_cfg = config.get("mlflow", {})
    if disable_mlflow or not bool(mlflow_cfg.get("enabled", False)):
        return None

    mlflow = _import_mlflow()
    tracking_uri = mlflow_cfg.get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(str(tracking_uri))

    experiment_name = str(mlflow_cfg.get("experiment_name", "weight-space-learning"))
    mlflow.set_experiment(experiment_name)

    return mlflow


def _mlflow_tags(config: dict[str, Any], config_path: Path) -> dict[str, str]:
    mlflow_cfg = config.get("mlflow", {})
    tags = {
        "config_path": str(config_path),
        "pipeline": "weight-space-learning",
    }
    tags.update({str(key): str(value) for key, value in mlflow_cfg.get("tags", {}).items()})
    return tags


def run_experiment(
    config_path: str | Path = "config/default.yaml",
    stages: Sequence[str] | None = None,
    *,
    disable_mlflow: bool = False,
) -> dict[str, Any]:
    """Run the configured pipeline and optionally track it as one MLflow run."""
    resolved_config_path = Path(config_path)
    config = load_config(resolved_config_path)
    create_output_dirs(config)
    selected_stages = _normalize_stages(stages)
    mlflow = _setup_mlflow(config, resolved_config_path, disable_mlflow)
    mlflow_cfg = config.get("mlflow", {})
    log_artifacts = mlflow is not None and bool(mlflow_cfg.get("log_artifacts", True))
    run_name = str(mlflow_cfg.get("run_name", resolved_config_path.stem))

    results: dict[str, Any] = {
        "config_path": str(resolved_config_path),
        "stages": selected_stages,
    }

    run_context = mlflow.start_run(run_name=run_name) if mlflow is not None else nullcontext()

    with run_context:
        if mlflow is not None:
            for key, value in _mlflow_tags(config, resolved_config_path).items():
                mlflow.set_tag(key, value)
            mlflow.log_params(_flatten_for_params(config))
            if log_artifacts:
                mlflow.log_artifact(str(resolved_config_path), artifact_path="config")

        for stage in selected_stages:
            if stage == "datasets":
                set_random_seed(int(config.get("seed", 42)))
                records = _run_stage(
                    stage,
                    lambda: generate_dataset_collection(config),
                    mlflow=mlflow,
                    log_artifacts=log_artifacts,
                    config=config,
                )
                _log_record_summary(mlflow, "datasets", records)
                results[stage] = {"count": len(records), "records": records}

            elif stage == "target_networks":
                set_random_seed(int(config.get("seed", 42)))
                records = _run_stage(
                    stage,
                    lambda: train_model_collection(config),
                    mlflow=mlflow,
                    log_artifacts=log_artifacts,
                    config=config,
                )
                _log_record_summary(mlflow, "target_networks", records)
                results[stage] = {"count": len(records), "records": records}

            elif stage == "weight_images":
                records = _run_stage(
                    stage,
                    lambda: generate_weight_images(config),
                    mlflow=mlflow,
                    log_artifacts=log_artifacts,
                    config=config,
                )
                _log_record_summary(mlflow, "weight_images", records)
                results[stage] = {"count": len(records), "records": records}

            elif stage == "autoencoder":
                set_random_seed(int(config.get("seed", 42)))
                ae_result = _run_stage(
                    stage,
                    lambda: _train_autoencoder_stage(config),
                    mlflow=mlflow,
                    log_artifacts=log_artifacts,
                    config=config,
                )
                _log_metrics_dict(mlflow, "autoencoder.final", ae_result["metrics"])
                for epoch, train_loss, val_loss in zip(
                    ae_result["history"]["epoch"],
                    ae_result["history"]["train_loss"],
                    ae_result["history"]["val_loss"],
                ):
                    step = int(epoch)
                    _log_metric(mlflow, "autoencoder.train_loss", train_loss, step=step)
                    _log_metric(mlflow, "autoencoder.val_loss", val_loss, step=step)
                results[stage] = {
                    "metrics": ae_result["metrics"],
                    "metadata": ae_result["metadata"],
                    "output_dir": ae_result["output_dir"],
                }

            elif stage == "dataset_encoder":
                set_random_seed(int(config.get("seed", 42)))
                encoder_result = _run_stage(
                    stage,
                    lambda: _train_dataset_encoder_stage(config),
                    mlflow=mlflow,
                    log_artifacts=log_artifacts,
                    config=config,
                )
                metadata = encoder_result["metadata"]
                for key in ("n_total", "n_train", "n_validation", "input_dim", "row_dim", "latent_dim"):
                    _log_metric(mlflow, f"dataset_encoder.{key}", metadata.get(key))
                _log_metrics_dict(mlflow, "dataset_encoder.final", encoder_result["metrics"])
                for epoch, train_loss, val_loss in zip(
                    encoder_result["history"]["epoch"],
                    encoder_result["history"]["train_loss"],
                    encoder_result["history"]["val_loss"],
                ):
                    step = int(epoch)
                    _log_metric(mlflow, "dataset_encoder.train_loss", train_loss, step=step)
                    _log_metric(mlflow, "dataset_encoder.val_loss", val_loss, step=step)
                results[stage] = {
                    "metrics": encoder_result["metrics"],
                    "metadata": metadata,
                    "output_dir": encoder_result["output_dir"],
                }

            elif stage == "evaluation":
                eval_result = _run_stage(
                    stage,
                    lambda: evaluate_trained_autoencoder(config),
                    mlflow=mlflow,
                    log_artifacts=log_artifacts,
                    config=config,
                )
                _log_metrics_dict(mlflow, "evaluation.reconstruction", eval_result["metrics"])
                results[stage] = eval_result

    return results
