"""Opt-in local-only MLflow; no dependency import when tracking is disabled."""
from pathlib import Path


def log_local_run(result: dict, workspace: Path, *, experiment_name="trafficpulse-synthetic-phase2a",
                  run_name="synthetic-two-baselines", artifacts=()):
    import os
    os.environ["MLFLOW_ENABLE_TELEMETRY"] = "false"
    os.environ["DO_NOT_TRACK"] = "true"
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("MLflow is optional and missing; ask before installing it") from exc
    location = (workspace / "artifacts/synthetic_mlflow").resolve()
    location.mkdir(parents=True, exist_ok=True)
    # Override inherited tracking/registry endpoints. No user-supplied remote URI.
    uri = "sqlite:///" + (location / "tracking.db").as_posix()
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    name = experiment_name
    experiment = mlflow.get_experiment_by_name(name)
    experiment_id = experiment.experiment_id if experiment else mlflow.create_experiment(
        name, artifact_location=(location / "artifacts").as_uri())
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        mlflow.set_tags({"data_origin": result["data_origin"], "real_dataset_access": "none"})
        mlflow.log_params(result["parameters"])
        for model, horizons in result["metrics"].items():
            for h, metrics in horizons.items():
                for metric in ("mae", "rmse", "valid_target_count", "coverage", "average_interval_width",
                               "calibrated_half_width", "calibration_count", "fallback_frequency",
                               "primary_frequency", "abstention_frequency", "switches_per_1000_origins"):
                    if metrics.get(metric) is not None:
                        mlflow.log_metric(f"{model}_h{h}_{metric}", metrics[metric])
        for name,value in result.get("resources",{}).items():
            if isinstance(value,(float,int)) and not isinstance(value,bool):
                mlflow.log_metric(name,value)
        for artifact in artifacts:
            path=Path(artifact).resolve()
            if not path.is_relative_to((workspace/"artifacts").resolve()):
                raise ValueError("Only explicitly local synthetic artifacts may be logged")
            mlflow.log_artifact(str(path),artifact_path="synthetic_outputs")
        return {"status": "logged_local", "run_id": run.info.run_id,
                "tracking_uri": uri}
