#!/usr/bin/env python3
"""Evaluate a model checkpoint produced by training/train.py.

This is intentionally a separate, single-process script: evaluation does not need to be
distributed, and keeping it separate mirrors the ``evaluate-model`` step of the AI
Pipeline (``pipeline/pipeline.py``), which runs after the TrainJob completes and reads the
checkpoint it produced.

Usage::

    python training/evaluate.py --model-dir /opt/app-root/src/output
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Import the exact model/dataset definitions used at training time, so evaluation never
# drifts from what was actually trained.
from train import SyntheticRegressionDataset, TinyRegressor  # noqa: E402


def load_model(model_dir: Path, n_features: int, hidden_size: int) -> TinyRegressor:
    model = TinyRegressor(n_features=n_features, hidden_size=hidden_size)
    state_dict = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate(args: argparse.Namespace) -> dict:
    model_dir = Path(args.model_dir)
    metrics_path = model_dir / "training_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"{metrics_path} not found -- did the training step run and write its output here?"
        )

    training_metrics = json.loads(metrics_path.read_text())
    config = training_metrics.get("config", {})
    n_features = config.get("n_features", args.n_features)
    hidden_size = config.get("hidden_size", args.hidden_size)

    model = load_model(model_dir, n_features=n_features, hidden_size=hidden_size)

    test_ds = SyntheticRegressionDataset(
        n_samples=args.test_samples, n_features=n_features, seed=args.seed + 999
    )
    loss_fn = nn.MSELoss()

    with torch.no_grad():
        x = test_ds.x
        y = test_ds.y
        pred = model(x)
        test_mse = loss_fn(pred, y).item()
        mae = torch.mean(torch.abs(pred - y)).item()

    passed = test_mse <= args.max_acceptable_mse

    result = {
        "test_mse": test_mse,
        "test_mae": mae,
        "max_acceptable_mse": args.max_acceptable_mse,
        "passed": passed,
        "training_final_train_loss": training_metrics.get("final_train_loss"),
        "training_final_val_loss": training_metrics.get("final_val_loss"),
        "training_world_size": training_metrics.get("world_size"),
    }

    status = "PASS" if passed else "FAIL"
    print(f"[evaluate] test_mse={test_mse:.6f} test_mae={mae:.6f} -> {status}")

    output_path = model_dir / "evaluation_result.json"
    output_path.write_text(json.dumps(result, indent=2))
    print(f"[evaluate] evaluation result written to {output_path}")

    maybe_log_mlflow(result, args)

    return result


def maybe_log_mlflow(result: dict, args: argparse.Namespace) -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        print("[evaluate] MLFLOW_TRACKING_URI not set -- skipping MLflow logging.")
        return
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(args.experiment_name)
        with mlflow.start_run(run_name="evaluate"):
            mlflow.log_metrics({"test_mse": result["test_mse"], "test_mae": result["test_mae"]})
            mlflow.log_param("passed", result["passed"])
        print(f"[evaluate] evaluation metrics logged to MLflow at {tracking_uri}")
    except Exception as exc:  # noqa: BLE001 - MLflow is always optional
        print(f"[evaluate] MLflow logging skipped (not fatal): {exc}")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=str, default=os.environ.get("TRAIN_OUTPUT_DIR", "/opt/app-root/src/output"))
    parser.add_argument("--test-samples", type=int, default=int(os.environ.get("TEST_SAMPLES", 256)))
    parser.add_argument("--n-features", type=int, default=int(os.environ.get("TRAIN_N_FEATURES", 8)))
    parser.add_argument("--hidden-size", type=int, default=int(os.environ.get("TRAIN_HIDDEN_SIZE", 32)))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("TRAIN_SEED", 42)))
    parser.add_argument("--max-acceptable-mse", type=float, default=float(os.environ.get("MAX_ACCEPTABLE_MSE", 0.5)))
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "rhoai-pytorch-trainer-demo"),
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    result = evaluate(args)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
