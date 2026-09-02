#!/usr/bin/env python3
"""Distributed PyTorch training entrypoint for the RHOAI trainer demo.

This is the SAME modeling logic explored interactively in ``notebooks/exploration.ipynb``,
extracted into a script that is:

  * reproducible: versioned in this repository, built into an immutable container image
    (see ``training/Containerfile``), never edited in-place on the cluster.
  * distributed-aware: it reads ``RANK`` / ``WORLD_SIZE`` / ``LOCAL_RANK`` (and
    ``MASTER_ADDR`` / ``MASTER_PORT``) from the environment instead of hardcoding any
    topology. Those variables are injected by the Kubeflow Trainer runtime (via
    ``torchrun``) when this script runs inside a ``TrainJob``, and can also be left unset
    to run as a single local process (e.g. on a laptop, or inside the notebook).
  * intentionally tiny: the goal is to demonstrate *distributed orchestration*, not to
    train a competitive model. Training data is synthetic and generation is instant.

Usage (single process, e.g. laptop or notebook)::

    python training/train.py --epochs 5

Usage (distributed, launched by torchrun -- this is what the TrainJob does)::

    torchrun --nnodes=2 --nproc-per-node=1 training/train.py --epochs 5

MLflow is entirely optional. If ``MLFLOW_TRACKING_URI`` is set in the environment, rank 0
logs parameters/metrics/the model artifact there. If it is not set, training proceeds
exactly the same way, minus the tracking calls.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler


# --------------------------------------------------------------------------------------
# Synthetic dataset
# --------------------------------------------------------------------------------------
class SyntheticRegressionDataset(Dataset):
    """A tiny, deterministic, synthetic regression dataset.

    y = sum(w_true * x) + bias + noise

    Generated on the fly from a fixed seed so every rank/worker/notebook run produces the
    exact same underlying data -- only the *shard* each rank sees differs, via
    DistributedSampler.
    """

    def __init__(self, n_samples: int, n_features: int, seed: int = 42, noise_std: float = 0.1):
        generator = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n_samples, n_features, generator=generator)
        true_weights = torch.linspace(0.5, 2.0, n_features)
        true_bias = 1.5
        noise = torch.randn(n_samples, generator=generator) * noise_std
        self.y = (self.x @ true_weights + true_bias + noise).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
class TinyRegressor(nn.Module):
    """A deliberately small MLP -- this demo is about the training *architecture*,
    not about model capacity."""

    def __init__(self, n_features: int, hidden_size: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------------------
# Distributed environment detection
# --------------------------------------------------------------------------------------
@dataclass
class DistEnv:
    rank: int
    world_size: int
    local_rank: int
    is_distributed: bool
    backend: str
    device: str


def detect_dist_env() -> DistEnv:
    """Detect distributed training parameters from the environment.

    RANK / WORLD_SIZE / LOCAL_RANK are set by torchrun (which is how the Kubeflow Trainer
    runtime launches each process inside a TrainJob's Pods). When absent, this script runs
    as a single, non-distributed process -- this is what makes it usable identically on a
    laptop, in the notebook, and in a CPU-only TrainJob demo mode.
    """
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1

    use_cuda = torch.cuda.is_available()
    backend_override = os.environ.get("TORCH_DISTRIBUTED_BACKEND")
    if backend_override:
        backend = backend_override
    else:
        backend = "nccl" if use_cuda else "gloo"

    if use_cuda:
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"

    return DistEnv(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        is_distributed=is_distributed,
        backend=backend,
        device=device,
    )


def setup_distributed(env: DistEnv) -> None:
    if not env.is_distributed:
        return
    if env.device.startswith("cuda"):
        torch.cuda.set_device(env.local_rank)
    dist.init_process_group(backend=env.backend, rank=env.rank, world_size=env.world_size)
    log(env, f"process group initialized (backend={env.backend})")


def teardown_distributed(env: DistEnv) -> None:
    if env.is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def log(env: DistEnv, message: str) -> None:
    print(f"[rank={env.rank}/{env.world_size} local_rank={env.local_rank}] {message}", flush=True)


def all_reduce_mean(value: float, env: DistEnv) -> float:
    if not env.is_distributed:
        return value
    tensor = torch.tensor(value, device=env.device if env.device != "cpu" else "cpu")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return (tensor / env.world_size).item()


# --------------------------------------------------------------------------------------
# Optional MLflow integration
# --------------------------------------------------------------------------------------
class MLflowLogger:
    """Best-effort, optional MLflow logging.

    Only rank 0 ever talks to MLflow. If MLFLOW_TRACKING_URI is unset, or the mlflow
    package is unavailable, or the server is unreachable, training continues without
    tracking -- MLflow is never a hard dependency of this demo.
    """

    def __init__(self, env: DistEnv, experiment_name: str):
        self.enabled = False
        self.env = env
        if env.rank != 0:
            return
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if not tracking_uri:
            log(env, "MLFLOW_TRACKING_URI not set -- MLflow logging disabled (this is fine).")
            return
        try:
            import mlflow  # noqa: F401

            self._mlflow = mlflow
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            self._run = mlflow.start_run()
            self.enabled = True
            log(env, f"MLflow logging enabled -> {tracking_uri}")
        except Exception as exc:  # noqa: BLE001 - deliberately broad: MLflow is optional
            log(env, f"MLflow logging disabled (could not initialize: {exc})")
            self.enabled = False

    def log_params(self, params: dict) -> None:
        if self.enabled:
            try:
                self._mlflow.log_params(params)
            except Exception as exc:  # noqa: BLE001
                log(self.env, f"MLflow log_params failed (ignored): {exc}")

    def log_metrics(self, metrics: dict, step: int) -> None:
        if self.enabled:
            try:
                self._mlflow.log_metrics(metrics, step=step)
            except Exception as exc:  # noqa: BLE001
                log(self.env, f"MLflow log_metrics failed (ignored): {exc}")

    def log_model_artifact(self, path: Path) -> None:
        if self.enabled:
            try:
                self._mlflow.log_artifact(str(path))
            except Exception as exc:  # noqa: BLE001
                log(self.env, f"MLflow log_artifact failed (ignored): {exc}")

    def end(self) -> None:
        if self.enabled:
            try:
                self._mlflow.end_run()
            except Exception:  # noqa: BLE001
                pass


# --------------------------------------------------------------------------------------
# Optional S3 checkpoint persistence
# --------------------------------------------------------------------------------------
class CheckpointStore:
    """Best-effort upload of the model checkpoint + training metrics to an S3-compatible
    bucket (the demo's namespace-local MinIO instance, see manifests/storage.yaml), so
    the pipeline's evaluate-model step can download and load the *real* checkpoint file
    instead of only inferring success from pod logs.

    Entirely optional: if CHECKPOINT_S3_BUCKET is unset, or boto3/the endpoint is
    unreachable, training still succeeds -- this mirrors how MLflowLogger degrades.
    """

    def __init__(self, env: DistEnv):
        self.enabled = False
        self.env = env
        if env.rank != 0:
            return
        self.bucket = os.environ.get("CHECKPOINT_S3_BUCKET")
        if not self.bucket:
            log(env, "CHECKPOINT_S3_BUCKET not set -- S3 checkpoint upload disabled (this is fine).")
            return
        self.prefix = os.environ.get("CHECKPOINT_S3_PREFIX", "").strip("/")
        try:
            import boto3

            self._client = boto3.client(
                "s3",
                endpoint_url=os.environ.get("S3_ENDPOINT_URL"),
                aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
                region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
            )
            self.enabled = True
            log(env, f"S3 checkpoint upload enabled -> bucket={self.bucket} prefix={self.prefix or '(none)'}")
        except Exception as exc:  # noqa: BLE001 - deliberately broad: this is optional
            log(env, f"S3 checkpoint upload disabled (could not initialize client: {exc})")
            self.enabled = False

    def _key(self, filename: str) -> str:
        return f"{self.prefix}/{filename}" if self.prefix else filename

    def upload(self, path: Path) -> None:
        if not self.enabled:
            return
        try:
            key = self._key(path.name)
            self._client.upload_file(str(path), self.bucket, key)
            log(self.env, f"uploaded {path.name} to s3://{self.bucket}/{key}")
        except Exception as exc:  # noqa: BLE001
            log(self.env, f"S3 upload of {path.name} failed (ignored, non-fatal): {exc}")


# --------------------------------------------------------------------------------------
# Training / validation loop
# --------------------------------------------------------------------------------------
def build_dataloaders(env: DistEnv, args: argparse.Namespace):
    train_ds = SyntheticRegressionDataset(
        n_samples=args.train_samples, n_features=args.n_features, seed=args.seed
    )
    val_ds = SyntheticRegressionDataset(
        n_samples=args.val_samples, n_features=args.n_features, seed=args.seed + 1
    )

    if env.is_distributed:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=env.world_size, rank=env.rank, shuffle=True, seed=args.seed
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        train_sampler = None

    # Validation is evaluated identically (whole set) on every rank; only rank 0 reports it.
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader, train_sampler


def train(args: argparse.Namespace) -> Path:
    env = detect_dist_env()
    setup_distributed(env)

    log(env, f"starting training: epochs={args.epochs} lr={args.lr} batch_size={args.batch_size}")
    log(env, f"world_size={env.world_size} distributed={env.is_distributed} device={env.device}")

    torch.manual_seed(args.seed + env.rank)

    train_loader, val_loader, train_sampler = build_dataloaders(env, args)

    model = TinyRegressor(n_features=args.n_features, hidden_size=args.hidden_size).to(env.device)
    if env.is_distributed:
        ddp_kwargs = {"device_ids": [env.local_rank]} if env.device.startswith("cuda") else {}
        model = DDP(model, **ddp_kwargs)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    git_sha = os.environ.get("GIT_SHA", "")
    trainjob_name = os.environ.get("TRAINJOB_NAME", "")
    pipeline_run_id = os.environ.get("PIPELINE_RUN_ID", "")

    mlflow_logger = MLflowLogger(env, experiment_name=args.experiment_name)
    mlflow_logger.log_params(
        {
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "hidden_size": args.hidden_size,
            "n_features": args.n_features,
            "world_size": env.world_size,
            "backend": env.backend,
        }
    )
    if mlflow_logger.enabled:
        try:
            tags = {"trainjob_name": trainjob_name, "git_commit": git_sha, "world_size": str(env.world_size)}
            if pipeline_run_id:
                tags["pipeline_run"] = pipeline_run_id
            mlflow_logger._mlflow.set_tags({k: v for k, v in tags.items() if v})
        except Exception as exc:  # noqa: BLE001
            log(env, f"MLflow set_tags failed (ignored): {exc}")

    checkpoint_store = CheckpointStore(env)

    history = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        running_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            x, y = x.to(env.device), y.to(env.device)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1

        local_train_loss = running_loss / max(n_batches, 1)
        global_train_loss = all_reduce_mean(local_train_loss, env)

        val_loss = evaluate_loss(model, val_loader, loss_fn, env.device)
        global_val_loss = all_reduce_mean(val_loss, env)

        log(
            env,
            f"epoch={epoch}/{args.epochs} train_loss={global_train_loss:.6f} "
            f"val_loss={global_val_loss:.6f}",
        )

        if env.rank == 0:
            history.append(
                {"epoch": epoch, "train_loss": global_train_loss, "val_loss": global_val_loss}
            )
            mlflow_logger.log_metrics(
                {"train_loss": global_train_loss, "val_loss": global_val_loss}, step=epoch
            )

    elapsed = time.time() - start_time
    log(env, f"training complete in {elapsed:.1f}s")

    output_path = Path(args.output_dir)
    if env.rank == 0:
        output_path.mkdir(parents=True, exist_ok=True)
        raw_model = model.module if isinstance(model, DDP) else model
        checkpoint_path = output_path / "model.pt"
        torch.save(raw_model.state_dict(), checkpoint_path)

        metrics_path = output_path / "training_metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "history": history,
                    "final_train_loss": history[-1]["train_loss"] if history else None,
                    "final_val_loss": history[-1]["val_loss"] if history else None,
                    "world_size": env.world_size,
                    "elapsed_seconds": elapsed,
                    "git_sha": git_sha,
                    "trainjob_name": trainjob_name,
                    "pipeline_run_id": pipeline_run_id,
                    "config": {
                        "epochs": args.epochs,
                        "lr": args.lr,
                        "batch_size": args.batch_size,
                        "hidden_size": args.hidden_size,
                        "n_features": args.n_features,
                    },
                },
                indent=2,
            )
        )
        log(env, f"model checkpoint written to {checkpoint_path}")
        log(env, f"training metrics written to {metrics_path}")

        mlflow_logger.log_model_artifact(checkpoint_path)
        mlflow_logger.log_model_artifact(metrics_path)
        checkpoint_store.upload(checkpoint_path)
        checkpoint_store.upload(metrics_path)

    mlflow_logger.end()
    teardown_distributed(env)
    return output_path


def evaluate_loss(model, loader, loss_fn, device) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            total_loss += loss_fn(pred, y).item()
            n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("TRAIN_EPOCHS", 5)))
    parser.add_argument("--lr", type=float, default=float(os.environ.get("TRAIN_LR", 0.01)))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("TRAIN_BATCH_SIZE", 32)))
    parser.add_argument("--hidden-size", type=int, default=int(os.environ.get("TRAIN_HIDDEN_SIZE", 32)))
    parser.add_argument("--n-features", type=int, default=int(os.environ.get("TRAIN_N_FEATURES", 8)))
    parser.add_argument("--train-samples", type=int, default=int(os.environ.get("TRAIN_SAMPLES", 2048)))
    parser.add_argument("--val-samples", type=int, default=int(os.environ.get("VAL_SAMPLES", 512)))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("TRAIN_SEED", 42)))
    parser.add_argument(
        "--output-dir", type=str, default=os.environ.get("TRAIN_OUTPUT_DIR", "/opt/app-root/src/output")
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "rhoai-pytorch-trainer-demo"),
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    try:
        train(args)
    except Exception:
        # Make sure the process group is torn down even on failure so the TrainJob
        # reports a clean failure instead of hanging.
        env = detect_dist_env()
        teardown_distributed(env)
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
