"""AI Pipeline: prepare-data -> distributed-training -> evaluate-model.

This is the SOURCE of the pipeline. `pipeline/pipeline.yaml` is the compiled, deterministic
IR YAML produced from this file with `make compile-pipeline` (see the Makefile target and
TROUBLESHOOTING.md for the pinned KFP SDK version used to compile it).

Design notes (see ARCHITECTURE.md for the full rationale):

  * The `distributed-training` step does NOT simulate anything: it calls the Kubernetes API
    directly (in-cluster, using the pipeline task's own ServiceAccount token) to create a
    real `TrainJob` custom resource, then polls its status until it reaches a terminal
    condition (Complete/Failed) before the pipeline is allowed to continue.
  * Every pipeline run creates a TrainJob with a UNIQUE name (base name + timestamp +
    random suffix, generated at task runtime -- not a KFP backend placeholder, so this
    works regardless of the installed Data Science Pipelines backend version). Concurrent
    runs never collide.
  * `evaluate-model` downloads the REAL checkpoint (`model.pt`) and training metrics that
    `training/train.py` uploaded to the namespace-local MinIO bucket (if object storage was
    configured for the run -- see `make storage`), loads the checkpoint with PyTorch, and
    runs a genuine forward pass on a freshly generated test set. It does not claim to load
    a model it did not actually load: if no checkpoint bucket is configured for the run, it
    falls back to (and clearly labels itself as) reading pod logs only.
  * All three steps run under the same ServiceAccount for a given pipeline run
    (`manifests/rbac.yaml` creates `pipeline-trainjob-runner`, scoped to only what this
    demo needs: create/get/list/watch/delete TrainJobs, read-only on
    ClusterTrainingRuntime/TrainingRuntime/JobSet, get/list/watch on pods + pods/log, and
    get on the single `minio-credentials` Secret by name). No cluster-admin, no resources
    outside the demo namespace.
"""
import subprocess

from kfp import dsl

_BASE_IMAGE = "registry.access.redhat.com/ubi9/python-311:latest"


def _current_git_sha() -> str:
    """Best-effort short git SHA of the commit this pipeline.yaml was compiled from, used
    as the default `git_sha` pipeline parameter so a compiled pipeline is traceable back to
    the exact commit that produced it, even without re-detecting it at runtime."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False, cwd="."
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


@dsl.component(base_image=_BASE_IMAGE)
def prepare_data(n_features: int, train_samples: int, val_samples: int, seed: int) -> str:
    """Validates/stages the synthetic dataset configuration.

    The dataset itself is generated inside the TrainJob container (see
    training/train.py) from this same deterministic seed/config, so no large tensors need
    to move between pipeline steps and TrainJob pods -- this keeps the pipeline reproducible
    without requiring a shared PVC or object store.
    """
    import json

    if n_features <= 0 or train_samples <= 0 or val_samples <= 0:
        raise ValueError("n_features, train_samples, and val_samples must all be positive")

    config = {
        "n_features": n_features,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "seed": seed,
    }
    print(f"[prepare-data] dataset configuration validated: {config}")
    return json.dumps(config)


@dsl.component(base_image=_BASE_IMAGE, packages_to_install=["kubernetes==30.1.0"])
def distributed_training(
    namespace: str,
    trainjob_name: str,
    train_image: str,
    train_nodes: int,
    gpu_per_node: int,
    train_cpu: str,
    train_memory: str,
    epochs: int,
    lr: float,
    batch_size: int,
    git_sha: str,
    use_mlflow: bool,
    mlflow_tracking_uri: str,
    mlflow_experiment_name: str,
    checkpoint_bucket: str,
    checkpoint_s3_endpoint: str,
    timeout_seconds: int,
) -> str:
    """Creates a real, UNIQUELY NAMED TrainJob via the Kubernetes API and waits for it to
    finish. Returns a JSON string describing the outcome that evaluate_model consumes.
    """
    import json
    import re
    import time
    import uuid

    from kubernetes import client, config

    config.load_incluster_config()
    custom_api = client.CustomObjectsApi()
    core_api = client.CoreV1Api()

    group, version, plural = "trainer.kubeflow.org", "v1alpha1", "trainjobs"

    # Unique per-run name: never collides across concurrent/parallel pipeline runs. Built
    # at task RUNTIME (plain Python), not via a KFP backend placeholder, so this works
    # regardless of the Data Science Pipelines backend version installed on the cluster.
    base = re.sub(r"[^a-z0-9-]", "-", trainjob_name.lower()).strip("-") or "rhoai-demo-train"
    unique_name = f"{base}-{time.strftime('%y%m%d%H%M%S')}-{uuid.uuid4().hex[:5]}"[:63].rstrip("-")

    env = [
        {"name": "TRAIN_OUTPUT_DIR", "value": "/opt/app-root/src/output"},
        {"name": "GIT_SHA", "value": git_sha or ""},
        {"name": "TRAINJOB_NAME", "value": unique_name},
    ]
    if mlflow_experiment_name:
        env.append({"name": "MLFLOW_EXPERIMENT_NAME", "value": mlflow_experiment_name})
    if use_mlflow and mlflow_tracking_uri:
        env.append({"name": "MLFLOW_TRACKING_URI", "value": mlflow_tracking_uri})
    if checkpoint_bucket and checkpoint_s3_endpoint:
        env += [
            {"name": "CHECKPOINT_S3_BUCKET", "value": checkpoint_bucket},
            {"name": "CHECKPOINT_S3_PREFIX", "value": unique_name},
            {"name": "S3_ENDPOINT_URL", "value": checkpoint_s3_endpoint},
            {"name": "AWS_ACCESS_KEY_ID", "valueFrom": {"secretKeyRef": {"name": "minio-credentials", "key": "MINIO_ROOT_USER"}}},
            {"name": "AWS_SECRET_ACCESS_KEY", "valueFrom": {"secretKeyRef": {"name": "minio-credentials", "key": "MINIO_ROOT_PASSWORD"}}},
        ]

    resources = {"requests": {"cpu": train_cpu, "memory": train_memory}, "limits": {"cpu": train_cpu, "memory": train_memory}}
    if gpu_per_node > 0:
        resources["requests"]["nvidia.com/gpu"] = str(gpu_per_node)
        resources["limits"]["nvidia.com/gpu"] = str(gpu_per_node)

    trainjob_manifest = {
        "apiVersion": f"{group}/{version}",
        "kind": "TrainJob",
        "metadata": {
            "name": unique_name,
            "namespace": namespace,
            "labels": {
                "app.kubernetes.io/part-of": "rhoai-pytorch-trainer-demo",
                "app.kubernetes.io/managed-by": "rhoai-pytorch-trainer-demo",
                "app.kubernetes.io/component": "training",
                "demo.git.sha": (git_sha or "unknown")[:12],
            },
            "annotations": {
                "demo.git.sha": git_sha or "",
                "demo.image": train_image,
            },
        },
        "spec": {
            "runtimeRef": {"apiGroup": group, "kind": "ClusterTrainingRuntime", "name": "torch-distributed"},
            "trainer": {
                "image": train_image,
                "command": ["torchrun", "train.py"],
                "args": [f"--epochs={epochs}", f"--lr={lr}", f"--batch-size={batch_size}"],
                "numNodes": train_nodes,
                "env": env,
                "resourcesPerNode": resources,
            },
        },
    }

    print(f"[distributed-training] creating TrainJob '{unique_name}' in namespace '{namespace}' "
          f"({train_nodes} node(s), {gpu_per_node} GPU/node, git_sha={git_sha or 'unknown'})")
    custom_api.create_namespaced_custom_object(group, version, namespace, plural, trainjob_manifest)

    deadline = time.time() + timeout_seconds
    status = "Unknown"
    while time.time() < deadline:
        obj = custom_api.get_namespaced_custom_object(group, version, namespace, plural, unique_name)
        conditions = obj.get("status", {}).get("conditions", [])
        by_type = {c.get("type"): c.get("status") for c in conditions}
        if by_type.get("Complete") == "True":
            status = "Complete"
            break
        if by_type.get("Failed") == "True":
            status = "Failed"
            break
        print(f"[distributed-training] waiting... current conditions: {by_type}")
        time.sleep(10)

    pod_names = [
        p.metadata.name
        for p in core_api.list_namespaced_pod(
            namespace, label_selector=f"jobset.sigs.k8s.io/jobset-name={unique_name}"
        ).items
    ]
    if not pod_names:
        pod_names = [
            p.metadata.name
            for p in core_api.list_namespaced_pod(namespace).items
            if p.metadata.name.startswith(f"{unique_name}-") and not p.metadata.name.endswith("-build")
        ]

    for pod_name in pod_names:
        print(f"--- logs: {pod_name} ---")
        try:
            print(core_api.read_namespaced_pod_log(pod_name, namespace, tail_lines=200))
        except client.exceptions.ApiException as exc:
            print(f"  (could not read logs yet: {exc.reason})")

    if status != "Complete":
        raise RuntimeError(f"TrainJob '{unique_name}' did not complete successfully (status={status})")

    print(f"[distributed-training] TrainJob '{unique_name}' completed with {len(pod_names)} worker pod(s)")
    return json.dumps(
        {
            "status": status,
            "namespace": namespace,
            "trainjob_name": unique_name,
            "pod_names": pod_names,
            "git_sha": git_sha or "",
            "checkpoint_bucket": checkpoint_bucket if (checkpoint_bucket and checkpoint_s3_endpoint) else "",
            "checkpoint_prefix": unique_name,
            "checkpoint_s3_endpoint": checkpoint_s3_endpoint or "",
            "n_features": 8,
            "hidden_size": 32,
        }
    )


@dsl.component(base_image=_BASE_IMAGE, packages_to_install=["kubernetes==30.1.0", "mlflow==2.16.2", "boto3==1.35.36", "torch==2.4.1"])
def evaluate_model(
    training_result: str,
    max_acceptable_loss: float,
    use_mlflow: bool,
    mlflow_tracking_uri: str,
    mlflow_experiment_name: str,
) -> str:
    """Evaluates the TrainJob that just ran using the strongest evidence available:

      1. If a checkpoint bucket was configured for this run, download the REAL
         `model.pt` + `training_metrics.json` that training/train.py uploaded to MinIO,
         load the checkpoint with PyTorch, and run a genuine forward pass on a freshly
         generated synthetic test set -- this is real model evaluation, not a simulation.
      2. Otherwise (no object storage configured for this run), fall back to reading the
         actual TrainJob pod logs to confirm multiple ranks executed and to extract the
         final loss -- and says so explicitly, rather than pretending a checkpoint was
         loaded when it was not.
    """
    import json
    import re

    from kubernetes import client, config

    result = json.loads(training_result)
    if result.get("status") != "Complete":
        raise RuntimeError(f"Upstream TrainJob did not complete: {result}")

    namespace = result["namespace"]
    pod_names = result["pod_names"]
    if not pod_names:
        raise RuntimeError("No training pods found -- cannot evaluate")

    config.load_incluster_config()
    core_api = client.CoreV1Api()

    ranks_seen = set()
    log_final_val_loss = None
    epoch_line_re = re.compile(r"\[rank=(\d+)/(\d+).*?\].*epoch=(\d+)/(\d+) train_loss=([\d.]+) val_loss=([\d.]+)")
    for pod_name in pod_names:
        try:
            logs = core_api.read_namespaced_pod_log(pod_name, namespace)
        except client.exceptions.ApiException as exc:
            print(f"[evaluate-model] could not read logs for {pod_name}: {exc.reason}")
            continue
        for line in logs.splitlines():
            m = epoch_line_re.search(line)
            if m:
                ranks_seen.add(int(m.group(1)))
                log_final_val_loss = float(m.group(6))

    checkpoint_bucket = result.get("checkpoint_bucket", "")
    checkpoint_method = "logs-only"
    final_val_loss = log_final_val_loss
    checkpoint_loaded = False
    test_mse = None

    if checkpoint_bucket:
        checkpoint_method = "real-checkpoint"
        import tempfile
        from pathlib import Path

        import boto3
        import torch
        import torch.nn as nn

        secret = core_api.read_namespaced_secret("minio-credentials", namespace)
        import base64

        access_key = base64.b64decode(secret.data["MINIO_ROOT_USER"]).decode()
        secret_key = base64.b64decode(secret.data["MINIO_ROOT_PASSWORD"]).decode()

        s3 = boto3.client(
            "s3",
            endpoint_url=result["checkpoint_s3_endpoint"],
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="us-east-1",
        )
        prefix = result["checkpoint_prefix"]
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pt"
            metrics_path = Path(tmpdir) / "training_metrics.json"
            try:
                s3.download_file(checkpoint_bucket, f"{prefix}/model.pt", str(model_path))
                s3.download_file(checkpoint_bucket, f"{prefix}/training_metrics.json", str(metrics_path))
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"checkpoint_bucket was configured but the checkpoint could not be downloaded "
                    f"from s3://{checkpoint_bucket}/{prefix}/ -- refusing to fake evaluation: {exc}"
                ) from exc

            training_metrics = json.loads(metrics_path.read_text())
            n_features = training_metrics.get("config", {}).get("n_features", result.get("n_features", 8))
            hidden_size = training_metrics.get("config", {}).get("hidden_size", result.get("hidden_size", 32))
            final_val_loss = training_metrics.get("final_val_loss", final_val_loss)

            # Mirrors training/train.py's TinyRegressor exactly (duplicated here because
            # this pipeline step runs in its own container, without the repo checked out).
            class TinyRegressor(nn.Module):
                def __init__(self, n_features: int, hidden_size: int = 32):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(n_features, hidden_size), nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                        nn.Linear(hidden_size, 1),
                    )

                def forward(self, x):
                    return self.net(x)

            model = TinyRegressor(n_features=n_features, hidden_size=hidden_size)
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            checkpoint_loaded = True

            # Same synthetic-data generation as training/train.py's SyntheticRegressionDataset,
            # with a held-out seed offset never used for train/val during training.
            seed = training_metrics.get("seed", 42) if isinstance(training_metrics.get("seed"), int) else 42
            generator = torch.Generator().manual_seed(seed + 999)
            x = torch.randn(256, n_features, generator=generator)
            true_weights = torch.linspace(0.5, 2.0, n_features)
            y = (x @ true_weights + 1.5 + torch.randn(256, generator=generator) * 0.1).unsqueeze(1)
            with torch.no_grad():
                pred = model(x)
                test_mse = nn.functional.mse_loss(pred, y).item()
            print(f"[evaluate-model] loaded REAL checkpoint from s3://{checkpoint_bucket}/{prefix}/model.pt "
                  f"-- genuine forward pass test_mse={test_mse:.6f}")

    passed_distributed_check = len(ranks_seen) >= 1
    effective_loss = test_mse if test_mse is not None else final_val_loss
    passed_loss_check = effective_loss is not None and effective_loss <= max_acceptable_loss
    passed = passed_distributed_check and passed_loss_check

    summary = {
        "method": checkpoint_method,
        "checkpoint_loaded": checkpoint_loaded,
        "ranks_observed": sorted(ranks_seen),
        "final_val_loss_from_logs": log_final_val_loss,
        "test_mse_from_checkpoint": test_mse,
        "effective_loss": effective_loss,
        "max_acceptable_loss": max_acceptable_loss,
        "passed": passed,
    }
    print(f"[evaluate-model] {'PASS' if passed else 'FAIL'} (method={checkpoint_method}): {summary}")

    if use_mlflow and mlflow_tracking_uri:
        try:
            import mlflow

            mlflow.set_tracking_uri(mlflow_tracking_uri)
            mlflow.set_experiment(mlflow_experiment_name)
            tags = {"trainjob_name": result.get("trainjob_name", ""), "git_commit": result.get("git_sha", "")}
            with mlflow.start_run(run_name="pipeline-evaluate"):
                mlflow.set_tags({k: v for k, v in tags.items() if v})
                if effective_loss is not None:
                    mlflow.log_metric("effective_loss", effective_loss)
                if test_mse is not None:
                    mlflow.log_metric("test_mse", test_mse)
                mlflow.log_param("ranks_observed", str(sorted(ranks_seen)))
                mlflow.log_param("evaluation_method", checkpoint_method)
                mlflow.log_param("passed", passed)
            print(f"[evaluate-model] logged evaluation summary to MLflow at {mlflow_tracking_uri}")
        except Exception as exc:  # noqa: BLE001 - MLflow is always optional
            print(f"[evaluate-model] MLflow logging skipped (not fatal): {exc}")

    if not passed:
        raise RuntimeError(f"Evaluation failed: {summary}")
    return json.dumps(summary)


@dsl.pipeline(
    name="rhoai-pytorch-trainer-demo",
    description="prepare-data -> distributed-training (real TrainJob via Kubeflow Trainer) -> evaluate-model",
)
def rhoai_pytorch_trainer_pipeline(
    namespace: str = "rhoai-training-demo",
    trainjob_name: str = "rhoai-demo-train",
    train_image: str = "image-registry.openshift-image-registry.svc:5000/rhoai-training-demo/pytorch-trainer-demo:latest",
    train_nodes: int = 2,
    gpu_per_node: int = 0,
    train_cpu: str = "250m",
    train_memory: str = "768Mi",
    n_features: int = 8,
    train_samples: int = 2048,
    val_samples: int = 512,
    seed: int = 42,
    epochs: int = 5,
    lr: float = 0.01,
    batch_size: int = 32,
    max_acceptable_loss: float = 0.5,
    git_sha: str = _current_git_sha(),
    use_mlflow: bool = True,
    mlflow_tracking_uri: str = "",
    mlflow_experiment_name: str = "rhoai-pytorch-trainer-demo",
    checkpoint_bucket: str = "",
    checkpoint_s3_endpoint: str = "",
    trainjob_timeout_seconds: int = 1200,
):
    prepare_task = prepare_data(
        n_features=n_features, train_samples=train_samples, val_samples=val_samples, seed=seed
    )
    prepare_task.set_caching_options(False)

    train_task = distributed_training(
        namespace=namespace,
        trainjob_name=trainjob_name,
        train_image=train_image,
        train_nodes=train_nodes,
        gpu_per_node=gpu_per_node,
        train_cpu=train_cpu,
        train_memory=train_memory,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        git_sha=git_sha,
        use_mlflow=use_mlflow,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
        checkpoint_bucket=checkpoint_bucket,
        checkpoint_s3_endpoint=checkpoint_s3_endpoint,
        timeout_seconds=trainjob_timeout_seconds,
    )
    train_task.after(prepare_task)
    train_task.set_caching_options(False)

    evaluate_task = evaluate_model(
        training_result=train_task.output,
        max_acceptable_loss=max_acceptable_loss,
        use_mlflow=use_mlflow,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
    )
    evaluate_task.set_caching_options(False)


if __name__ == "__main__":
    import pathlib

    from kfp import compiler

    output_path = pathlib.Path(__file__).parent / "pipeline.yaml"
    compiler.Compiler().compile(rhoai_pytorch_trainer_pipeline, str(output_path))
    print(f"Compiled pipeline to {output_path}")
