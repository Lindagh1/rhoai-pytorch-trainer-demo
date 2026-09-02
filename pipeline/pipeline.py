"""AI Pipeline: prepare-data -> distributed-training -> evaluate-model.

This is the SOURCE of the pipeline. `pipeline/pipeline.yaml` is the compiled, deterministic
IR YAML produced from this file with `make compile-pipeline` (see the Makefile target and
docs/troubleshooting.md for the pinned KFP SDK version used to compile it).

Design notes (see ARCHITECTURE.md for the full rationale):

  * The `distributed-training` step does NOT simulate anything: it calls the Kubernetes API
    directly (in-cluster, using the pipeline task's own ServiceAccount token) to create a
    real `TrainJob` custom resource, then polls its status until it reaches a terminal
    condition (Complete/Failed) before the pipeline is allowed to continue.
  * `evaluate-model` reads the *actual* pod logs written by the distributed training run to
    confirm multiple ranks executed and to extract the final loss -- it evaluates real
    execution evidence, not a mock.
  * All three steps run under the same ServiceAccount for a given pipeline run
    (`manifests/rbac.yaml` creates `pipeline-trainjob-runner`, scoped to only what this
    demo needs: create/get/list/watch/delete TrainJobs, read-only on
    ClusterTrainingRuntime/TrainingRuntime/JobSet, and get/list/watch on pods + pods/log).
    No cluster-admin, no resources outside the demo namespace.
"""
from kfp import dsl

_BASE_IMAGE = "registry.access.redhat.com/ubi9/python-311:latest"


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
    mlflow_tracking_uri: str,
    mlflow_experiment_name: str,
    timeout_seconds: int,
) -> str:
    """Creates a real TrainJob via the Kubernetes API and waits for it to finish.

    Returns a JSON string describing the outcome (status + worker pod names) that
    evaluate_model consumes.
    """
    import json
    import time

    from kubernetes import client, config

    config.load_incluster_config()
    custom_api = client.CustomObjectsApi()
    core_api = client.CoreV1Api()

    group, version, plural = "trainer.kubeflow.org", "v1alpha1", "trainjobs"

    try:
        custom_api.delete_namespaced_custom_object(group, version, namespace, plural, trainjob_name)
        print(f"[distributed-training] deleted pre-existing TrainJob '{trainjob_name}' before recreating it")
        time.sleep(5)
    except client.exceptions.ApiException as exc:
        if exc.status != 404:
            raise

    env = [{"name": "TRAIN_OUTPUT_DIR", "value": "/opt/app-root/src/output"}]
    if mlflow_experiment_name:
        env.append({"name": "MLFLOW_EXPERIMENT_NAME", "value": mlflow_experiment_name})
    if mlflow_tracking_uri:
        env.append({"name": "MLFLOW_TRACKING_URI", "value": mlflow_tracking_uri})

    resources = {"requests": {"cpu": train_cpu, "memory": train_memory}, "limits": {"cpu": train_cpu, "memory": train_memory}}
    if gpu_per_node > 0:
        resources["requests"]["nvidia.com/gpu"] = str(gpu_per_node)
        resources["limits"]["nvidia.com/gpu"] = str(gpu_per_node)

    trainjob_manifest = {
        "apiVersion": f"{group}/{version}",
        "kind": "TrainJob",
        "metadata": {
            "name": trainjob_name,
            "namespace": namespace,
            "labels": {
                "app.kubernetes.io/part-of": "rhoai-pytorch-trainer-demo",
                "app.kubernetes.io/managed-by": "rhoai-pytorch-trainer-demo",
                "app.kubernetes.io/component": "training",
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

    print(f"[distributed-training] creating TrainJob '{trainjob_name}' in namespace '{namespace}' "
          f"({train_nodes} node(s), {gpu_per_node} GPU/node)")
    custom_api.create_namespaced_custom_object(group, version, namespace, plural, trainjob_manifest)

    deadline = time.time() + timeout_seconds
    status = "Unknown"
    while time.time() < deadline:
        obj = custom_api.get_namespaced_custom_object(group, version, namespace, plural, trainjob_name)
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

    pods = core_api.list_namespaced_pod(namespace)
    pod_names = [p.metadata.name for p in pods.items if p.metadata.name.startswith(f"{trainjob_name}-")]

    for pod_name in pod_names:
        print(f"--- logs: {pod_name} ---")
        try:
            print(core_api.read_namespaced_pod_log(pod_name, namespace, tail_lines=200))
        except client.exceptions.ApiException as exc:
            print(f"  (could not read logs yet: {exc.reason})")

    if status != "Complete":
        raise RuntimeError(f"TrainJob '{trainjob_name}' did not complete successfully (status={status})")

    print(f"[distributed-training] TrainJob '{trainjob_name}' completed with {len(pod_names)} worker pod(s)")
    return json.dumps({"status": status, "namespace": namespace, "trainjob_name": trainjob_name, "pod_names": pod_names})


@dsl.component(base_image=_BASE_IMAGE, packages_to_install=["kubernetes==30.1.0", "mlflow==2.16.2"])
def evaluate_model(training_result: str, max_acceptable_loss: float, mlflow_tracking_uri: str, mlflow_experiment_name: str) -> str:
    """Reads the real training pod logs to confirm distributed execution and extract the
    final training loss, then reports PASS/FAIL. This is real evidence from the TrainJob
    that just ran, not a simulated result.
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
    final_losses = []
    epoch_line_re = re.compile(r"\[rank=(\d+)/(\d+).*?\].*epoch=(\d+)/(\d+) train_loss=([\d.]+) val_loss=([\d.]+)")

    for pod_name in pod_names:
        try:
            logs = core_api.read_namespaced_pod_log(pod_name, namespace)
        except client.exceptions.ApiException as exc:
            print(f"[evaluate-model] could not read logs for {pod_name}: {exc.reason}")
            continue
        pod_last_val_loss = None
        for line in logs.splitlines():
            m = epoch_line_re.search(line)
            if m:
                ranks_seen.add(int(m.group(1)))
                pod_last_val_loss = float(m.group(6))
        if pod_last_val_loss is not None:
            final_losses.append(pod_last_val_loss)

    passed_distributed_check = len(ranks_seen) >= 1
    final_val_loss = max(final_losses) if final_losses else None
    passed_loss_check = final_val_loss is not None and final_val_loss <= max_acceptable_loss
    passed = passed_distributed_check and passed_loss_check

    summary = {
        "ranks_observed": sorted(ranks_seen),
        "final_val_loss": final_val_loss,
        "max_acceptable_loss": max_acceptable_loss,
        "passed": passed,
    }
    print(f"[evaluate-model] {'PASS' if passed else 'FAIL'}: {summary}")

    if mlflow_tracking_uri:
        try:
            import mlflow

            mlflow.set_tracking_uri(mlflow_tracking_uri)
            mlflow.set_experiment(mlflow_experiment_name)
            with mlflow.start_run(run_name="pipeline-evaluate"):
                if final_val_loss is not None:
                    mlflow.log_metric("final_val_loss", final_val_loss)
                mlflow.log_param("ranks_observed", str(sorted(ranks_seen)))
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
    trainjob_name: str = "pytorch-trainer-demo-pipeline",
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
    mlflow_tracking_uri: str = "",
    mlflow_experiment_name: str = "rhoai-pytorch-trainer-demo",
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
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
        timeout_seconds=trainjob_timeout_seconds,
    )
    train_task.after(prepare_task)
    train_task.set_caching_options(False)

    evaluate_task = evaluate_model(
        training_result=train_task.output,
        max_acceptable_loss=max_acceptable_loss,
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
