#!/usr/bin/env python3
"""Thin Kubeflow Pipelines client used by scripts/deploy-pipeline.sh.

Connection details (the Data Science Pipelines API route + an OpenShift bearer token)
are always obtained live through `oc` -- never hardcoded, never written to disk, never
printed. This script only talks to a Pipeline Server if one is actually reachable; it
never pretends a pipeline ran when it did not (see README.md "Pipeline Server").

Subcommands:
  detect            -- exit 0 and print the route if a Pipeline Server is reachable, else exit 1
  upload            -- upload/version the compiled pipeline (pipeline/pipeline.yaml)
  run               -- start a one-off run of the uploaded pipeline
  status [run-id]   -- print the state of a run (latest run if run-id omitted)
  create-schedule   -- create a recurring run (see SCHEDULE_CRON)
  delete-schedule   -- delete the recurring run created by create-schedule
"""
import argparse
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_YAML = os.path.join(REPO_ROOT, "pipeline", "pipeline.yaml")

NAMESPACE = os.environ.get("NAMESPACE", "rhoai-training-demo")
PIPELINE_NAME = os.environ.get("PIPELINE_NAME", "pytorch-trainer-demo-pipeline")
IMAGE_STREAM_NAME = os.environ.get("IMAGE_STREAM_NAME", "pytorch-trainer-demo")
IMAGE_TAG = os.environ.get("IMAGE_TAG", "latest")
PIPELINE_SERVICE_ACCOUNT = os.environ.get("PIPELINE_SERVICE_ACCOUNT", "pipeline-trainjob-runner")
SCHEDULE_NAME = os.environ.get("SCHEDULE_NAME", "pytorch-trainer-demo-nightly")
SCHEDULE_CRON = os.environ.get("SCHEDULE_CRON", "0 2 * * *")


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def get_route():
    override = os.environ.get("PIPELINE_ROUTE_HOST")
    if override:
        return override
    result = sh(["oc", "get", "route", "-n", NAMESPACE, "-o",
                 "jsonpath={range .items[*]}{.metadata.name}={.spec.host}{\"\\n\"}{end}"])
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        name, host = line.split("=", 1)
        if name.startswith("ds-pipeline-") and "ui" not in name:
            return f"https://{host}"
    return None


def get_token():
    result = sh(["oc", "whoami", "-t"])
    return result.stdout.strip() if result.returncode == 0 else None


def build_client():
    from kfp import Client

    host = get_route()
    if not host:
        print("No Data Science Pipelines route found in namespace "
              f"'{NAMESPACE}'. Is a DataSciencePipelinesApplication deployed and Ready?", file=sys.stderr)
        return None
    token = get_token()
    if not token:
        print("Could not obtain an OpenShift token via 'oc whoami -t'. Are you logged in?", file=sys.stderr)
        return None

    ca_cert = os.environ.get("PIPELINE_SSL_CA_CERT")
    try:
        client = Client(host=host, existing_token=token, ssl_ca_cert=ca_cert)
        client.get_kfp_healthz()
        return client
    except Exception as exc:  # noqa: BLE001
        print(f"Could not reach the Pipeline Server at {host}: {exc}", file=sys.stderr)
        print("If this is a certificate error, set PIPELINE_SSL_CA_CERT to a CA bundle "
              "(see docs/troubleshooting.md).", file=sys.stderr)
        return None


def cmd_detect(_args):
    client = build_client()
    if client is None:
        print("NOT AVAILABLE")
        return 1
    print(f"AVAILABLE: {get_route()}")
    return 0


def get_or_create_experiment(client):
    experiment_name = "rhoai-pytorch-trainer-demo"
    try:
        exp = client.get_experiment(experiment_name=experiment_name)
    except Exception:  # noqa: BLE001
        exp = client.create_experiment(experiment_name)
    return exp


def cmd_upload(_args):
    client = build_client()
    if client is None:
        return 1
    if not os.path.exists(PIPELINE_YAML):
        print(f"{PIPELINE_YAML} not found. Run 'make compile-pipeline' first.", file=sys.stderr)
        return 1

    try:
        existing = client.get_pipeline_id(PIPELINE_NAME)
    except Exception:  # noqa: BLE001
        existing = None

    if existing:
        version_name = f"v-{int(time.time())}"
        client.upload_pipeline_version(PIPELINE_YAML, pipeline_version_name=version_name, pipeline_id=existing)
        print(f"Uploaded new version '{version_name}' of existing pipeline '{PIPELINE_NAME}' ({existing})")
    else:
        result = client.upload_pipeline(PIPELINE_YAML, pipeline_name=PIPELINE_NAME)
        print(f"Uploaded new pipeline '{PIPELINE_NAME}' ({result.pipeline_id})")
    return 0


def default_run_params():
    train_image = os.environ.get(
        "TRAIN_IMAGE",
        f"image-registry.openshift-image-registry.svc:5000/{NAMESPACE}/{IMAGE_STREAM_NAME}:{IMAGE_TAG}",
    )
    return {
        "namespace": NAMESPACE,
        "trainjob_name": os.environ.get("TRAINJOB_NAME", "pytorch-trainer-demo-pipeline"),
        "train_image": train_image,
        "train_nodes": int(os.environ.get("TRAIN_NODES", 2)),
        "gpu_per_node": int(os.environ.get("GPU_PER_NODE", 0)),
        "train_cpu": os.environ.get("TRAIN_CPU", "1"),
        "train_memory": os.environ.get("TRAIN_MEMORY", "2Gi"),
        "n_features": int(os.environ.get("TRAIN_N_FEATURES", 8)),
        "train_samples": int(os.environ.get("TRAIN_SAMPLES", 2048)),
        "val_samples": int(os.environ.get("VAL_SAMPLES", 512)),
        "seed": int(os.environ.get("TRAIN_SEED", 42)),
        "epochs": int(os.environ.get("TRAIN_EPOCHS", 5)),
        "lr": float(os.environ.get("TRAIN_LR", 0.01)),
        "batch_size": int(os.environ.get("TRAIN_BATCH_SIZE", 32)),
        "max_acceptable_loss": float(os.environ.get("MAX_ACCEPTABLE_MSE", 0.5)),
        "mlflow_tracking_uri": os.environ.get("MLFLOW_TRACKING_URI", ""),
        "mlflow_experiment_name": os.environ.get("MLFLOW_EXPERIMENT_NAME", "rhoai-pytorch-trainer-demo"),
        "trainjob_timeout_seconds": int(os.environ.get("TRAINJOB_WAIT_TIMEOUT_SECONDS", 1200)),
    }


def cmd_run(_args):
    client = build_client()
    if client is None:
        return 1
    pipeline_id = client.get_pipeline_id(PIPELINE_NAME)
    if not pipeline_id:
        print(f"Pipeline '{PIPELINE_NAME}' not found. Run 'make pipeline-upload' first.", file=sys.stderr)
        return 1
    experiment = get_or_create_experiment(client)
    run_name = f"{PIPELINE_NAME}-{int(time.time())}"
    run = client.run_pipeline(
        experiment_id=experiment.experiment_id,
        job_name=run_name,
        pipeline_id=pipeline_id,
        params=default_run_params(),
        service_account=PIPELINE_SERVICE_ACCOUNT,
    )
    print(f"Started run '{run_name}' (run_id={run.run_id})")
    print(f"run_id={run.run_id}")
    return 0


def cmd_status(args):
    client = build_client()
    if client is None:
        return 1
    run_id = args.run_id
    if not run_id:
        experiment = get_or_create_experiment(client)
        runs = client.list_runs(experiment_id=experiment.experiment_id, sort_by="created_at desc", page_size=1)
        if not runs.runs:
            print("No runs found.")
            return 1
        run_id = runs.runs[0].run_id
    run_detail = client.get_run(run_id)
    state = run_detail.state
    print(f"run_id={run_id} state={state}")
    return 0 if state == "SUCCEEDED" else (2 if state in ("RUNNING", "PENDING") else 1)


def cmd_create_schedule(_args):
    client = build_client()
    if client is None:
        return 1
    pipeline_id = client.get_pipeline_id(PIPELINE_NAME)
    if not pipeline_id:
        print(f"Pipeline '{PIPELINE_NAME}' not found. Run 'make pipeline-upload' first.", file=sys.stderr)
        return 1
    experiment = get_or_create_experiment(client)

    job = client.create_recurring_run(
        experiment_id=experiment.experiment_id,
        job_name=SCHEDULE_NAME,
        pipeline_id=pipeline_id,
        params=default_run_params(),
        cron_expression=SCHEDULE_CRON,
        enabled=True,
        service_account=PIPELINE_SERVICE_ACCOUNT,
    )
    print(f"Created recurring run '{SCHEDULE_NAME}' (cron='{SCHEDULE_CRON}', id={job.recurring_run_id})")
    print("This WILL consume cluster resources on every trigger until you run 'make delete-schedule'.")
    return 0


def cmd_delete_schedule(_args):
    client = build_client()
    if client is None:
        return 1
    try:
        jobs = client.list_recurring_runs(page_size=100)
    except Exception as exc:  # noqa: BLE001
        print(f"Could not list recurring runs: {exc}", file=sys.stderr)
        return 1
    matched = [j for j in jobs.recurring_runs or [] if j.display_name == SCHEDULE_NAME]
    if not matched:
        print(f"No recurring run named '{SCHEDULE_NAME}' found (nothing to delete).")
        return 0
    for job in matched:
        client.delete_recurring_run(job.recurring_run_id)
        print(f"Deleted recurring run '{SCHEDULE_NAME}' ({job.recurring_run_id})")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("detect")
    sub.add_parser("upload")
    sub.add_parser("run")
    status_parser = sub.add_parser("status")
    status_parser.add_argument("run_id", nargs="?", default=None)
    sub.add_parser("create-schedule")
    sub.add_parser("delete-schedule")

    args = parser.parse_args()
    handlers = {
        "detect": cmd_detect,
        "upload": cmd_upload,
        "run": cmd_run,
        "status": cmd_status,
        "create-schedule": cmd_create_schedule,
        "delete-schedule": cmd_delete_schedule,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
