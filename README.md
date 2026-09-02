# RHOAI PyTorch Trainer Demo

A small, fully reproducible demonstration of **distributed PyTorch training on Red Hat
OpenShift AI**, orchestrated end-to-end from this GitHub repository:

```
GitHub repository
  |
  v
bootstrap.sh / make bootstrap
  |
  v
new OpenShift AI sandbox (namespace + RBAC)
  |
  v
Notebook (exploration)          training/train.py (reproducible workload)
  |                                        |
  v                                        v
OpenShift Build  ------------------->  training image
                                             |
                                             v
                                     TrainJob (Kubeflow Trainer, PyTorch distributed)
                                             |
                                             v
                                       AI Pipeline (prepare -> train -> evaluate)
                                             |
                                             v
                                     Scheduled run (optional, on demand)
                                             |
                                             v
                                     MLflow tracking (optional)
```

**The GitHub repository is the only source of truth.** The sandbox used to validate this
repository is temporary and will be deleted; nothing in this demo is allowed to depend on
a manual, undocumented change made directly on that sandbox. Anyone with `oc login` access
to a fresh OpenShift AI cluster and this repository can rebuild the entire demo.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the technical rationale and
[`DEMO_GUIDE.md`](DEMO_GUIDE.md) for the presenter walkthrough.

## What this demonstrates

1. A GitHub repository as the single source of truth for an ML training workload.
2. A notebook used purely for exploration (`notebooks/exploration.ipynb`).
3. That exploration code extracted into a reproducible script, `training/train.py`.
4. A real, distributed `TrainJob` (Kubeflow Trainer v2) built from an image compiled by an
   OpenShift Build directly from this repository -- no private registry required.
5. Multiple training workers actually created and coordinating (`RANK`/`WORLD_SIZE`).
6. Per-rank distributed training logs.
7. A real AI Pipeline (`prepare-data -> distributed-training -> evaluate-model`) that
   creates and waits on a genuine `TrainJob` -- never simulated.
8. A scheduled/recurring pipeline run example (not enabled by default).
9. Optional MLflow experiment tracking, entirely opt-in.

## Quickstart on a brand new sandbox

```bash
git clone <this-repository-url>
cd rhoai-pytorch-trainer-demo

oc login --token=... --server=...        # your credentials, never stored by this repo

make preflight     # read-only: what does this cluster actually support?
make bootstrap      # create the namespace + RBAC (idempotent)
make build          # build the training image from THIS repo via an OpenShift Build
make train           # run a real distributed PyTorch TrainJob
make compile-pipeline
make pipeline        # upload + run the AI Pipeline (needs a Pipeline Server, see below)
make validate         # PASS/FAIL report of everything above
```

or simply:

```bash
make demo
```

`make demo` runs preflight -> bootstrap -> build -> train -> pipeline (if a Pipeline
Server is available) -> validate, in order, and stops with a clear message at the first
hard failure.

## Nothing is hardcoded

Every variable below has a default (see the top of the `Makefile`) and can be overridden
with a plain environment variable, e.g.:

```bash
export NAMESPACE=my-demo
export TRAIN_NODES=3
export GPU_PER_NODE=1
make train
```

| Variable | Default | Meaning |
|---|---|---|
| `NAMESPACE` | `rhoai-training-demo` | Namespace created and owned by this demo |
| `IMAGE_STREAM_NAME` / `IMAGE_TAG` | `pytorch-trainer-demo` / `latest` | Training image built by the OpenShift Build |
| `GIT_REPO_URL` / `GIT_REF` | auto-detected from `git remote`/`git branch` | Source the OpenShift Build pulls from |
| `TRAIN_NODES` | `2` | Number of distributed training nodes |
| `GPU_PER_NODE` | `0` | GPUs requested per node (`0` = CPU-only demo mode) |
| `TRAIN_CPU` / `TRAIN_MEMORY` | `250m` / `768Mi` | CPU/memory requested per training node (deliberately tiny on CPU -- the workload is a synthetic toy model; memory is kept high enough for the PyTorch import itself. Lower/raise via env vars to fit your sandbox) |
| `TRAIN_EPOCHS` / `TRAIN_LR` / `TRAIN_BATCH_SIZE` | `5` / `0.01` / `32` | Training hyperparameters |
| `PIPELINE_NAME` | `pytorch-trainer-demo-pipeline` | Name used when uploading the pipeline |
| `SCHEDULE_CRON` | `0 2 * * *` | Cron expression for `make create-schedule` |
| `MLFLOW_TRACKING_URI` | unset | If set, training/evaluation log to this MLflow server |

Nothing hardcodes the sandbox hostname, cluster API, your username, a token, a specific
GPU model, or a storage class. `manifests/trainjob-gpu-example.yaml` is a static,
fully-written GPU example kept only for reference/reading.

## Repository layout

```
.
├── README.md                    this file
├── DEMO_GUIDE.md                 presenter walkthrough
├── ARCHITECTURE.md               technical rationale
├── Makefile                      all commands
├── requirements-dev.txt          local tooling only (never used inside the training image)
├── notebooks/exploration.ipynb   exploration phase
├── training/                     train.py, evaluate.py, Containerfile -- the reproducible workload
├── pipeline/                     pipeline.py (source) + pipeline.yaml (compiled) + requirements.txt
├── manifests/                    namespace, RBAC, BuildConfig/ImageStream, TrainJob templates
├── scripts/                      preflight, bootstrap, build, run-trainjob, deploy-pipeline, validate, cleanup
└── docs/troubleshooting.md
```

## Preflight: know your sandbox before touching it

```bash
make preflight
```

`scripts/preflight.sh` is strictly read-only. It reports `PASS` / `WARN` / `FAIL` /
`OPTIONAL` for: `oc` connectivity, OpenShift version, OpenShift AI version, the Kubeflow
Trainer v2 CRDs (`TrainJob`, `ClusterTrainingRuntime`, `TrainingRuntime`), the `JobSet`
CRD, a PyTorch-compatible `ClusterTrainingRuntime` (`torch-distributed`), GPU
visibility/availability, Data Science Pipelines, an existing
`DataSciencePipelinesApplication`, pipeline-server object storage, and MLflow. A hard
`FAIL` (missing `oc` login, or the Kubeflow Trainer CRDs entirely absent) stops before
`make bootstrap` would do anything.

## The demo owns only its own resources

Every resource created by this repository is labeled:

```yaml
app.kubernetes.io/part-of: rhoai-pytorch-trainer-demo
```

`make bootstrap` creates its own namespace (default `rhoai-training-demo`) and only
resources inside it (plus a dedicated `ServiceAccount`/`Role`/`RoleBinding`, never
cluster-admin). It never reuses or modifies an existing model, PVC, Secret, application,
or GPU workload that isn't this demo's. The only cluster-scoped resource it *reads* (never
writes) is the `ClusterTrainingRuntime` provided by OpenShift AI. `make cleanup` refuses to
delete a namespace that isn't labeled as belonging to this demo (see
[`ARCHITECTURE.md`](ARCHITECTURE.md)).

## Pipeline Server storage

`make pipeline` requires a Data Science Pipelines server (`DataSciencePipelinesApplication`)
to already be `Ready` in your namespace. This repository **does not** deploy one
automatically, because that requires S3-compatible object storage credentials, which must
never be committed to GitHub.

If you need to provision a Pipeline Server on a fresh sandbox:

1. Create an S3-compatible bucket (e.g. an existing S3 bucket, Noobaa/ODF, or MinIO).
2. Create a Secret with those credentials **directly in OpenShift** (never in this repo):
   ```bash
   oc create secret generic pipeline-storage \
     --from-literal=accesskey="$AWS_ACCESS_KEY_ID" \
     --from-literal=secretkey="$AWS_SECRET_ACCESS_KEY" \
     -n "$NAMESPACE"
   ```
3. Create a `DataSciencePipelinesApplication` (via the OpenShift AI Dashboard, or your own
   CR) referencing that Secret and your S3 endpoint/bucket.
4. Re-run `make preflight` to confirm it is detected, then `make pipeline`.

If no Pipeline Server is available, `scripts/deploy-pipeline.sh` says so explicitly and
exits non-zero -- it never pretends a pipeline ran.

## MLflow (optional)

MLflow is never deployed automatically by this repository. If `MLFLOW_TRACKING_URI` is set
(as a plain environment variable, or as a Secret you create yourself in OpenShift and wire
into the TrainJob's env), `training/train.py`, `training/evaluate.py`, and the pipeline's
`evaluate-model` step will log parameters/metrics/artifacts there. If it is unset, or the
`mlflow` package/server is unreachable, training and evaluation proceed identically minus
tracking -- MLflow is never a hard dependency.

## Commands

| Command | Purpose |
|---|---|
| `make help` | List all available targets |
| `make preflight` | Read-only cluster capability check |
| `make bootstrap` | Create namespace + RBAC (idempotent) |
| `make build` | Build the training image via an OpenShift Build |
| `make train` | Run a distributed TrainJob directly |
| `make compile-pipeline` | Compile `pipeline/pipeline.py` -> `pipeline/pipeline.yaml` |
| `make pipeline` | Upload + run the AI Pipeline |
| `make pipeline-status` | Check the latest pipeline run |
| `make create-schedule` / `make delete-schedule` | Manage the recurring run example |
| `make validate` | PASS/FAIL verification of the whole demo |
| `make status` | Quick snapshot of what currently exists |
| `make cleanup` | Delete only this demo's namespace/resources |
| `make demo` | Run the whole sequence above |

## Local development

```bash
make install     # creates .venv with local tooling (jupyter, torch-cpu, kfp SDK, ruff...)
make notebook     # launch notebooks/exploration.ipynb
make lint          # ruff check training/ pipeline/ scripts/
python training/train.py --epochs 5   # run the training script locally, single process
```

## Documentation

- [`DEMO_GUIDE.md`](DEMO_GUIDE.md) -- exact presenter walkthrough (what to show, what to say)
- [`ARCHITECTURE.md`](ARCHITECTURE.md) -- why the notebook isn't production, why the TrainJob
  isn't the script, why MLflow is complementary, and why GitHub is the source of truth
- [`docs/troubleshooting.md`](docs/troubleshooting.md)
