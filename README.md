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

make preflight        # read-only: what does this cluster actually support?
make bootstrap        # create the namespace + RBAC (idempotent)
make storage          # namespace-local MinIO (S3-compatible object storage)
make pipeline-server   # DataSciencePipelinesApplication, waits for Ready
make mlflow             # (optional) lightweight MLflow tracking server, auto-detected afterwards
make build             # build the training image from THIS repo via an OpenShift Build
make train MODE=cpu    # run a real distributed PyTorch TrainJob (MODE=cpu|gpu)
make compile-pipeline
make pipeline          # upload + run the AI Pipeline (needs the Pipeline Server above)
make schedule           # (optional) create a recurring/scheduled pipeline run
make validate           # PASS/FAIL report of everything above, incl. a secret scan
```

or simply:

```bash
make demo
```

`make demo` runs preflight -> bootstrap -> storage -> pipeline-server -> build -> train ->
pipeline (skipped with a clear `WARN` if the Pipeline Server isn't Ready) -> validate ->
status, in order, and stops with a clear message at the first hard failure. Re-run it any
time -- every step is idempotent. `make demo-reset` deletes this demo's namespace first
and then runs `make demo` again, to prove the whole thing is rebuildable from zero.

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
| `MODE` | unset | Convenience switch: `MODE=cpu` forces `GPU_PER_NODE=0`, `MODE=gpu` defaults `GPU_PER_NODE=1`. An explicit `GPU_PER_NODE` always wins over `MODE`. |
| `TRAIN_NODES` | `2` | Number of distributed training nodes |
| `GPU_PER_NODE` | unset (`0` unless `MODE=gpu`) | GPUs requested per node |
| `TRAIN_CPU` / `TRAIN_MEMORY` | `250m` / `768Mi` | CPU/memory requested per training node (deliberately tiny on CPU -- the workload is a synthetic toy model; memory is kept high enough for the PyTorch import itself. Lower/raise via env vars to fit your sandbox) |
| `TRAIN_EPOCHS` / `TRAIN_LR` / `TRAIN_BATCH_SIZE` | `5` / `0.01` / `32` | Training hyperparameters |
| `PIPELINE_NAME` | `pytorch-trainer-demo-pipeline` | Name used when uploading the pipeline |
| `USE_MLFLOW` | `auto` | `auto`/unset = use this demo's own MLflow if `make mlflow` was run; `false` = never log to MLflow even if available |
| `MLFLOW_TRACKING_URI` | auto-detected (this demo's own MLflow `Route`, if `make mlflow` was run) | Explicit override; if unset and no such Route exists, tracking is simply skipped |
| `CHECKPOINT_BUCKET` / `PIPELINE_BUCKET` / `MLFLOW_BUCKET` | `checkpoints` / `mlpipeline` / `mlflow` | Bucket names created by `make storage` inside the namespace-local MinIO |
| `SCHEDULE_CRON` | `0 2 * * 0` | Cron expression for `make schedule` (default: once a week) |

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
├── mlflow/                       Containerfile + entrypoint.sh for the optional MLflow server image
├── pipeline/                     pipeline.py (source) + pipeline.yaml (compiled) + requirements.txt
├── manifests/                    namespace, RBAC, BuildConfig/ImageStream, TrainJob, storage (MinIO), DSPA, MLflow
├── scripts/                      preflight, bootstrap, storage, pipeline-server, mlflow, build, run-trainjob,
│                                  deploy-pipeline, validate, security-check, negative-tests, status, cleanup
└── TROUBLESHOOTING.md
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

`make pipeline` requires a Data Science Pipelines server
(`DataSciencePipelinesApplication`, "DSPA") to already be `Ready` in your namespace, which
in turn requires an S3-compatible bucket. This repository provisions BOTH reproducibly,
entirely inside your own namespace, with no manually-created external bucket and no secret
ever committed to Git:

```bash
make storage           # namespace-local MinIO: PVC + Deployment + Service + Secret (random creds)
make pipeline-server   # DataSciencePipelinesApplication referencing that MinIO, waits for Ready
make preflight          # confirm OBJECT STORAGE / PIPELINE SERVER / DSPA STATUS are all PASS
make pipeline            # only now will this actually upload+run the pipeline
```

`scripts/storage.sh` generates a random MinIO root user/password on first run and stores it
ONLY in a Kubernetes `Secret` (`minio-credentials`) inside your namespace -- it is never
printed, written to a file, or committed. Re-running `make storage` is idempotent (existing
credentials/buckets are reused, never rotated or duplicated).

If you already have a real S3-compatible bucket and/or an external DSPA you'd rather use,
skip `make storage`/`make pipeline-server` and create your own `DataSciencePipelinesApplication`
(via the Dashboard or your own CR) referencing it instead, then re-run `make preflight` to
confirm it is detected.

If no Pipeline Server is `Ready`, `scripts/deploy-pipeline.sh` says so explicitly and exits
non-zero -- it never pretends a pipeline ran.

## MLflow (optional, but really deployable)

MLflow is never deployed as part of `make demo` -- but it is one command away:

```bash
make storage   # MLflow's artifact store needs the same namespace-local MinIO
make mlflow    # builds mlflow/Containerfile via an OpenShift Build, deploys, waits for Ready
```

Once deployed, `make train` / `make pipeline` **auto-detect** it (via its `Route`) with no
further configuration -- see `detect_mlflow_uri` in `scripts/lib.sh`. Set `USE_MLFLOW=false`
to skip logging even if an instance is available, or export `MLFLOW_TRACKING_URI` explicitly
to point at a different (e.g. externally managed) MLflow server instead of this demo's own.
`training/train.py`, `training/evaluate.py`, and the pipeline's `evaluate-model` step log
`learning_rate`/`epochs`/`batch_size`/`world_size`/`loss` (and tag every run with
`trainjob_name` + `git_commit`) whenever a tracking URI is available. If none is available,
or the `mlflow` package/server is unreachable, training and evaluation proceed identically
minus tracking -- MLflow is never a hard dependency.

## Commands

| Command | Purpose |
|---|---|
| `make help` | List all available targets |
| `make preflight` | Read-only cluster capability check |
| `make bootstrap` | Create namespace + RBAC (idempotent) |
| `make storage` | Deploy namespace-local MinIO (S3-compatible object storage) |
| `make pipeline-server` | Create the DataSciencePipelinesApplication, wait for Ready |
| `make mlflow` | (optional) Build + deploy a lightweight MLflow tracking server |
| `make build` | Build the training image via an OpenShift Build |
| `make train` | Run a distributed TrainJob directly (`MODE=cpu`\|`gpu`) |
| `make compile-pipeline` | Compile `pipeline/pipeline.py` -> `pipeline/pipeline.yaml` |
| `make pipeline` | Upload + run the AI Pipeline |
| `make pipeline-status` | Check the latest pipeline run |
| `make schedule` / `make schedule-status` / `make unschedule` | Manage the recurring run |
| `make validate` | PASS/FAIL verification of the whole demo (incl. a secret scan) |
| `make security-check` | Scan the working tree for tokens/secrets before pushing |
| `make test-negative` | Optional: confirm the demo fails clearly on bad inputs (invalid image, exit 1, too many GPUs) |
| `make status` | Full CLUSTER/TRAINING/PIPELINES/MLFLOW status table |
| `make cleanup` | Delete only this demo's namespace/resources |
| `make demo` | Run the whole sequence above (idempotent, optional stages `WARN` instead of failing) |
| `make demo-reset` | `make cleanup` then `make demo`, to prove reproducibility from zero |

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
- [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md)
