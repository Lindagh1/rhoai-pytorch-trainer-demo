# Architecture

This document explains the technical choices behind this demo and, more importantly,
*why* -- because several of these choices look like they add complexity for no reason
until you consider the constraint that drove them: **this repository must rebuild the
entire demo on a sandbox it has never seen, with nothing manual and undocumented.**

## Why the notebook is not used for training in production

A notebook is a great environment for *exploring* a problem: you can inspect intermediate
tensors, replot a loss curve, and change one line without re-running everything. All of
that interactivity is exactly what makes a notebook a poor *deployment artifact*:

- It has no meaningful versioning of "what actually ran" (cell execution order can drift
  from the file's cell order).
- It cannot be trivially parameterized for `RANK`/`WORLD_SIZE` across N pods.
- It cannot be baked into an immutable container image the way a `.py` file can.

So `notebooks/exploration.ipynb` is kept **only** as the exploration artifact, and its last
section explicitly imports from `training/train.py` to make the hand-off visible: the
modeling logic doesn't get rewritten when it moves from notebook to script, it gets
*extracted*.

## Why the code is contained in an image, not a ConfigMap

An earlier, simpler version of this kind of demo might put `train.py` in a `ConfigMap` and
mount it into a generic PyTorch image. That was deliberately rejected here as the primary
architecture, because:

- A `ConfigMap` is not versioned the way a container image digest is; two people applying
  "the same" `ConfigMap` at different times can silently end up with different code.
- It decouples "what code ran" from "what image ran it" -- exactly the traceability this
  demo is trying to demonstrate for an ML platform.
- It sidesteps the actual, common enterprise pattern: **build a training image from your
  Git repository the same way you'd build any other service.**

Instead, `training/Containerfile` builds `training/train.py` and `training/evaluate.py`
directly into an image via an OpenShift `BuildConfig` (`manifests/buildconfig.yaml`) that
pulls straight from this Git repository and pushes to the internal OpenShift image
registry. No private/external registry credentials are required:

```
GitHub  ->  OpenShift Build  ->  training image (internal registry)  ->  TrainJob
```

## Why the `TrainJob` is not the script

`spec.trainer.command`/`args` in a `TrainJob` tell Kubeflow Trainer *how to invoke* the
training container; they are not the training logic itself. Keeping the two separate is
what makes it possible to:

- change the image (a new dependency, a new PyTorch version) without touching the
  `TrainJob` shape, and
- change `numNodes`/`resourcesPerNode` (scale up for a bigger GPU node pool) without
  touching a single line of Python.

The `TrainJob` is a *description of a workload*: which image, which command, how many
nodes, how much CPU/memory/GPU per node. `training/train.py` is the workload.

## Why the `TrainJob` references a runtime instead of embedding one

```yaml
spec:
  runtimeRef:
    apiGroup: trainer.kubeflow.org
    kind: ClusterTrainingRuntime
    name: torch-distributed
```

`ClusterTrainingRuntime` is a cluster-scoped resource **provided by OpenShift AI itself**
(enabled via the `trainer` component of the `DataScienceCluster`), not something this demo
creates. It encapsulates `torchrun` invocation, `MASTER_ADDR`/`MASTER_PORT` wiring, and
node coordination -- exactly the platform-owned complexity a data scientist should not have
to reimplement per job. `manifests/trainjob-template.yaml` only supplies what's specific to
*this* workload (image, command, node count, resources); it reads, but never modifies or
recreates, the `ClusterTrainingRuntime`.

## Why the pipeline creates the `TrainJob` itself, not a mock

The `distributed-training` step of `pipeline/pipeline.py` calls the Kubernetes API directly
(`kubernetes.client.CustomObjectsApi`, using the pipeline task's own in-cluster
`ServiceAccount` token) to create a real `TrainJob`, then polls `status.conditions` until it
reaches `Complete` or `Failed` before letting `evaluate-model` run. There is no Python
script that "pretends" to train inside the pipeline step -- the actual distributed workload
runs exactly the way `make train` runs it directly, just triggered from within the pipeline
instead of from a script on your laptop.

## Why every pipeline run's TrainJob has a unique name

`trainjob_name` is only a *base* name (default `rhoai-demo-train`). The
`distributed-training` component appends a timestamp and a short random suffix at task
**runtime** (plain Python `time`/`uuid`, not a KFP backend placeholder) before creating the
`TrainJob`, e.g. `rhoai-demo-train-260902143012-a1b2c`. This is deliberately NOT done with
one of KFP's `dsl.PIPELINE_JOB_ID_PLACEHOLDER`-style backend substitutions: that feature is
recent enough in the Kubeflow Pipelines backend that relying on it would silently break on
older Data Science Pipelines versions (the raw, unresolved placeholder string is not a
valid Kubernetes object name and the TrainJob creation would fail outright). Generating the
suffix in plain Python inside the component works identically on any KFP v2 backend and
guarantees two concurrent pipeline runs (or a recurring/scheduled run firing while a manual
run is still in flight) never collide on the same `TrainJob` name.

## Checkpoint persistence: worker -> checkpoint -> object storage -> evaluate

`training/train.py` always writes `model.pt` and `training_metrics.json` to
`TRAIN_OUTPUT_DIR` on the training pod's local (ephemeral) filesystem -- that part is
unconditional and doesn't require any extra infrastructure (this is what `make train`,
which never touches object storage, already relies on for its own summary).

When the namespace-local MinIO instance is present (`make storage`), the pipeline's
`distributed-training` step ALSO injects `CHECKPOINT_S3_BUCKET` /
`CHECKPOINT_S3_PREFIX=<unique-trainjob-name>` / `S3_ENDPOINT_URL` and MinIO credentials
(via `valueFrom.secretKeyRef`, never a literal value) into the TrainJob's env. Rank 0 of
`training/train.py` then uploads both files to
`s3://<checkpoint-bucket>/<unique-trainjob-name>/` using `boto3` (see `CheckpointStore` in
`training/train.py`) as a best-effort, non-fatal step -- training still succeeds even if
this upload fails, the same way MLflow logging degrades.

`evaluate-model` downloads those two objects, and:

1. loads `training_metrics.json` for the real `final_val_loss` the training run recorded
   (not a value re-derived from logs), and
2. loads `model.pt`'s state dict into a `TinyRegressor` (the same architecture as
   `training/train.py`, necessarily duplicated inline since this pipeline step runs in its
   own container without the repository checked out) and runs a genuine forward pass on a
   freshly generated held-out test set, producing `test_mse`.

If no object storage was configured for a given run (e.g. `make storage` was never run),
`evaluate-model` falls back to reading the TrainJob's pod logs to confirm multiple ranks
executed and to extract the final loss from log lines -- and reports
`"method": "logs-only"` in its output so it is always clear which evidence a given
evaluation is actually based on. It never claims to have loaded a checkpoint it did not
load.

```
Worker (rank 0) --writes--> model.pt, training_metrics.json (local disk)
       |
       +--uploads (boto3, best-effort)--> s3://checkpoints/<trainjob-name>/ (MinIO)
                                                      |
                                        evaluate-model downloads + torch.load()s
                                                      |
                                        genuine forward pass -> test_mse
```

## Why object storage is a namespace-local MinIO, not an external dependency

Data Science Pipelines (and this demo's own checkpoint persistence) both need an
S3-compatible bucket. Requiring the user to bring their own external bucket/credentials
would break the "clone and `make demo`" reproducibility goal on a sandbox that has none.
`scripts/storage.sh` deploys a small, single-replica MinIO `Deployment` + `PersistentVolumeClaim`
+ `Service` entirely inside this demo's own namespace, with credentials generated randomly
at apply time and stored ONLY in a `Secret` in that namespace (never in Git, never
printed). It is explicitly not a production storage recommendation; it exists purely so a
brand new sandbox with nothing but a default `StorageClass` can reach a fully working
Pipeline Server without a human provisioning an external bucket first. If you already have
a real S3-compatible bucket/DSPA, skip `make storage`/`make pipeline-server` entirely and
point `MLFLOW_TRACKING_URI`/your own DSPA at it instead.

## Why the Pipeline Server is a DataSciencePipelinesApplication this repo creates

`scripts/pipeline-server.sh` applies `manifests/dspa.yaml` (one
`DataSciencePipelinesApplication` per namespace, which is what the OpenShift AI Dashboard's
"Pipelines" tab looks for) referencing the MinIO `Service` DNS name and the
`minio-credentials` Secret as `objectStorage.externalStorage`, and waits for
`status.conditions[type=Ready]=True` before returning success. `make pipeline` is never run
against a DSPA that isn't actually Ready -- this script fails loudly instead.

## Why MLflow can be a real, deployed instance -- not just a URL you bring

`scripts/mlflow.sh` builds a small MLflow Tracking Server image from THIS repository
(`mlflow/Containerfile`, same OpenShift-Build-from-Git pattern as the training image) and
deploys it with a SQLite backend store and the namespace-local MinIO bucket `mlflow` as its
artifact root. This is optional and separate from `make demo` (which never fails if MLflow
isn't deployed) -- but when you run `make mlflow`, every subsequent `make train` / `make
pipeline` run auto-detects it (via its `Route`, see `scripts/lib.sh`'s `detect_mlflow_uri`)
without needing to export `MLFLOW_TRACKING_URI` by hand. `training/train.py`,
`training/evaluate.py`, and the pipeline's `evaluate-model` step tag every MLflow run with
`trainjob_name` and `git_commit`, so a run in the MLflow UI is always traceable back to the
exact `TrainJob` and Git commit that produced it.

`evaluate-model` reads the *real* per-pod logs of that TrainJob (looking for multiple
distinct `rank=` values) as one source of evidence of distributed execution, in addition
to the checkpoint-based evaluation described above.

## Why the pipeline's `ServiceAccount` has minimal RBAC

`manifests/rbac.yaml` creates one namespace-scoped `ServiceAccount`
(`pipeline-trainjob-runner`) and one `Role` granting only:

- full control of `trainjobs.trainer.kubeflow.org` (needed to create/watch/delete the
  TrainJob the pipeline manages),
- read-only on `clustertrainingruntimes`/`trainingruntimes` (to validate `runtimeRef`
  before creating a job),
- read-only on `jobsets.jobset.x-k8s.io` (the underlying primitive Kubeflow Trainer uses),
- read-only on `pods`/`pods/log` (to observe workers and extract training metrics),
- `get` on exactly one named Secret, `minio-credentials` (via `resourceNames:
  ["minio-credentials"]`, not a wildcard on all Secrets) -- needed only so `evaluate-model`
  can obtain the MinIO access/secret key to download the real checkpoint (see "Checkpoint
  persistence" above). No other Secret in the namespace is readable by this Role.

No `cluster-admin`, no cluster-scoped `RoleBinding`, no access outside the demo namespace.
`scripts/deploy-pipeline.sh` passes `service_account=pipeline-trainjob-runner` explicitly
when starting or scheduling a run, so every step in a given run shares this identity.

## Git SHA / image traceability

Every training image built by `scripts/build-training-image.sh` is annotated on its
`ImageStreamTag` with `demo.git.sha` (read straight from the OpenShift `Build` object's own
`.spec.revision.git.commit`, not re-derived) and `demo.git.repo`. Every `TrainJob` --
whether created directly by `make train` or by the pipeline's `distributed-training` step
-- carries the same commit as a `demo.git.sha` label (truncated to 12 chars, to stay within
Kubernetes label length/charset limits) and a full-length `demo.git.sha` annotation, plus a
`GIT_SHA` environment variable that reaches `training/train.py`, which in turn writes it
into `training_metrics.json` and (if MLflow is active) as an MLflow run tag `git_commit`.
The result: given any MLflow run, `TrainJob`, or training image, you can always answer
"which exact commit produced this?" without guessing.

## Why MLflow is complementary, not an orchestrator

MLflow's Tracking Server is good at one thing: recording parameters/metrics/artifacts for
comparison across runs. It does not create Kubernetes resources, does not know about
`TrainJob`s, and is not what waits for training to finish -- Kubeflow Trainer and the AI
Pipeline already do that. `training/train.py`, `training/evaluate.py`, and the pipeline's
`evaluate-model` step all detect `MLFLOW_TRACKING_URI` and log to it *if it's set*; if it
isn't, or the server is unreachable, they degrade to printing the same information to
stdout. MLflow augments observability; it is never load-bearing for orchestration.

## Why GitHub is the source of truth

Everything that determines what the demo does -- the training code, the container
recipe, the Kubernetes manifests, the pipeline definition, the automation scripts -- lives
in this repository and nowhere else. The one sandbox used to validate all of this is
explicitly temporary and disposable:

- `scripts/bootstrap.sh` only ever calls `oc apply` on manifests rendered from files in
  this repo.
- `manifests/buildconfig.yaml` builds directly from this repo's own `git remote`, not from
  a snapshot copied onto the cluster.
- No script writes back to the cluster anything that isn't already expressed as a file
  here.

If the current sandbox disappears, cloning this repository onto a new one and running
`make demo` reproduces the same result -- that is the actual acceptance criterion for this
project, not "the demo works on the sandbox I built it on."
