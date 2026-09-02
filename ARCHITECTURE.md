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

`evaluate-model` reads the *real* per-pod logs of that TrainJob (looking for multiple
distinct `rank=` values and the final loss) rather than reading a shared checkpoint file
from disk. This is a deliberate reproducibility trade-off: sharing a PVC between TrainJob
pods (whose container/job names are defined by the platform-provided
`ClusterTrainingRuntime`, and can differ across OpenShift AI versions) and pipeline task
pods would make the pipeline's correctness depend on an internal implementation detail of
the runtime template. Reading pod logs through the Kubernetes API only depends on the
stable, documented `TrainJob`/pod contract, so the same pipeline definition works
unmodified across sandboxes.

## Why the pipeline's `ServiceAccount` has minimal RBAC

`manifests/rbac.yaml` creates one namespace-scoped `ServiceAccount`
(`pipeline-trainjob-runner`) and one `Role` granting only:

- full control of `trainjobs.trainer.kubeflow.org` (needed to create/watch/delete the
  TrainJob the pipeline manages),
- read-only on `clustertrainingruntimes`/`trainingruntimes` (to validate `runtimeRef`
  before creating a job),
- read-only on `jobsets.jobset.x-k8s.io` (the underlying primitive Kubeflow Trainer uses),
- read-only on `pods`/`pods/log` (to observe workers and extract training metrics).

No `cluster-admin`, no cluster-scoped `RoleBinding`, no access outside the demo namespace.
`scripts/deploy-pipeline.sh` passes `service_account=pipeline-trainjob-runner` explicitly
when starting or scheduling a run, so every step in a given run shares this identity.

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
