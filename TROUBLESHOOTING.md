# Troubleshooting

## `make preflight` fails on "CRD missing: trainjobs.trainer.kubeflow.org"

The Kubeflow Trainer v2 component isn't enabled on this cluster. A cluster/platform
administrator needs to enable the `trainer` component in the `DataScienceCluster` (and
install the JobSet operator via OLM if it isn't already present). This demo cannot enable
cluster-wide components itself -- that is intentionally outside the scope of a
namespace-scoped demo.

## `make preflight` shows "ClusterTrainingRuntime 'torch-distributed' not found"

Check what's actually available:

```bash
oc get clustertrainingruntime
```

If a differently-named PyTorch-compatible runtime exists, override the reference used by
`manifests/trainjob-template.yaml` and `pipeline/pipeline.py` (search for
`torch-distributed` in both) or ask your platform administrator to install the standard
one that ships with OpenShift AI.

## `make train` reports "Namespace does not exist yet"

Run `make bootstrap` first. `make train` deliberately refuses to create the namespace
itself, to keep bootstrap and training as separate, individually-idempotent steps.

## `make build` fails with a Git authentication error

`GIT_REPO_URL` is auto-detected from `git remote get-url origin`. If this repository is
private, the OpenShift Build needs a source secret:

```bash
oc create secret generic git-source-auth \
  --type=kubernetes.io/basic-auth \
  --from-literal=username=<your-username> \
  --from-literal=password=<a-personal-access-token> \
  -n "$NAMESPACE"
oc annotate secret git-source-auth "build.openshift.io/source-secret-match-uri-1=https://github.com/*" -n "$NAMESPACE"
export SOURCE_SECRET_NAME=git-source-auth
make build
```

Never commit that token anywhere in this repository.

## `TrainJob` stays `Pending` / workers never schedule

Almost always a resource request the cluster can't satisfy:

```bash
oc describe trainjob <name> -n "$NAMESPACE"
oc get pods -n "$NAMESPACE"
oc describe pod <pod> -n "$NAMESPACE"   # look at Events at the bottom
```

Common causes: `GPU_PER_NODE` higher than what's actually allocatable (see
`make preflight`'s GPU section), or `TRAIN_CPU`/`TRAIN_MEMORY` larger than any node's
allocatable capacity. Lower them and re-run `make train` (it deletes and recreates a
finished/failed TrainJob automatically, but will refuse to touch a still-Pending one --
delete it manually first: `oc delete trainjob <name> -n "$NAMESPACE"`).

## `make pipeline` / `scripts/deploy-pipeline.sh` says "No usable Data Science Pipelines server found"

Run, in order:

```bash
make storage           # namespace-local MinIO (S3-compatible object storage)
make pipeline-server   # DataSciencePipelinesApplication, waits for Ready
make preflight          # confirm OBJECT STORAGE / PIPELINE SERVER / DSPA STATUS all PASS
```

If you already have an external Pipeline Server / S3 bucket you'd rather use instead of
this demo's own MinIO, skip `make storage` and create your own
`DataSciencePipelinesApplication` (via the Dashboard or your own CR) referencing it, then
re-run `make preflight` to confirm it's detected before `make pipeline`.

## `scripts/pipeline_client.py` fails with an SSL/certificate error

The Data Science Pipelines API route is usually served with the cluster's default ingress
certificate, which may not be in your local trust store. Provide a CA bundle:

```bash
export PIPELINE_SSL_CA_CERT=/path/to/your-cluster-ca.pem
```

You can typically extract the relevant CA from the `openshift-ingress-operator` namespace
or your organization's certificate management process; this repository does not assume
any particular certificate authority.

## `scripts/pipeline_client.py` can't find the Pipeline Server route

By default it looks for a route named `ds-pipeline-<something>` (excluding the UI route)
in `$NAMESPACE`. If your DSPA uses a different naming convention, override it directly:

```bash
export PIPELINE_ROUTE_HOST=https://<your-actual-route-host>
```

## "Forbidden" errors from the pipeline's `distributed-training` step

Confirm the RBAC from `manifests/rbac.yaml` was applied (`make bootstrap` does this) and
that the run was actually started with `service_account=pipeline-trainjob-runner`. If you
started a run manually from the OpenShift AI Dashboard instead of via
`scripts/pipeline_client.py`, the Dashboard's default run configuration may use a
different `ServiceAccount` (often the DSPA's own `pipeline-runner-<dspa-name>`) -- in that
case, either always trigger runs via `make pipeline`, or grant the same `Role` to that
other `ServiceAccount` as well:

```bash
oc create rolebinding pipeline-runner-trainjob-operator \
  --role=trainjob-operator \
  --serviceaccount="$NAMESPACE:pipeline-runner-<dspa-name>" \
  -n "$NAMESPACE"
```

## `make cleanup` refuses to delete the namespace

This is intentional: it only deletes a namespace labeled
`app.kubernetes.io/part-of=rhoai-pytorch-trainer-demo`. If you pointed `NAMESPACE` at an
existing namespace this demo didn't create, cleanup will not touch it. Use a fresh
namespace name, or `make bootstrap` in it first.

## `make storage` fails / MinIO never becomes Ready

Check the pod directly:

```bash
oc get pods -l app.kubernetes.io/name=minio -n "$NAMESPACE"
oc describe pod -l app.kubernetes.io/name=minio -n "$NAMESPACE"
```

The most common cause on a constrained sandbox is no default `StorageClass` (the PVC stays
`Pending`) -- check with `oc get pvc minio-data -n "$NAMESPACE"` and
`oc describe pvc minio-data -n "$NAMESPACE"`. If your cluster requires a specific
`StorageClass`, add it to `manifests/storage.yaml`'s PVC spec.

## `make pipeline-server` times out waiting for Ready

```bash
oc get datasciencepipelinesapplication -n "$NAMESPACE" -o yaml
oc get pods -n "$NAMESPACE" -l app.kubernetes.io/component=data-science-pipelines
```

Common causes: the MinIO Service/bucket isn't reachable from the DSPA's API server pod
(re-run `make storage` and confirm the `minio-init-buckets` Job completed), or MariaDB
(deployed automatically by the Data Science Pipelines Operator itself) is stuck `Pending`
on storage the same way MinIO can be -- see the PVC troubleshooting above.

**A very common cause on a shared/busy sandbox: the cluster itself has no free CPU**, even
though this demo's own components (`manifests/dspa.yaml`) request as little as 20-50m CPU
each. `make preflight`'s "Cluster CPU/memory headroom" section reports this explicitly.
Confirm it yourself:

```bash
oc describe node <node-name> | grep -A6 "Allocated resources"
oc get pods -n "${NAMESPACE}" | grep -iE "mariadb|ds-pipeline"
oc describe pod -n "${NAMESPACE}" -l app=mariadb-${DSPA_NAME:-dspa}   # look at Events: "Insufficient cpu"
```

If `Allocated resources` shows CPU at or near 100% and the MariaDB/API-server pods are
stuck `Pending` with `FailedScheduling: Insufficient cpu`, this is **cluster capacity
consumed by other tenants/namespaces**, not a defect in this demo -- and this demo will
**never** free capacity by scaling down or deleting a workload it doesn't own. Options,
in order of preference:
1. Wait and retry (`make pipeline-server`) once other workloads on the sandbox finish or
   are cleaned up by their owners.
2. Ask whoever administers the shared sandbox for a small amount of headroom, or use a
   less busy sandbox / a dedicated one.
3. If you administer the cluster yourself, you already know how to free or add capacity;
   this repo intentionally does not attempt it.

Also note that if `mc: <ERROR> Unable to save new mc config. mkdir /.mc: permission
denied.` shows up in `oc logs job/minio-init-buckets -n "${NAMESPACE}"`, that's a
different, already-fixed issue: `manifests/storage-init-job.yaml` now sets `HOME=/tmp` in
the `mc` container so it can write its config under OpenShift's restricted (arbitrary
non-root UID) SCC.

## `make mlflow` — MLflow deployment stays `Pending` (never `CrashLoopBackOff`)

`oc get pods -l app.kubernetes.io/name=mlflow -n "${NAMESPACE}"` shows `Pending` with a
`FailedScheduling` event mentioning `Insufficient cpu`. This is the exact same
shared-sandbox capacity issue as `make pipeline-server` above (see "Cluster CPU/memory
headroom" in `make preflight`) -- MLflow's own request is already as low as 20m CPU, so if
even that can't schedule, the cluster itself has no free CPU right now, cluster-wide, from
other tenants. `make preflight` / `make status` report this honestly as `WARN`/`NOT READY`
rather than claiming Ready because a Route exists (a Route existing never means the pod
behind it is actually up).

## `make mlflow` — MLflow pod is `CrashLoopBackOff`

```bash
oc logs deployment/mlflow -n "$NAMESPACE"
```

Almost always: `minio-credentials` doesn't exist yet (`make storage` first) or the
`mlflow` bucket doesn't exist (re-run `make storage`, which is idempotent and re-creates
buckets if missing).

## Pipeline run fails at `evaluate-model` with a checkpoint download error

This means object storage WAS configured for the run (`checkpoint_bucket` was non-empty)
but the checkpoint object itself is missing from MinIO -- almost always because
`training/train.py` could not reach MinIO from inside the TrainJob pod. Check:

```bash
oc logs <trainjob-worker-pod> -n "$NAMESPACE" | grep -i "s3 checkpoint"
```

If it says "S3 checkpoint upload disabled", `CHECKPOINT_S3_BUCKET` wasn't injected into the
TrainJob's env -- confirm `make storage` ran before this pipeline run (the pipeline only
attaches S3 checkpoint env vars when it can see `checkpoint_bucket` was passed as a
non-empty run parameter, which `scripts/pipeline_client.py` only does when
`minio-credentials` exists in the namespace at run-submission time).

## `make security-check` / `make validate` flags a false positive

Read the flagged file: if it is genuinely a template/placeholder (not a real secret),
narrow the regex in `scripts/security-check.sh` rather than disabling the check entirely.
If the file should never be tracked by git at all, add it to `.gitignore` instead.

## Running everything locally without a cluster

`training/train.py` and `training/evaluate.py` work as plain local Python scripts with no
Kubernetes/OpenShift dependency at all:

```bash
make install
.venv/bin/python training/train.py --epochs 3 --output-dir /tmp/out
.venv/bin/python training/evaluate.py --model-dir /tmp/out
```

You can even simulate 2 local distributed processes on a laptop with `torchrun`:

```bash
.venv/bin/python -m torch.distributed.run --nproc-per-node=2 training/train.py --epochs 3 --output-dir /tmp/out
```
