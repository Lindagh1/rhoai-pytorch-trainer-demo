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

This demo does not create a Pipeline Server automatically (it needs S3 credentials that
must never be committed to Git). See README.md "Pipeline Server storage" for the manual,
one-time steps to provision one, then re-run `make preflight` to confirm it's detected.

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
