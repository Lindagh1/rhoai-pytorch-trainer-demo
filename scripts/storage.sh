#!/usr/bin/env bash
# Idempotently deploys a namespace-local MinIO instance to serve as this demo's
# S3-compatible object storage (used by the Pipeline Server and, optionally, MLflow and
# TrainJob checkpoint uploads). Generates random credentials on first run and stores them
# ONLY in a Kubernetes Secret in this namespace -- never in Git, never printed to stdout.
#
# Re-running this script never regenerates existing credentials (so re-running `make
# storage` never breaks an already-configured DataSciencePipelinesApplication that
# references them), and never fails if the buckets already exist.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source scripts/lib.sh

if ! oc_login_check; then
  log_fail "oc is not logged in. Run 'oc login ...' first."
  exit 1
fi

if ! oc get namespace "${NAMESPACE}" >/dev/null 2>&1; then
  log_fail "Namespace '${NAMESPACE}' does not exist yet. Run 'make bootstrap' first."
  exit 1
fi

log_section "MinIO credentials (namespace='${NAMESPACE}')"
if oc get secret minio-credentials -n "${NAMESPACE}" >/dev/null 2>&1; then
  log_pass "Secret 'minio-credentials' already exists -- reusing it (credentials are never rotated by this script)"
else
  ROOT_USER="minio-$(openssl rand -hex 4 2>/dev/null || echo demo1234)"
  ROOT_PASSWORD="$(openssl rand -hex 16 2>/dev/null || date +%s%N)"
  oc create secret generic minio-credentials \
    --from-literal=MINIO_ROOT_USER="${ROOT_USER}" \
    --from-literal=MINIO_ROOT_PASSWORD="${ROOT_PASSWORD}" \
    -n "${NAMESPACE}"
  oc label secret minio-credentials -n "${NAMESPACE}" \
    app.kubernetes.io/part-of=rhoai-pytorch-trainer-demo \
    app.kubernetes.io/managed-by=rhoai-pytorch-trainer-demo \
    app.kubernetes.io/component=storage --overwrite >/dev/null
  log_pass "Generated a new random 'minio-credentials' Secret (never printed, never committed)"
fi

log_section "Deploying MinIO (PVC + Deployment + Service + Route)"
export NAMESPACE MINIO_STORAGE_SIZE MINIO_IMAGE
render_manifest manifests/storage.yaml | oc apply -f -

log_section "Waiting for MinIO to become Ready"
if ! oc rollout status deployment/minio -n "${NAMESPACE}" --timeout=180s; then
  log_fail "MinIO deployment did not become Ready within 180s. Check: oc describe pod -l app.kubernetes.io/name=minio -n ${NAMESPACE}"
  exit 1
fi
log_pass "MinIO is Ready"

log_section "Creating required buckets (idempotent)"
oc delete job minio-init-buckets -n "${NAMESPACE}" --ignore-not-found --wait=true >/dev/null 2>&1 || true
export PIPELINE_BUCKET MLFLOW_BUCKET CHECKPOINT_BUCKET MINIO_MC_IMAGE
render_manifest manifests/storage-init-job.yaml | oc apply -f -
if ! oc wait --for=condition=complete --timeout=120s job/minio-init-buckets -n "${NAMESPACE}" >/dev/null 2>&1; then
  log_fail "Bucket creation Job did not complete. Logs:"
  oc logs job/minio-init-buckets -n "${NAMESPACE}" 2>/dev/null || true
  exit 1
fi
log_pass "Buckets ready: ${PIPELINE_BUCKET}, ${MLFLOW_BUCKET}, ${CHECKPOINT_BUCKET}"

echo ""
echo "Object storage is ready. Next steps:"
echo "  make pipeline-server   # create the DataSciencePipelinesApplication backed by this MinIO"
echo "  make mlflow            # (optional) deploy a lightweight MLflow tracking server, also backed by this MinIO"
