#!/usr/bin/env bash
# Idempotently builds (from THIS repository, via an OpenShift Build -- same pattern as
# training/train.py's image) and deploys a small, namespace-local MLflow Tracking Server,
# backed by the MinIO instance created by `make storage`.
#
# This is entirely optional: nothing else in this demo requires it. If you already have a
# usable MLflow instance elsewhere, just export MLFLOW_TRACKING_URI and skip this script
# (see scripts/lib.sh detect_mlflow_uri / README.md "MLflow").
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source scripts/lib.sh

if ! oc_login_check; then
  log_fail "oc is not logged in. Run 'oc login ...' first."
  exit 1
fi

if ! oc get secret minio-credentials -n "${NAMESPACE}" >/dev/null 2>&1; then
  log_fail "Secret 'minio-credentials' not found in '${NAMESPACE}'. Run 'make storage' first (MLflow artifacts need object storage)."
  exit 1
fi

if [ -z "${GIT_REPO_URL}" ]; then
  log_fail "GIT_REPO_URL could not be auto-detected (no 'origin' remote configured) and is not set."
  exit 1
fi

if [ -n "${SOURCE_SECRET_NAME}" ]; then
  SOURCE_SECRET_BLOCK="    sourceSecret:
      name: ${SOURCE_SECRET_NAME}"
else
  SOURCE_SECRET_BLOCK=""
fi
export NAMESPACE MLFLOW_IMAGE_STREAM_NAME IMAGE_TAG GIT_REPO_URL GIT_REF SOURCE_SECRET_BLOCK

log_section "Building the MLflow image from ${GIT_REPO_URL}@${GIT_REF} (OpenShift Build)"
render_manifest manifests/buildconfig-mlflow.yaml | oc apply -f -
BUILD_NAME=$(oc start-build "${MLFLOW_IMAGE_STREAM_NAME}" -n "${NAMESPACE}" -o name)
log_info "Started ${BUILD_NAME}"
oc logs -f "${BUILD_NAME}" -n "${NAMESPACE}" || true

PHASE=""
for _ in $(seq 1 30); do
  PHASE=$(oc get "${BUILD_NAME}" -n "${NAMESPACE}" -o jsonpath='{.status.phase}' 2>/dev/null || echo "")
  case "$PHASE" in
    Complete|Failed|Error|Cancelled) break ;;
  esac
  sleep 1
done
if [ "$PHASE" != "Complete" ]; then
  log_fail "MLflow image build ended in phase '${PHASE}'. See: oc logs ${BUILD_NAME} -n ${NAMESPACE}"
  exit 1
fi
log_pass "MLflow image built"

MLFLOW_IMAGE="image-registry.openshift-image-registry.svc:5000/${NAMESPACE}/${MLFLOW_IMAGE_STREAM_NAME}:${IMAGE_TAG}"
export NAMESPACE MLFLOW_IMAGE MLFLOW_BUCKET

log_section "Deploying MLflow (PVC + Deployment + Service + Route)"
render_manifest manifests/mlflow.yaml | oc apply -f -

log_section "Waiting for MLflow to become Ready"
if ! oc rollout status deployment/mlflow -n "${NAMESPACE}" --timeout=180s; then
  log_fail "MLflow deployment did not become Ready within 180s. Check: oc describe pod -l app.kubernetes.io/name=mlflow -n ${NAMESPACE}"
  exit 1
fi

MLFLOW_HOST=$(oc get route mlflow -n "${NAMESPACE}" -o jsonpath='{.spec.host}' 2>/dev/null || true)
if [ -z "$MLFLOW_HOST" ]; then
  log_fail "MLflow Route not found -- cannot determine its URI"
  exit 1
fi

log_pass "MLflow is Ready at https://${MLFLOW_HOST}"
echo "export MLFLOW_TRACKING_URI=https://${MLFLOW_HOST}"
echo "(scripts/lib.sh auto-detects this Route if MLFLOW_TRACKING_URI is left unset -- see detect_mlflow_uri)"
