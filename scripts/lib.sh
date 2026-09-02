#!/usr/bin/env bash
# Shared defaults, helpers, and PASS/WARN/FAIL/OPTIONAL reporting used by every script in
# this repository. Source this file; do not execute it directly.
#
# Every variable below has a default and can be overridden with a plain environment
# variable, e.g.:
#   export NAMESPACE=my-demo
#   export TRAIN_NODES=4
# No script or manifest in this repository hardcodes a sandbox hostname, cluster API,
# username, token, namespace, route, GPU count, or storage class.

set -euo pipefail

# --------------------------------------------------------------------------------------
# Defaults (all overridable via environment variables)
# --------------------------------------------------------------------------------------
: "${NAMESPACE:=rhoai-training-demo}"
: "${PART_OF_LABEL:=rhoai-pytorch-trainer-demo}"
: "${MANAGED_BY_LABEL:=rhoai-pytorch-trainer-demo}"

: "${IMAGE_STREAM_NAME:=pytorch-trainer-demo}"
: "${IMAGE_TAG:=latest}"
: "${GIT_REF:=$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)}"
: "${GIT_REPO_URL:=$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." config --get remote.origin.url 2>/dev/null || echo "")}"
: "${SOURCE_SECRET_NAME:=}"
: "${GIT_SHA:=$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse HEAD 2>/dev/null || echo unknown)}"
GIT_SHA_SHORT="${GIT_SHA:0:12}"

: "${TRAINJOB_NAME:=pytorch-trainer-demo}"
: "${TRAIN_NODES:=2}"

# MODE=cpu|gpu is a convenience switch (`make train MODE=gpu`) that only takes effect if
# GPU_PER_NODE was NOT already explicitly set -- an explicit GPU_PER_NODE always wins, so
# `GPU_PER_NODE=2 MODE=gpu` still requests 2 GPUs/node, not the MODE=gpu default of 1.
: "${MODE:=}"
if [ -z "${GPU_PER_NODE:-}" ]; then
  case "${MODE}" in
    gpu) GPU_PER_NODE=1 ;;
    *) GPU_PER_NODE=0 ;;
  esac
fi
: "${TRAIN_CPU:=250m}"
: "${TRAIN_MEMORY:=768Mi}"
: "${TRAIN_EPOCHS:=5}"
: "${TRAIN_LR:=0.01}"
: "${TRAIN_BATCH_SIZE:=32}"

: "${MLFLOW_TRACKING_URI:=}"
: "${MLFLOW_EXPERIMENT_NAME:=rhoai-pytorch-trainer-demo}"
: "${USE_MLFLOW:=auto}"   # auto|true|false -- see detect_mlflow_uri()

: "${PIPELINE_NAME:=pytorch-trainer-demo-pipeline}"
: "${PIPELINE_SERVICE_ACCOUNT:=pipeline-trainjob-runner}"
: "${PIPELINE_RUN_TIMEOUT_SECONDS:=1800}"
: "${SCHEDULE_CRON:=0 2 * * *}"
: "${SCHEDULE_NAME:=pytorch-trainer-demo-nightly}"

: "${TRAINJOB_WAIT_TIMEOUT_SECONDS:=1200}"

# --------------------------------------------------------------------------------------
# Object storage (MinIO) / Pipeline Server / MLflow -- all namespace-local, all optional
# except where `make pipeline` needs the Pipeline Server.
# --------------------------------------------------------------------------------------
: "${MINIO_IMAGE:=quay.io/minio/minio:RELEASE.2024-01-16T16-07-38Z}"
: "${MINIO_MC_IMAGE:=quay.io/minio/mc:latest}"
: "${MINIO_STORAGE_SIZE:=5Gi}"
: "${PIPELINE_BUCKET:=mlpipeline}"
: "${MLFLOW_BUCKET:=mlflow}"
: "${CHECKPOINT_BUCKET:=checkpoints}"
: "${DSPA_NAME:=dspa}"
: "${DSPA_WAIT_TIMEOUT_SECONDS:=300}"
: "${MLFLOW_IMAGE_STREAM_NAME:=pytorch-trainer-demo-mlflow}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Prefer the project-local virtualenv (created by `make install`) so that scripts which
# need kfp/kubernetes/mlflow python packages (e.g. pipeline_client.py) don't silently fall
# back to a bare system `python3` that is missing those packages.
if [ -x "${REPO_ROOT}/.venv/bin/python" ]; then
  PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

# --------------------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------------------
if [ -t 1 ]; then
  C_RESET="\033[0m"; C_GREEN="\033[32m"; C_YELLOW="\033[33m"; C_RED="\033[31m"; C_BLUE="\033[34m"
else
  C_RESET=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_BLUE=""
fi

log_info() { echo -e "${C_BLUE}[info]${C_RESET} $*"; }
log_pass() { echo -e "${C_GREEN}PASS${C_RESET}     $*"; }
log_warn() { echo -e "${C_YELLOW}WARN${C_RESET}     $*"; }
log_fail() { echo -e "${C_RED}FAIL${C_RESET}     $*"; }
log_optional() { echo -e "${C_YELLOW}OPTIONAL${C_RESET} $*"; }
log_section() { echo -e "\n${C_BLUE}== $* ==${C_RESET}"; }

# --------------------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------------------
require_cmd() {
  command -v "$1" >/dev/null 2>&1
}

demo_label_selector() {
  echo "app.kubernetes.io/part-of=${PART_OF_LABEL}"
}

# Renders a manifest template through envsubst with only the variables it references,
# so unrelated `$` characters elsewhere are never mangled.
render_manifest() {
  local template_path="$1"
  # shellcheck disable=SC2016
  local vars
  vars=$(grep -oE '\$\{[A-Z0-9_]+\}' "$template_path" | sort -u | sed 's/[${}]//g' | sed 's/^/$/' | tr '\n' ' ')
  envsubst "$vars" < "$template_path"
}

gpu_resource_lines() {
  local gpu_count="${1:-0}"
  if [ "$gpu_count" -gt 0 ] 2>/dev/null; then
    echo "        nvidia.com/gpu: \"${gpu_count}\""
  else
    echo ""
  fi
}

mlflow_env_block() {
  local uri
  uri=$(detect_mlflow_uri)
  if [ -n "$uri" ]; then
    printf '      - name: MLFLOW_TRACKING_URI\n        value: "%s"\n' "${uri}"
  else
    echo ""
  fi
}

oc_login_check() {
  oc whoami >/dev/null 2>&1
}

# Resolves the MLflow tracking URI to actually use, honoring USE_MLFLOW:
#   USE_MLFLOW=false -> always empty (MLflow explicitly disabled for this run)
#   MLFLOW_TRACKING_URI already set -> use it as-is (explicit override wins)
#   USE_MLFLOW=auto|true -> auto-detect this demo's own MLflow Route (see scripts/mlflow.sh)
# Prints the URI (possibly empty) on stdout; never fails.
detect_mlflow_uri() {
  if [ "${USE_MLFLOW}" = "false" ]; then
    echo ""
    return 0
  fi
  if [ -n "${MLFLOW_TRACKING_URI}" ]; then
    echo "${MLFLOW_TRACKING_URI}"
    return 0
  fi
  local host
  host=$(oc get route mlflow -n "${NAMESPACE}" -o jsonpath='{.spec.host}' 2>/dev/null || true)
  if [ -n "$host" ]; then
    echo "https://${host}"
    return 0
  fi
  echo ""
}

# Emits the env entries that let a TrainJob pod upload its checkpoint to this demo's
# namespace-local MinIO instance, IF that instance actually exists in ${NAMESPACE}. Never
# hardcodes credentials -- they are always sourced from the 'minio-credentials' Secret via
# valueFrom.secretKeyRef, so the rendered manifest never contains a plaintext secret.
checkpoint_s3_env_block() {
  if ! oc get secret minio-credentials -n "${NAMESPACE}" >/dev/null 2>&1; then
    echo ""
    return 0
  fi
  cat <<EOF
      - name: CHECKPOINT_S3_BUCKET
        value: "${CHECKPOINT_BUCKET}"
      - name: CHECKPOINT_S3_PREFIX
        value: "${TRAINJOB_NAME}"
      - name: S3_ENDPOINT_URL
        value: "http://minio.${NAMESPACE}.svc.cluster.local:9000"
      - name: AWS_ACCESS_KEY_ID
        valueFrom:
          secretKeyRef:
            name: minio-credentials
            key: MINIO_ROOT_USER
      - name: AWS_SECRET_ACCESS_KEY
        valueFrom:
          secretKeyRef:
            name: minio-credentials
            key: MINIO_ROOT_PASSWORD
EOF
}
