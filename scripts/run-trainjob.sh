#!/usr/bin/env bash
# Renders manifests/trainjob-template.yaml and runs a real distributed PyTorch TrainJob
# through Kubeflow Trainer, waits for it to finish, and prints the per-rank training logs.
#
# Idempotent: if a TrainJob with the same name already exists and has finished (Complete
# or Failed), it is deleted and recreated so you can re-run `make train` repeatedly. If it
# is still running, this script attaches to it instead of creating a duplicate.
#
# Nothing here hardcodes node count or GPU count -- override with e.g.:
#   TRAIN_NODES=3 GPU_PER_NODE=1 make train
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

TRAIN_IMAGE="${TRAIN_IMAGE:-image-registry.openshift-image-registry.svc:5000/${NAMESPACE}/${IMAGE_STREAM_NAME}:${IMAGE_TAG}}"

log_section "TrainJob configuration"
echo "namespace:        ${NAMESPACE}"
echo "trainjob name:    ${TRAINJOB_NAME}"
echo "image:            ${TRAIN_IMAGE}"
echo "nodes:            ${TRAIN_NODES}"
echo "gpu per node:     ${GPU_PER_NODE}"
echo "cpu/mem per node: ${TRAIN_CPU} / ${TRAIN_MEMORY}"
echo "epochs:           ${TRAIN_EPOCHS}"

EXISTING_PHASE=""
if oc get trainjob "${TRAINJOB_NAME}" -n "${NAMESPACE}" >/dev/null 2>&1; then
  EXISTING_COMPLETE=$(oc get trainjob "${TRAINJOB_NAME}" -n "${NAMESPACE}" -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' 2>/dev/null || true)
  EXISTING_FAILED=$(oc get trainjob "${TRAINJOB_NAME}" -n "${NAMESPACE}" -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null || true)
  if [ "$EXISTING_COMPLETE" = "True" ] || [ "$EXISTING_FAILED" = "True" ]; then
    log_info "Existing TrainJob '${TRAINJOB_NAME}' already finished -- deleting it before re-creating (idempotent re-run)"
    oc delete trainjob "${TRAINJOB_NAME}" -n "${NAMESPACE}" --wait=true
  else
    log_warn "TrainJob '${TRAINJOB_NAME}' already exists and is still running -- attaching instead of creating a new one"
    EXISTING_PHASE="running"
  fi
fi

if [ "$EXISTING_PHASE" != "running" ]; then
  GPU_REQUEST_LINE=$(gpu_resource_lines "${GPU_PER_NODE}")
  GPU_LIMIT_LINE=$(gpu_resource_lines "${GPU_PER_NODE}")
  MLFLOW_TRACKING_URI_ENV=$(mlflow_env_block)
  export NAMESPACE TRAINJOB_NAME TRAIN_IMAGE TRAIN_NODES TRAIN_CPU TRAIN_MEMORY TRAIN_EPOCHS \
    TRAIN_LR TRAIN_BATCH_SIZE MLFLOW_EXPERIMENT_NAME GPU_REQUEST_LINE GPU_LIMIT_LINE MLFLOW_TRACKING_URI_ENV

  log_section "Creating TrainJob"
  render_manifest manifests/trainjob-template.yaml | oc apply -f -
fi

log_section "Waiting for TrainJob to complete (timeout: ${TRAINJOB_WAIT_TIMEOUT_SECONDS}s)"
DEADLINE=$((SECONDS + TRAINJOB_WAIT_TIMEOUT_SECONDS))
STATUS="Unknown"
while [ $SECONDS -lt "$DEADLINE" ]; do
  COMPLETE=$(oc get trainjob "${TRAINJOB_NAME}" -n "${NAMESPACE}" -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' 2>/dev/null || true)
  FAILED=$(oc get trainjob "${TRAINJOB_NAME}" -n "${NAMESPACE}" -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null || true)
  if [ "$COMPLETE" = "True" ]; then STATUS="Complete"; break; fi
  if [ "$FAILED" = "True" ]; then STATUS="Failed"; break; fi
  RUNNING_PODS=$(oc get pods -n "${NAMESPACE}" --no-headers 2>/dev/null | awk -v p="${TRAINJOB_NAME}-" '$1 ~ "^"p {print $1, $3}' || true)
  echo "  [$((DEADLINE - SECONDS))s left] pods: ${RUNNING_PODS:-none yet}"
  sleep 10
done

log_section "TrainJob workers"
oc get pods -n "${NAMESPACE}" --no-headers 2>/dev/null | awk -v p="${TRAINJOB_NAME}-" '$1 ~ "^"p {print}' || true

log_section "Per-rank training logs"
POD_NAMES=$(oc get pods -n "${NAMESPACE}" --no-headers 2>/dev/null | awk -v p="${TRAINJOB_NAME}-" '$1 ~ "^"p {print $1}' || true)
if [ -z "$POD_NAMES" ]; then
  log_warn "No pods found for TrainJob '${TRAINJOB_NAME}' -- it may not have been scheduled yet"
else
  for pod in $POD_NAMES; do
    echo "--- logs: ${pod} ---"
    oc logs "${pod}" -n "${NAMESPACE}" --tail=200 2>/dev/null || echo "  (log not available yet)"
  done
fi

log_section "Result"
if [ "$STATUS" = "Complete" ]; then
  log_pass "TrainJob '${TRAINJOB_NAME}' completed successfully"
elif [ "$STATUS" = "Failed" ]; then
  log_fail "TrainJob '${TRAINJOB_NAME}' failed. See logs above."
  exit 1
else
  log_warn "TrainJob '${TRAINJOB_NAME}' did not reach a terminal state within ${TRAINJOB_WAIT_TIMEOUT_SECONDS}s. Check with: oc get trainjob ${TRAINJOB_NAME} -n ${NAMESPACE} -o yaml"
  exit 1
fi
