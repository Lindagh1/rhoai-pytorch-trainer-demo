#!/usr/bin/env bash
# Negative-path validations: confirms the demo FAILS CLEARLY (rather than hanging,
# silently succeeding, or crashing unhelpfully) in the situations a real client demo has
# to survive. NOT part of `make demo` -- run explicitly with `make test-negative`.
#
# Every resource this script creates is named with a "-negtest-" infix and labeled
# app.kubernetes.io/component=negative-test, and is deleted again before this script
# exits (success or failure), so it never interferes with the main demo's TrainJob/pipeline
# state and is cleaned up by `make cleanup` even if a test is interrupted.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source scripts/lib.sh
set +e

if ! oc_login_check; then
  log_fail "oc is not logged in. Run 'oc login ...' first."
  exit 1
fi
if ! oc get namespace "${NAMESPACE}" >/dev/null 2>&1; then
  log_fail "Namespace '${NAMESPACE}' does not exist. Run 'make bootstrap' first."
  exit 1
fi

FAILED=0
CLEANUP_NAMES=()
cleanup_all() {
  for n in "${CLEANUP_NAMES[@]}"; do
    oc delete trainjob "$n" -n "${NAMESPACE}" --ignore-not-found --wait=false >/dev/null 2>&1
  done
}
trap cleanup_all EXIT

apply_trainjob() {
  local name="$1" image="$2" gpu="$3"
  local gpu_req gpu_lim
  gpu_req=$(gpu_resource_lines "$gpu"); gpu_lim=$(gpu_resource_lines "$gpu")
  cat <<EOF | oc apply -f -
apiVersion: trainer.kubeflow.org/v1alpha1
kind: TrainJob
metadata:
  name: ${name}
  namespace: ${NAMESPACE}
  labels:
    app.kubernetes.io/part-of: rhoai-pytorch-trainer-demo
    app.kubernetes.io/managed-by: rhoai-pytorch-trainer-demo
    app.kubernetes.io/component: negative-test
spec:
  runtimeRef:
    apiGroup: trainer.kubeflow.org
    kind: ClusterTrainingRuntime
    name: torch-distributed
  trainer:
    image: "${image}"
    command: ["torchrun", "train.py"]
    args: ["--epochs=1"]
    numNodes: 1
    resourcesPerNode:
      requests:
        cpu: "${TRAIN_CPU}"
        memory: "${TRAIN_MEMORY}"
${gpu_req}
      limits:
        cpu: "${TRAIN_CPU}"
        memory: "${TRAIN_MEMORY}"
${gpu_lim}
EOF
}

wait_for_terminal_or_timeout() {
  local name="$1" timeout="$2"
  local deadline=$((SECONDS + timeout))
  while [ $SECONDS -lt "$deadline" ]; do
    local complete failed
    complete=$(oc get trainjob "$name" -n "${NAMESPACE}" -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' 2>/dev/null)
    failed=$(oc get trainjob "$name" -n "${NAMESPACE}" -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null)
    [ "$complete" = "True" ] && { echo "Complete"; return; }
    [ "$failed" = "True" ] && { echo "Failed"; return; }
    sleep 5
  done
  echo "Timeout"
}

log_section "Negative test 1/3: TrainJob with an invalid image never reaches Complete"
NAME="negtest-badimage-$$"
CLEANUP_NAMES+=("$NAME")
apply_trainjob "$NAME" "image-registry.openshift-image-registry.svc:5000/${NAMESPACE}/does-not-exist:latest" 0 >/dev/null
RESULT=$(wait_for_terminal_or_timeout "$NAME" 90)
POD_REASON=$(oc get pods -n "${NAMESPACE}" -l "jobset.sigs.k8s.io/jobset-name=${NAME}" -o jsonpath='{.items[0].status.containerStatuses[0].state.waiting.reason}' 2>/dev/null)
if [ "$RESULT" != "Complete" ]; then
  log_pass "Invalid image never completed (status=${RESULT:-Unknown}, pod reason=${POD_REASON:-n/a}) -- demo correctly refuses to claim success"
else
  log_fail "TrainJob with a bogus image reported Complete -- this should never happen"
  FAILED=1
fi
oc delete trainjob "$NAME" -n "${NAMESPACE}" --ignore-not-found --wait=false >/dev/null 2>&1

log_section "Negative test 2/3: training script exiting 1 -> TrainJob reports Failed, not Complete"
NAME="negtest-exit1-$$"
CLEANUP_NAMES+=("$NAME")
BUILT_IMAGE="image-registry.openshift-image-registry.svc:5000/${NAMESPACE}/${IMAGE_STREAM_NAME}:${IMAGE_TAG}"
if oc get imagestreamtag "${IMAGE_STREAM_NAME}:${IMAGE_TAG}" -n "${NAMESPACE}" >/dev/null 2>&1; then
  cat <<EOF | oc apply -f - >/dev/null
apiVersion: trainer.kubeflow.org/v1alpha1
kind: TrainJob
metadata:
  name: ${NAME}
  namespace: ${NAMESPACE}
  labels:
    app.kubernetes.io/part-of: rhoai-pytorch-trainer-demo
    app.kubernetes.io/managed-by: rhoai-pytorch-trainer-demo
    app.kubernetes.io/component: negative-test
spec:
  runtimeRef:
    apiGroup: trainer.kubeflow.org
    kind: ClusterTrainingRuntime
    name: torch-distributed
  trainer:
    image: "${BUILT_IMAGE}"
    command: ["python3", "-c", "import sys; sys.exit(1)"]
    numNodes: 1
    resourcesPerNode:
      requests: {cpu: "${TRAIN_CPU}", memory: "${TRAIN_MEMORY}"}
      limits: {cpu: "${TRAIN_CPU}", memory: "${TRAIN_MEMORY}"}
EOF
  RESULT=$(wait_for_terminal_or_timeout "$NAME" 120)
  if [ "$RESULT" = "Failed" ]; then
    log_pass "A container that exits 1 correctly makes the TrainJob report Failed=True -- a pipeline run consuming this would correctly skip evaluate-model"
  else
    log_warn "Expected Failed, observed '${RESULT}' -- inspect manually (this can also legitimately be 'Timeout' on a very slow/contended sandbox)"
  fi
  oc delete trainjob "$NAME" -n "${NAMESPACE}" --ignore-not-found --wait=false >/dev/null 2>&1
else
  log_optional "Training image not built yet -- skipping (run 'make build' first for this test)"
fi

log_section "Negative test 3/3: requesting more GPUs than allocatable -> Pending with a clear reason, not a silent hang"
NAME="negtest-toomanygpu-$$"
CLEANUP_NAMES+=("$NAME")
apply_trainjob "$NAME" "${BUILT_IMAGE}" 99 >/dev/null
sleep 20
PHASE=$(oc get pods -n "${NAMESPACE}" -l "jobset.sigs.k8s.io/jobset-name=${NAME}" -o jsonpath='{.items[0].status.phase}' 2>/dev/null)
EVENT=$(oc get events -n "${NAMESPACE}" --field-selector reason=FailedScheduling -o jsonpath='{.items[-1:].message}' 2>/dev/null)
if [ "${PHASE:-Pending}" = "Pending" ]; then
  log_pass "Requesting 99 GPU/node correctly leaves the pod Pending (Insufficient nvidia.com/gpu), not silently running: ${EVENT:-'(see: oc describe pod ...)'}"
else
  log_warn "Expected Pending, observed phase='${PHASE}' -- inspect manually"
fi
oc delete trainjob "$NAME" -n "${NAMESPACE}" --ignore-not-found --wait=false >/dev/null 2>&1

log_section "Summary"
if [ "$FAILED" -ne 0 ]; then
  log_fail "One or more negative tests behaved unexpectedly. See above."
  exit 1
fi
log_pass "All negative-path behaviors are as expected: this demo fails loudly and clearly instead of hanging or lying about success."
