#!/usr/bin/env bash
# Human-readable snapshot of everything this demo currently has running, grouped exactly
# like a presenter would narrate it: CLUSTER, TRAINING, PIPELINES, MLFLOW. Read-only.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source scripts/lib.sh
set +e

line() { printf '%-24s %s\n' "$1" "$2"; }

echo "== CLUSTER =="
if oc_login_check; then
  line "OpenShift" "PASS ($(oc whoami --show-server 2>/dev/null || echo unknown))"
else
  line "OpenShift" "FAIL (not logged in)"
fi
# See scripts/preflight.sh for why the version is derived from the CSV name rather than a
# second `oc get <name> -A` query (invalid for a namespaced resource, fails silently here).
RHOAI_CSV=$(oc get csv -A -o name 2>/dev/null | grep -E 'rhods-operator|opendatahub-operator' | head -n1)
if [ -n "$RHOAI_CSV" ]; then
  RHOAI_NAME="${RHOAI_CSV#*/}"
  RHOAI_VERSION="${RHOAI_NAME##*.v}"
  [ "$RHOAI_VERSION" = "$RHOAI_NAME" ] && RHOAI_VERSION="${RHOAI_NAME#*-operator.}"
  line "OpenShift AI" "${RHOAI_VERSION:-installed}"
else
  line "OpenShift AI" "unknown (no permission to list CSVs, or not installed)"
fi
line "Namespace" "${NAMESPACE}$(oc get namespace "${NAMESPACE}" >/dev/null 2>&1 && echo ' (exists)' || echo ' (missing -- make bootstrap)')"

echo ""
echo "== TRAINING =="
if oc get crd trainjobs.trainer.kubeflow.org >/dev/null 2>&1; then line "Trainer v2 CRDs"  "PASS"; else line "Trainer v2 CRDs" "FAIL"; fi
if oc get clustertrainingruntime torch-distributed >/dev/null 2>&1; then line "torch-distributed" "PASS"; else line "torch-distributed" "MISSING"; fi
if oc get imagestreamtag "${IMAGE_STREAM_NAME}:${IMAGE_TAG}" -n "${NAMESPACE}" >/dev/null 2>&1; then
  GIT_SHA_TAG=$(oc get imagestreamtag "${IMAGE_STREAM_NAME}:${IMAGE_TAG}" -n "${NAMESPACE}" -o jsonpath='{.metadata.annotations.demo\.git\.sha}' 2>/dev/null)
  line "Training image" "PASS${GIT_SHA_TAG:+ (git_sha=${GIT_SHA_TAG:0:12})}"
else
  line "Training image" "NOT BUILT -- make build"
fi
LAST_TJ=$(oc get trainjob -n "${NAMESPACE}" -l app.kubernetes.io/part-of=rhoai-pytorch-trainer-demo --sort-by=.metadata.creationTimestamp -o name 2>/dev/null | tail -n1)
if [ -n "$LAST_TJ" ]; then
  TJ_NAME=${LAST_TJ#*/}
  COMPLETE=$(oc get "$LAST_TJ" -n "${NAMESPACE}" -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' 2>/dev/null)
  FAILED=$(oc get "$LAST_TJ" -n "${NAMESPACE}" -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null)
  if [ "$COMPLETE" = "True" ]; then TJ_STATE="COMPLETE"; elif [ "$FAILED" = "True" ]; then TJ_STATE="FAILED"; else TJ_STATE="RUNNING/PENDING"; fi
  WORKERS=$(oc get pods -n "${NAMESPACE}" -l "jobset.sigs.k8s.io/jobset-name=${TJ_NAME}" --no-headers 2>/dev/null | wc -l)
  line "Last TrainJob" "${TJ_NAME} -> ${TJ_STATE}"
  line "Workers" "${WORKERS}"
else
  line "Last TrainJob" "NONE -- make train"
fi

echo ""
echo "== PIPELINES =="
if oc get secret minio-credentials -n "${NAMESPACE}" >/dev/null 2>&1 && oc get deployment minio -n "${NAMESPACE}" >/dev/null 2>&1; then
  MINIO_READY=$(oc get deployment minio -n "${NAMESPACE}" -o jsonpath='{.status.readyReplicas}' 2>/dev/null)
  line "Object storage" "$([ "${MINIO_READY:-0}" -ge 1 ] 2>/dev/null && echo READY || echo NOT READY)"
else
  line "Object storage" "NOT CONFIGURED -- make storage"
fi
DSPA=$(oc get datasciencepipelinesapplication -n "${NAMESPACE}" -o name 2>/dev/null | head -n1)
if [ -n "$DSPA" ]; then
  DSPA_READY=$(oc get "$DSPA" -n "${NAMESPACE}" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null)
  line "Pipeline Server" "$([ "$DSPA_READY" = "True" ] && echo READY || echo "NOT READY (${DSPA_READY:-unknown})")"
else
  line "Pipeline Server" "NOT CONFIGURED -- make pipeline-server"
fi
if [ -x "${REPO_ROOT}/.venv/bin/python" ]; then
  PIPELINE_DETECT=$("${PYTHON_BIN}" scripts/pipeline_client.py detect 2>/dev/null)
  PIPELINE_RC=$?
  if [ $PIPELINE_RC -eq 0 ]; then
    line "Pipeline" "READY (${PIPELINE_DETECT})"
    RUN_STATUS=$("${PYTHON_BIN}" scripts/pipeline_client.py status 2>/dev/null)
    RUN_RC=$?
    case $RUN_RC in
      0) line "Last run" "SUCCEEDED (${RUN_STATUS})" ;;
      2) line "Last run" "RUNNING/PENDING (${RUN_STATUS})" ;;
      *) line "Last run" "NONE OR FAILED (${RUN_STATUS:-no runs found})" ;;
    esac
    SCHED=$("${PYTHON_BIN}" scripts/pipeline_client.py schedule-status 2>/dev/null)
    if echo "$SCHED" | grep -q "EXISTS"; then line "Schedule" "CONFIGURED (${SCHEDULE_CRON})"; else line "Schedule" "NOT CONFIGURED -- make schedule"; fi
  else
    line "Pipeline" "NOT AVAILABLE"
  fi
else
  line "Pipeline" "unknown -- run 'make install' first"
fi

echo ""
echo "== MLFLOW =="
MLFLOW_URI=$(detect_mlflow_uri)
if [ -n "$MLFLOW_URI" ]; then
  # A Route existing doesn't mean the server behind it is actually up -- check the
  # Deployment's own readyReplicas too (this demo's own MLflow only; an externally-managed
  # MLFLOW_TRACKING_URI has no Deployment here to check, so it's reported as-is).
  if oc get deployment mlflow -n "${NAMESPACE}" >/dev/null 2>&1; then
    MLFLOW_READY=$(oc get deployment mlflow -n "${NAMESPACE}" -o jsonpath='{.status.readyReplicas}' 2>/dev/null)
    if [ "${MLFLOW_READY:-0}" -ge 1 ] 2>/dev/null; then
      line "MLflow" "READY (${MLFLOW_URI})"
    else
      line "MLflow" "NOT READY -- Route exists but deployment/mlflow has 0 ready replicas (check: oc describe pod -l app.kubernetes.io/name=mlflow -n ${NAMESPACE})"
    fi
  else
    line "MLflow" "READY (${MLFLOW_URI}, externally managed)"
  fi
else
  line "MLflow" "OPTIONAL -- not configured (make mlflow, or export MLFLOW_TRACKING_URI)"
fi
echo ""
