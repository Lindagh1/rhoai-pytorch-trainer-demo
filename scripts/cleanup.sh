#!/usr/bin/env bash
# Deletes ONLY resources that belong to this demo. Never touches operators, CRDs,
# ClusterTrainingRuntimes, or any namespace/workload this demo did not create.
#
# Safety: this script refuses to delete the namespace unless it is labeled
# app.kubernetes.io/part-of=${PART_OF_LABEL} (set by manifests/namespace.yaml), so pointing
# NAMESPACE at an existing, unrelated namespace can never cause data loss.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source scripts/lib.sh

if ! oc_login_check; then
  log_fail "oc is not logged in. Run 'oc login ...' first."
  exit 1
fi

if ! oc get namespace "${NAMESPACE}" >/dev/null 2>&1; then
  log_info "Namespace '${NAMESPACE}' does not exist. Nothing to clean up."
  exit 0
fi

NS_LABEL=$(oc get namespace "${NAMESPACE}" -o jsonpath="{.metadata.labels['app\.kubernetes\.io/part-of']}" 2>/dev/null || true)
if [ "$NS_LABEL" != "${PART_OF_LABEL}" ]; then
  log_fail "Namespace '${NAMESPACE}' exists but is NOT labeled app.kubernetes.io/part-of=${PART_OF_LABEL}."
  log_fail "Refusing to delete a namespace this demo did not create. Set NAMESPACE to the value used at bootstrap time."
  exit 1
fi

log_section "Best-effort: deleting the scheduled run (if any) before removing the namespace"
"${PYTHON_BIN}" scripts/pipeline_client.py delete-schedule 2>/dev/null || log_warn "Could not delete the recurring run (Pipeline Server may be unreachable) -- it will be removed along with the namespace's pipeline database anyway"

log_section "Deleting namespace '${NAMESPACE}'"
echo "This namespace is labeled app.kubernetes.io/part-of=${PART_OF_LABEL} and owns only this demo's resources"
echo "(training, storage/MinIO, Pipeline Server/DSPA, MLflow -- all namespace-scoped, nothing cluster-wide):"
oc get all,trainjob,datasciencepipelinesapplication,role,rolebinding,serviceaccount,buildconfig,imagestream,pvc -n "${NAMESPACE}" -o name 2>/dev/null | sed 's/^/  - /' || true

if [ -t 0 ] && [ "${CONFIRM:-}" != "yes" ]; then
  read -r -p "Type the namespace name ('${NAMESPACE}') to confirm deletion: " REPLY
  if [ "$REPLY" != "${NAMESPACE}" ]; then
    log_fail "Confirmation did not match. Aborting."
    exit 1
  fi
elif [ "${CONFIRM:-}" != "yes" ]; then
  log_fail "Non-interactive shell: set CONFIRM=yes to proceed without a prompt."
  exit 1
fi

oc delete namespace "${NAMESPACE}"
log_pass "Namespace '${NAMESPACE}' deleted. No operators, CRDs, or cluster-scoped ClusterTrainingRuntime resources were touched."
