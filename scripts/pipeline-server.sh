#!/usr/bin/env bash
# Idempotently creates a DataSciencePipelinesApplication (the Data Science Pipelines /
# "AI Pipelines" server the OpenShift AI Dashboard looks for) in this namespace, backed by
# the namespace-local MinIO instance created by `make storage`. Waits for it to actually
# report Ready=True before returning success -- never claims the Pipeline Server is usable
# if it is not.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source scripts/lib.sh

if ! oc_login_check; then
  log_fail "oc is not logged in. Run 'oc login ...' first."
  exit 1
fi

if ! oc get crd datasciencepipelinesapplications.datasciencepipelinesapplications.opendatahub.io >/dev/null 2>&1; then
  log_fail "CRD 'datasciencepipelinesapplications' is not installed on this cluster -- enable the 'datasciencepipelines' component in the DataScienceCluster."
  exit 1
fi

if ! oc get secret minio-credentials -n "${NAMESPACE}" >/dev/null 2>&1; then
  log_fail "Secret 'minio-credentials' not found in '${NAMESPACE}'. Run 'make storage' first."
  exit 1
fi

log_section "Applying DataSciencePipelinesApplication '${DSPA_NAME}' (namespace=${NAMESPACE})"
export NAMESPACE DSPA_NAME PIPELINE_BUCKET
render_manifest manifests/dspa.yaml | oc apply -f -

log_section "Waiting for DataSciencePipelinesApplication to report Ready=True (timeout: ${DSPA_WAIT_TIMEOUT_SECONDS}s)"
DEADLINE=$((SECONDS + DSPA_WAIT_TIMEOUT_SECONDS))
READY="False"
while [ $SECONDS -lt "$DEADLINE" ]; do
  READY=$(oc get datasciencepipelinesapplication "${DSPA_NAME}" -n "${NAMESPACE}" \
    -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo "False")
  if [ "$READY" = "True" ]; then
    break
  fi
  echo "  [$((DEADLINE - SECONDS))s left] Ready=${READY:-Unknown}"
  sleep 10
done

if [ "$READY" != "True" ]; then
  log_fail "DataSciencePipelinesApplication '${DSPA_NAME}' did not become Ready within ${DSPA_WAIT_TIMEOUT_SECONDS}s."
  echo "Current conditions:"
  oc get datasciencepipelinesapplication "${DSPA_NAME}" -n "${NAMESPACE}" -o jsonpath='{.status.conditions}' 2>/dev/null | python3 -m json.tool 2>/dev/null || true
  echo ""
  echo "Do NOT proceed to 'make pipeline' -- see TROUBLESHOOTING.md 'Pipeline Server'."
  exit 1
fi

log_pass "DataSciencePipelinesApplication '${DSPA_NAME}' is Ready"
echo "Dashboard path: Data Science Projects -> ${NAMESPACE} -> Pipelines"
echo "Next: make pipeline"
