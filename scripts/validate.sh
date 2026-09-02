#!/usr/bin/env bash
# Verifies the demo actually did what it claims to have done -- reads real cluster state,
# never assumes success. Prints PASS/FAIL for each checkpoint. Exit code is non-zero if any
# REQUIRED checkpoint fails (Pipeline Server checks are OPTIONAL if none is available).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source scripts/lib.sh
set +e

FAILED=0
fail() { log_fail "$1"; FAILED=1; }

log_section "Namespace"
if oc get namespace "${NAMESPACE}" >/dev/null 2>&1; then
  log_pass "Namespace '${NAMESPACE}' exists"
else
  fail "Namespace '${NAMESPACE}' does not exist -- run 'make bootstrap'"
fi

log_section "Training image"
if oc get imagestreamtag "${IMAGE_STREAM_NAME}:${IMAGE_TAG}" -n "${NAMESPACE}" >/dev/null 2>&1; then
  IMAGE_REF=$(oc get imagestreamtag "${IMAGE_STREAM_NAME}:${IMAGE_TAG}" -n "${NAMESPACE}" -o jsonpath='{.image.dockerImageReference}' 2>/dev/null)
  log_pass "Training image built: ${IMAGE_REF}"
else
  fail "Training image '${IMAGE_STREAM_NAME}:${IMAGE_TAG}' not found in '${NAMESPACE}' -- run 'make build'"
fi

log_section "TrainJob"
if oc get trainjob "${TRAINJOB_NAME}" -n "${NAMESPACE}" >/dev/null 2>&1; then
  log_pass "TrainJob '${TRAINJOB_NAME}' exists"

  COMPLETE=$(oc get trainjob "${TRAINJOB_NAME}" -n "${NAMESPACE}" -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' 2>/dev/null)
  if [ "$COMPLETE" = "True" ]; then
    log_pass "TrainJob '${TRAINJOB_NAME}' completed successfully"
  else
    fail "TrainJob '${TRAINJOB_NAME}' has not completed successfully (Complete=${COMPLETE:-unknown})"
  fi

  POD_NAMES=$(oc get pods -n "${NAMESPACE}" --no-headers 2>/dev/null | awk -v p="${TRAINJOB_NAME}-" '$1 ~ "^"p {print $1}')
  POD_COUNT=$(echo "$POD_NAMES" | grep -c . || true)
  if [ "$POD_COUNT" -ge "$TRAIN_NODES" ] 2>/dev/null; then
    log_pass "${POD_COUNT} worker pod(s) visible for TrainJob '${TRAINJOB_NAME}' (expected >= ${TRAIN_NODES})"
  else
    fail "Only ${POD_COUNT} worker pod(s) visible, expected >= ${TRAIN_NODES}"
  fi

  ALL_LOGS=""
  for pod in $POD_NAMES; do
    ALL_LOGS="${ALL_LOGS}
$(oc logs "$pod" -n "${NAMESPACE}" 2>/dev/null)"
  done
  RANKS_SEEN=$(echo "$ALL_LOGS" | grep -oE 'rank=[0-9]+' | sort -u | wc -l)
  if [ "$RANKS_SEEN" -ge 1 ] 2>/dev/null; then
    log_pass "Logs contain ${RANKS_SEEN} distinct rank(s)"
    if [ "$TRAIN_NODES" -gt 1 ] 2>/dev/null && [ "$RANKS_SEEN" -lt 2 ]; then
      log_warn "TRAIN_NODES=${TRAIN_NODES} but only 1 distinct rank observed in logs -- check pod scheduling"
    fi
  else
    fail "No 'rank=' lines found in TrainJob pod logs"
  fi
else
  fail "TrainJob '${TRAINJOB_NAME}' not found in '${NAMESPACE}' -- run 'make train'"
fi

log_section "AI Pipeline"
PIPELINE_AVAILABLE=0
if python3 scripts/pipeline_client.py detect >/dev/null 2>&1; then
  PIPELINE_AVAILABLE=1
  log_pass "Data Science Pipelines server is reachable"

  UPLOAD_CHECK=$(python3 - <<'PYEOF'
import os, sys
sys.path.insert(0, "scripts")
from pipeline_client import build_client, PIPELINE_NAME
client = build_client()
if client is None:
    sys.exit(1)
pid = client.get_pipeline_id(PIPELINE_NAME)
print(pid or "")
PYEOF
)
  if [ -n "$UPLOAD_CHECK" ]; then
    log_pass "Pipeline '${PIPELINE_NAME}' is present on the server (id=${UPLOAD_CHECK})"

    RUN_STATUS_OUTPUT=$(python3 scripts/pipeline_client.py status 2>/dev/null)
    RUN_STATUS_CODE=$?
    echo "  ${RUN_STATUS_OUTPUT}"
    if [ "$RUN_STATUS_CODE" -eq 0 ]; then
      log_pass "Latest pipeline run succeeded"
    elif [ "$RUN_STATUS_CODE" -eq 2 ]; then
      log_warn "Latest pipeline run is still in progress"
    else
      fail "Latest pipeline run did not succeed -- run 'make pipeline'"
    fi
  else
    log_optional "Pipeline '${PIPELINE_NAME}' not yet uploaded -- run 'make pipeline'"
  fi
else
  log_optional "No Data Science Pipelines server available -- AI Pipeline checks skipped (this is OPTIONAL, not a failure of the core demo)"
fi

log_section "Summary"
if [ "$FAILED" -ne 0 ]; then
  log_fail "One or more required checks failed. See FAIL lines above."
  exit 1
else
  log_pass "All required checks passed."
  exit 0
fi
