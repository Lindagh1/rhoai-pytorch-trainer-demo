#!/usr/bin/env bash
# Detects whether a Data Science Pipelines server is available in ${NAMESPACE}. If it is,
# uploads pipeline/pipeline.yaml and starts a real run. If it is NOT available, this
# script says so clearly and exits non-zero -- it never pretends the pipeline ran.
#
# A Pipeline Server itself is not created automatically here: it requires object storage
# credentials, which must be provided explicitly (see README.md "Pipeline Server storage").
# This keeps that step reproducible without ever committing a secret to GitHub.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source scripts/lib.sh

if ! oc_login_check; then
  log_fail "oc is not logged in. Run 'oc login ...' first."
  exit 1
fi

ACTION="${1:-run}"

log_section "Detecting Data Science Pipelines server in namespace '${NAMESPACE}'"
if ! python3 scripts/pipeline_client.py detect; then
  log_fail "No usable Data Science Pipelines server found in '${NAMESPACE}'."
  echo "This demo does NOT simulate a pipeline run without a real Pipeline Server."
  echo "See README.md 'Pipeline Server storage' to provision one (needs S3-compatible storage credentials)."
  exit 1
fi
log_pass "Pipeline Server detected"

if [ ! -f pipeline/pipeline.yaml ]; then
  log_info "pipeline/pipeline.yaml missing -- compiling it now"
  make compile-pipeline
fi

case "$ACTION" in
  upload)
    log_section "Uploading pipeline"
    python3 scripts/pipeline_client.py upload
    ;;
  run)
    log_section "Uploading pipeline"
    python3 scripts/pipeline_client.py upload
    log_section "Starting pipeline run"
    python3 scripts/pipeline_client.py run
    ;;
  status)
    python3 scripts/pipeline_client.py status "${2:-}"
    ;;
  create-schedule)
    log_section "Uploading pipeline"
    python3 scripts/pipeline_client.py upload
    log_section "Creating recurring run (cron: ${SCHEDULE_CRON})"
    python3 scripts/pipeline_client.py create-schedule
    ;;
  delete-schedule)
    log_section "Deleting recurring run"
    python3 scripts/pipeline_client.py delete-schedule
    ;;
  *)
    echo "Usage: $0 [upload|run|status|create-schedule|delete-schedule]"
    exit 2
    ;;
esac
