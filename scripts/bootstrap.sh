#!/usr/bin/env bash
# Creates the demo namespace and its RBAC. Idempotent: safe to run repeatedly (uses
# `oc apply`, never destroys anything, never touches resources it doesn't own).
#
# Creates ONLY:
#   - Namespace ${NAMESPACE}
#   - ServiceAccount/Role/RoleBinding for the pipeline (manifests/rbac.yaml)
#
# Does NOT create a Data Science Pipelines server (see scripts/deploy-pipeline.sh) and
# does NOT create the training image (see scripts/build-training-image.sh) -- those are
# separate, explicit steps so this script stays fast and safe to re-run.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source scripts/lib.sh

if ! oc_login_check; then
  log_fail "oc is not logged in. Run 'oc login ...' first."
  exit 1
fi

log_section "Bootstrap: namespace and RBAC (namespace=${NAMESPACE})"

export NAMESPACE
render_manifest manifests/namespace.yaml | oc apply -f -
log_pass "Namespace '${NAMESPACE}' ensured"

render_manifest manifests/rbac.yaml | oc apply -f -
log_pass "RBAC ensured: ServiceAccount/${PIPELINE_SERVICE_ACCOUNT}, Role/trainjob-operator, RoleBinding in namespace '${NAMESPACE}'"

log_section "Bootstrap complete"
echo "Next steps:"
echo "  make build     # build the training image from this repository (OpenShift Build)"
echo "  make train     # run a distributed TrainJob"
echo "  make pipeline  # compile + deploy + run the AI Pipeline (requires a Pipeline Server)"
