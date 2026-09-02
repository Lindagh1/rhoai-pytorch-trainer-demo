#!/usr/bin/env bash
# Builds the training image directly from THIS GitHub repository using an OpenShift
# Build (Docker strategy), pushing to an ImageStream in the internal OpenShift registry.
#
#   GitHub  ->  OpenShift Build  ->  training image (internal registry)  ->  TrainJob
#
# GIT_REPO_URL / GIT_REF are auto-detected from `git remote`/`git branch` unless already
# exported, so nothing sandbox- or fork-specific is hardcoded here. Idempotent: re-running
# just re-applies the same BuildConfig/ImageStream and starts a new build.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source scripts/lib.sh

if ! oc_login_check; then
  log_fail "oc is not logged in. Run 'oc login ...' first."
  exit 1
fi

if [ -z "${GIT_REPO_URL}" ]; then
  log_fail "GIT_REPO_URL could not be auto-detected (no 'origin' remote configured) and is not set."
  echo "Set it explicitly, e.g.:"
  echo "  export GIT_REPO_URL=https://github.com/<you>/<your-fork>.git"
  exit 1
fi
log_info "Building from ${GIT_REPO_URL}@${GIT_REF}"

if [ -n "${SOURCE_SECRET_NAME}" ]; then
  SOURCE_SECRET_BLOCK="    sourceSecret:
      name: ${SOURCE_SECRET_NAME}"
  log_info "Using source secret '${SOURCE_SECRET_NAME}' for a private repository"
else
  SOURCE_SECRET_BLOCK=""
fi
export NAMESPACE IMAGE_STREAM_NAME IMAGE_TAG GIT_REPO_URL GIT_REF SOURCE_SECRET_BLOCK

log_section "Applying ImageStream + BuildConfig (namespace=${NAMESPACE}, image=${IMAGE_STREAM_NAME}:${IMAGE_TAG})"
render_manifest manifests/buildconfig.yaml | oc apply -f -

log_section "Starting build"
BUILD_NAME=$(oc start-build "${IMAGE_STREAM_NAME}" -n "${NAMESPACE}" -o name)
log_info "Started ${BUILD_NAME}"

log_section "Waiting for build to complete"
if oc logs -f "${BUILD_NAME}" -n "${NAMESPACE}"; then
  :
fi

PHASE=$(oc get "${BUILD_NAME}" -n "${NAMESPACE}" -o jsonpath='{.status.phase}')
if [ "$PHASE" = "Complete" ]; then
  IMAGE_REF=$(oc get imagestreamtag "${IMAGE_STREAM_NAME}:${IMAGE_TAG}" -n "${NAMESPACE}" -o jsonpath='{.image.dockerImageReference}' 2>/dev/null || echo "unknown")
  log_pass "Build complete. Image: ${IMAGE_REF}"
  echo "internal reference for TrainJob: image-registry.openshift-image-registry.svc:5000/${NAMESPACE}/${IMAGE_STREAM_NAME}:${IMAGE_TAG}"
else
  log_fail "Build ended in phase '${PHASE}'. See 'oc logs ${BUILD_NAME} -n ${NAMESPACE}' for details."
  exit 1
fi
