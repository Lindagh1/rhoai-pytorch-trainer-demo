#!/usr/bin/env bash
# Read-only capability check for a new OpenShift AI sandbox. Never creates, modifies, or
# deletes anything. Run this BEFORE `make bootstrap` on any new cluster.
#
# Every check prints one of: PASS, WARN, FAIL, OPTIONAL.
#   PASS/FAIL  -> required for the corresponding demo capability.
#   WARN       -> works, but degraded (e.g. no GPU -> CPU demo mode).
#   OPTIONAL   -> a nice-to-have (e.g. MLflow) that the demo works fine without.
#
# Exit code is non-zero only if a hard-required capability is missing (no oc login, no
# Kubeflow Trainer CRDs at all). GPU/MLflow/Pipelines absence never fails the script --
# they downgrade specific demo capabilities instead.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source scripts/lib.sh
set +e  # preflight deliberately continues past individual check failures

HARD_FAIL=0

log_section "Local tooling"
if require_cmd oc; then log_pass "oc CLI available ($(oc version --client 2>/dev/null | head -n1))"; else log_fail "oc CLI not found on PATH"; HARD_FAIL=1; fi
if require_cmd envsubst; then log_pass "envsubst available (used to render manifests)"; else log_fail "envsubst not found (install gettext)"; HARD_FAIL=1; fi
if require_cmd python3; then log_pass "python3 available ($(python3 --version 2>&1))"; else log_warn "python3 not found -- needed for make compile-pipeline / make evaluate"; fi

log_section "OpenShift cluster connectivity"
if oc_login_check; then
  CURRENT_USER=$(oc whoami 2>/dev/null || echo unknown)
  CURRENT_SERVER=$(oc whoami --show-server 2>/dev/null || echo unknown)
  log_pass "oc is logged in as '${CURRENT_USER}' against '${CURRENT_SERVER}'"
else
  log_fail "oc is not logged in. Run 'oc login ...' first."
  HARD_FAIL=1
fi

log_section "OpenShift version"
# `oc version -o json` can exit non-zero (e.g. while logged out) while still printing valid
# client-only JSON on stdout; decouple the oc call from the python parse below so a
# non-zero oc exit code can never also trigger the `|| echo` fallback on top of python's
# own (already correct) output under `set -o pipefail`.
OCP_JSON=$(oc version -o json 2>/dev/null || true)
OCP_VERSION=$(printf '%s' "$OCP_JSON" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("openshiftVersion","unknown"))' 2>/dev/null || echo "unknown")
if [ "$OCP_VERSION" != "unknown" ] && [ -n "$OCP_VERSION" ]; then
  log_pass "OpenShift version: ${OCP_VERSION}"
else
  log_warn "Could not determine OpenShift version (non-fatal)"
fi

log_section "OpenShift AI (RHOAI/ODH) version"
RHOAI_CSV=$(oc get csv -A -o name 2>/dev/null | grep -E 'rhods-operator|opendatahub-operator' | head -n1 || true)
if [ -n "$RHOAI_CSV" ]; then
  RHOAI_VERSION=$(oc get "$RHOAI_CSV" -A -o jsonpath='{.items[0].spec.version}' 2>/dev/null || true)
  log_pass "OpenShift AI operator installed (${RHOAI_CSV#*/}${RHOAI_VERSION:+, version $RHOAI_VERSION})"
else
  log_warn "Could not find a rhods-operator/opendatahub-operator CSV (may lack cluster-wide list permission, or RHOAI is not installed)"
fi

log_section "Kubeflow Trainer v2 (distributed training)"
if oc get crd trainjobs.trainer.kubeflow.org >/dev/null 2>&1; then
  log_pass "CRD present: trainjobs.trainer.kubeflow.org"
else
  log_fail "CRD missing: trainjobs.trainer.kubeflow.org (enable the 'trainer' component in the DataScienceCluster)"
  HARD_FAIL=1
fi

if oc get crd clustertrainingruntimes.trainer.kubeflow.org >/dev/null 2>&1; then
  log_pass "CRD present: clustertrainingruntimes.trainer.kubeflow.org"
else
  log_fail "CRD missing: clustertrainingruntimes.trainer.kubeflow.org"
  HARD_FAIL=1
fi

if oc get crd jobsets.jobset.x-k8s.io >/dev/null 2>&1; then
  log_pass "CRD present: jobsets.jobset.x-k8s.io (JobSet operator installed)"
else
  log_fail "CRD missing: jobsets.jobset.x-k8s.io (Kubeflow Trainer v2 requires the JobSet operator)"
  HARD_FAIL=1
fi

if oc get clustertrainingruntime torch-distributed >/dev/null 2>&1; then
  log_pass "ClusterTrainingRuntime 'torch-distributed' is available"
elif oc get clustertrainingruntime >/dev/null 2>&1; then
  OTHER_RUNTIMES=$(oc get clustertrainingruntime -o name 2>/dev/null | tr '\n' ' ')
  log_warn "ClusterTrainingRuntime 'torch-distributed' not found. Available runtimes: ${OTHER_RUNTIMES:-none}"
else
  log_fail "Cannot list ClusterTrainingRuntime resources (CRD missing or no permission)"
fi

log_section "GPU availability"
GPU_NODES=$(oc get nodes -o jsonpath='{range .items[*]}{.metadata.name}{"="}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}' 2>/dev/null | grep -v '=$' || true)
if [ -n "$GPU_NODES" ]; then
  TOTAL_GPUS=$(echo "$GPU_NODES" | awk -F= '{sum+=$2} END {print sum+0}')
  log_pass "GPU-visible nodes found (total allocatable nvidia.com/gpu: ${TOTAL_GPUS})"
  echo "$GPU_NODES" | while IFS='=' read -r node gpus; do echo "         - ${node}: ${gpus} GPU(s)"; done
  NODES_JSON=$(oc get nodes -o json 2>/dev/null || true)
  FREE_GPUS=$(printf '%s' "$NODES_JSON" | python3 -c '
import json,sys
data=json.load(sys.stdin)
total=0
for n in data.get("items", []):
    alloc = n.get("status", {}).get("allocatable", {}).get("nvidia.com/gpu")
    if alloc:
        total += int(alloc)
print(total)
' 2>/dev/null || echo 0)
  if [ "${FREE_GPUS:-0}" -gt 0 ] 2>/dev/null; then
    log_pass "GPU capacity currently allocatable: ${FREE_GPUS}"
  else
    log_warn "GPU nodes exist but none currently allocatable (likely fully consumed by other workloads) -- CPU demo mode will be used"
  fi
else
  log_warn "No GPU-visible nodes found -- the demo will run in CPU-only distributed mode (see manifests/trainjob-gpu-example.yaml for the GPU scenario)"
fi

log_section "Data Science Pipelines / AI Pipelines"
if oc get crd datasciencepipelinesapplications.datasciencepipelinesapplications.opendatahub.io >/dev/null 2>&1; then
  log_pass "CRD present: DataSciencePipelinesApplication (Data Science Pipelines component installed)"
else
  log_fail "CRD missing: DataSciencePipelinesApplication -- enable the 'datasciencepipelines' component in the DataScienceCluster"
fi

DSPA_IN_NAMESPACE=$(oc get datasciencepipelinesapplication -n "${NAMESPACE}" -o name 2>/dev/null || true)
if [ -n "$DSPA_IN_NAMESPACE" ]; then
  log_pass "A DataSciencePipelinesApplication already exists in namespace '${NAMESPACE}': ${DSPA_IN_NAMESPACE}"
  DSPA_READY=$(oc get datasciencepipelinesapplication -n "${NAMESPACE}" -o jsonpath='{.items[0].status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || true)
  if [ "$DSPA_READY" = "True" ]; then
    log_pass "DataSciencePipelinesApplication reports Ready=True"
  else
    log_warn "DataSciencePipelinesApplication exists but Ready condition is '${DSPA_READY:-unknown}'"
  fi
else
  log_warn "No DataSciencePipelinesApplication found in namespace '${NAMESPACE}' yet (namespace may not exist yet -- this is expected before 'make bootstrap'; a pipeline server must be created before 'make pipeline' will work, see docs/troubleshooting.md)"
fi

log_section "Object storage for the pipeline server"
if [ -n "${AWS_S3_ENDPOINT:-}" ] || [ -n "${PIPELINE_S3_ENDPOINT:-}" ]; then
  log_pass "S3-compatible endpoint provided via environment (AWS_S3_ENDPOINT/PIPELINE_S3_ENDPOINT)"
else
  log_optional "No S3 endpoint provided via environment. Needed only if you want 'make deploy-pipeline' to also provision a DataSciencePipelinesApplication -- see README.md 'Pipeline server storage'"
fi

log_section "MLflow (optional)"
MLFLOW_ROUTE=$(oc get route -A -l 'app.kubernetes.io/name=mlflow' -o name 2>/dev/null | head -n1 || true)
if [ -n "${MLFLOW_TRACKING_URI:-}" ]; then
  log_optional "MLFLOW_TRACKING_URI is set (${MLFLOW_TRACKING_URI}) -- training/evaluation will log to it"
elif [ -n "$MLFLOW_ROUTE" ]; then
  log_optional "An MLflow route was found on the cluster (${MLFLOW_ROUTE}) but MLFLOW_TRACKING_URI is not set -- export it to use it"
else
  log_optional "No MLflow detected and MLFLOW_TRACKING_URI is not set -- training/evaluation will simply skip experiment tracking"
fi

log_section "Summary"
echo "Namespace this demo would use: ${NAMESPACE} (override with 'export NAMESPACE=...')"
if [ "$HARD_FAIL" -ne 0 ]; then
  log_fail "One or more REQUIRED capabilities are missing. See FAIL lines above before running 'make bootstrap'."
  exit 1
else
  log_pass "All required capabilities are present. Optional gaps (GPU/MLflow/Pipelines) only affect specific demo capabilities, not the core distributed-training demo."
  exit 0
fi
