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
# `oc get <name> -A` is invalid for a namespaced resource (a CSV) -- it errors with "a
# resource cannot be retrieved by name across all namespaces" and, under `set +e`, silently
# leaves RHOAI_VERSION empty. The CSV's own name already ends in its version
# (e.g. "rhods-operator.3.5.0"), so derive it from there instead of a second, broken query.
RHOAI_CSV=$(oc get csv -A -o name 2>/dev/null | grep -E 'rhods-operator|opendatahub-operator' | head -n1 || true)
if [ -n "$RHOAI_CSV" ]; then
  RHOAI_NAME="${RHOAI_CSV#*/}"
  RHOAI_VERSION="${RHOAI_NAME##*.v}"
  [ "$RHOAI_VERSION" = "$RHOAI_NAME" ] && RHOAI_VERSION="${RHOAI_NAME#*-operator.}"
  log_pass "OpenShift AI operator installed (${RHOAI_NAME}${RHOAI_VERSION:+, version $RHOAI_VERSION})"
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
  # Allocatable capacity alone is not "free" -- subtract what other pods (this demo's own
  # or anyone else's) are already requesting cluster-wide, so this never recommends
  # MODE=gpu when the only GPU is already fully claimed by someone else's workload. This
  # demo never scales down or evicts another workload to make a GPU "available".
  USED_GPUS=$(oc get pods -A -o json 2>/dev/null | python3 -c '
import json,sys
data=json.load(sys.stdin)
total=0
for p in data.get("items", []):
    if p.get("status", {}).get("phase") not in ("Running", "Pending"):
        continue
    for c in p.get("spec", {}).get("containers", []):
        req = c.get("resources", {}).get("requests", {}).get("nvidia.com/gpu")
        if req:
            total += int(req)
print(total)
' 2>/dev/null || echo 0)
  FREE_GPUS=$(( TOTAL_GPUS - USED_GPUS ))
  if [ "$FREE_GPUS" -gt 0 ] 2>/dev/null; then
    log_pass "GPU capacity currently free: ${FREE_GPUS} (allocatable=${TOTAL_GPUS}, already requested by running/pending pods cluster-wide=${USED_GPUS})"
  else
    log_warn "GPU nodes exist (allocatable=${TOTAL_GPUS}) but ${USED_GPUS} already requested by other pods cluster-wide -- none free right now. CPU demo mode will be used; this demo will NOT scale down or evict the workload holding the GPU."
    FREE_GPUS=0
  fi
else
  log_warn "No GPU-visible nodes found -- the demo will run in CPU-only distributed mode (see manifests/trainjob-gpu-example.yaml for the GPU scenario)"
  FREE_GPUS=0
fi

log_section "Recommended MODE for 'make train' / 'make demo'"
if [ -n "${GPU_NODES:-}" ] && [ "${FREE_GPUS:-0}" -gt 0 ] 2>/dev/null; then
  log_pass "Recommended: MODE=gpu (GPU capacity is currently allocatable) -- e.g. 'make train MODE=gpu'"
else
  log_pass "Recommended: MODE=cpu (no currently-allocatable GPU) -- e.g. 'make train MODE=cpu' (this is also the default)"
fi

log_section "Cluster CPU/memory headroom"
# This demo's own components are deliberately tiny (see manifests/dspa.yaml,
# manifests/storage.yaml), but on a small/shared sandbox the CLUSTER itself can be at or
# near 100% CPU allocated by OTHER tenants' workloads before this demo requests anything.
# This is a real, common failure mode for 'make pipeline-server' (MariaDB/API server pods
# stuck Pending) that has nothing to do with this repo -- surface it here explicitly
# instead of letting it look like a silent hang later.
NODE_JSON=$(oc get nodes -o json 2>/dev/null || true)
if [ -n "$NODE_JSON" ]; then
  printf '%s' "$NODE_JSON" | python3 -c '
import json, sys
data = json.load(sys.stdin)

def to_millicpu(q):
    if q is None:
        return 0
    q = str(q)
    return int(q[:-1]) if q.endswith("m") else int(float(q) * 1000)

cap_cpu = sum(to_millicpu(n.get("status", {}).get("allocatable", {}).get("cpu")) for n in data.get("items", []))
print(f"__CAP_CPU__{cap_cpu}")
' > /tmp/preflight_cap_cpu.$$  2>/dev/null || true
  CAP_CPU=$(grep -o '__CAP_CPU__[0-9]*' /tmp/preflight_cap_cpu.$$ 2>/dev/null | sed 's/__CAP_CPU__//')
  rm -f /tmp/preflight_cap_cpu.$$
  PODS_JSON=$(oc get pods -A -o json 2>/dev/null || true)
  USED_CPU=$(printf '%s' "$PODS_JSON" | python3 -c '
import json, sys

def to_millicpu(q):
    if q is None:
        return 0
    q = str(q)
    return int(q[:-1]) if q.endswith("m") else int(float(q) * 1000)

data = json.load(sys.stdin)
total = 0
for p in data.get("items", []):
    if p.get("status", {}).get("phase") not in ("Running", "Pending"):
        continue
    for c in p.get("spec", {}).get("containers", []):
        total += to_millicpu(c.get("resources", {}).get("requests", {}).get("cpu"))
print(total)
' 2>/dev/null || echo 0)
  if [ -n "${CAP_CPU:-}" ] && [ "${CAP_CPU:-0}" -gt 0 ] 2>/dev/null; then
    FREE_CPU=$(( CAP_CPU - USED_CPU ))
    PCT_USED=$(( USED_CPU * 100 / CAP_CPU ))
    if [ "$FREE_CPU" -lt 200 ]; then
      log_warn "Cluster-wide CPU requests are at ${PCT_USED}% of allocatable (${USED_CPU}m/${CAP_CPU}m, ~${FREE_CPU}m free). 'make pipeline-server' (MariaDB + API server pods) may stay Pending until other tenants' workloads free CPU. This is a shared-sandbox capacity issue, not a bug in this repo -- see TROUBLESHOOTING.md 'make pipeline-server times out waiting for Ready'."
    else
      log_pass "Cluster-wide CPU requests: ${PCT_USED}% of allocatable (${USED_CPU}m/${CAP_CPU}m, ~${FREE_CPU}m free) -- enough headroom for this demo's own (small) components"
    fi
  else
    log_warn "Could not compute cluster CPU headroom (non-fatal)"
  fi
else
  log_warn "Could not read node capacity (non-fatal)"
fi

log_section "OBJECT STORAGE"
MINIO_DEPLOYED=0
if oc get deployment minio -n "${NAMESPACE}" >/dev/null 2>&1; then
  MINIO_DEPLOYED=1
  MINIO_READY=$(oc get deployment minio -n "${NAMESPACE}" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo 0)
  if [ "${MINIO_READY:-0}" -ge 1 ] 2>/dev/null; then
    log_pass "Namespace-local MinIO (object storage) is deployed and Ready in '${NAMESPACE}'"
  else
    log_warn "Namespace-local MinIO is deployed but not yet Ready in '${NAMESPACE}' (check: oc get pods -l app.kubernetes.io/name=minio -n ${NAMESPACE})"
  fi
elif [ -n "${AWS_S3_ENDPOINT:-}" ] || [ -n "${PIPELINE_S3_ENDPOINT:-}" ]; then
  log_pass "External S3-compatible endpoint provided via environment (AWS_S3_ENDPOINT/PIPELINE_S3_ENDPOINT)"
else
  log_optional "No object storage configured yet in '${NAMESPACE}' -- run 'make storage' to deploy a namespace-local MinIO (needed by 'make pipeline-server')"
fi

log_section "PIPELINE SERVER"
if oc get crd datasciencepipelinesapplications.datasciencepipelinesapplications.opendatahub.io >/dev/null 2>&1; then
  log_pass "CRD present: DataSciencePipelinesApplication (Data Science Pipelines component installed)"
else
  log_fail "CRD missing: DataSciencePipelinesApplication -- enable the 'datasciencepipelines' component in the DataScienceCluster"
fi
DSPA_IN_NAMESPACE=$(oc get datasciencepipelinesapplication -n "${NAMESPACE}" -o name 2>/dev/null || true)
if [ -n "$DSPA_IN_NAMESPACE" ]; then
  log_pass "A DataSciencePipelinesApplication already exists in namespace '${NAMESPACE}': ${DSPA_IN_NAMESPACE}"
else
  log_optional "No DataSciencePipelinesApplication found in namespace '${NAMESPACE}' yet -- run 'make storage' then 'make pipeline-server' (needs object storage above to be PASS first)"
fi

log_section "DSPA STATUS"
if [ -n "$DSPA_IN_NAMESPACE" ]; then
  DSPA_READY=$(oc get datasciencepipelinesapplication -n "${NAMESPACE}" -o jsonpath='{.items[0].status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || true)
  if [ "$DSPA_READY" = "True" ]; then
    log_pass "DataSciencePipelinesApplication reports Ready=True -- 'make pipeline' can run"
  else
    log_warn "DataSciencePipelinesApplication exists but Ready condition is '${DSPA_READY:-unknown}' -- do not run 'make pipeline' yet"
  fi
else
  log_optional "No DataSciencePipelinesApplication to check yet (see PIPELINE SERVER above)"
fi

log_section "MLflow (optional)"
DETECTED_MLFLOW_URI=$(detect_mlflow_uri)
if [ -n "$DETECTED_MLFLOW_URI" ]; then
  # A Route existing doesn't mean the server behind it actually responds -- check this
  # demo's own Deployment readiness too when it's the one that created the Route.
  if oc get deployment mlflow -n "${NAMESPACE}" >/dev/null 2>&1; then
    MLFLOW_DEPLOY_READY=$(oc get deployment mlflow -n "${NAMESPACE}" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo 0)
    if [ "${MLFLOW_DEPLOY_READY:-0}" -ge 1 ] 2>/dev/null; then
      log_optional "MLflow is available and Ready at ${DETECTED_MLFLOW_URI} -- training/evaluation/pipeline will log to it"
    else
      log_warn "MLflow Route exists (${DETECTED_MLFLOW_URI}) but deployment/mlflow has 0 ready replicas -- training/evaluation will try to log to it and silently skip tracking if it can't connect (check: oc describe pod -l app.kubernetes.io/name=mlflow -n ${NAMESPACE})"
    fi
  else
    log_optional "MLflow is available at ${DETECTED_MLFLOW_URI} (externally managed) -- training/evaluation/pipeline will log to it"
  fi
else
  log_optional "No MLflow detected in '${NAMESPACE}' -- run 'make mlflow' to deploy a lightweight namespace-local instance, or export MLFLOW_TRACKING_URI to point at an existing one. Training/evaluation work identically without it."
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
