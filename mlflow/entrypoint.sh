#!/usr/bin/env bash
# Starts the MLflow Tracking Server using only environment variables injected by
# manifests/mlflow.yaml -- nothing here is hardcoded to a sandbox, bucket name, or
# credential. Backend store is a small SQLite file on a PVC (fine for a demo: one
# tracking server replica, no concurrent-writer requirement); artifacts go to the
# namespace-local MinIO bucket via the S3 API.
set -euo pipefail

: "${MLFLOW_BACKEND_STORE_URI:=sqlite:////data/mlflow.db}"
: "${MLFLOW_ARTIFACT_ROOT:=s3://mlflow/artifacts}"
: "${MLFLOW_PORT:=8080}"

mkdir -p "$(dirname "${MLFLOW_BACKEND_STORE_URI#sqlite:///}")" 2>/dev/null || true

exec mlflow server \
  --host 0.0.0.0 \
  --port "${MLFLOW_PORT}" \
  --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
  --default-artifact-root "${MLFLOW_ARTIFACT_ROOT}" \
  --serve-artifacts
