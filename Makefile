# RHOAI PyTorch Trainer demo -- reproducible on any new OpenShift AI sandbox.
#
# Nothing here hardcodes a sandbox hostname, cluster API, username, token, namespace,
# route, GPU count, or storage class. Every variable below has a sane default and can be
# overridden either on the command line (`make train TRAIN_NODES=3`) or via environment
# (`export TRAIN_NODES=3; make train`).
SHELL := /usr/bin/env bash

NAMESPACE          ?= rhoai-training-demo
IMAGE_STREAM_NAME  ?= pytorch-trainer-demo
IMAGE_TAG          ?= latest
TRAINJOB_NAME      ?= pytorch-trainer-demo
TRAIN_NODES        ?= 2
MODE               ?=
GPU_PER_NODE       ?=
TRAIN_CPU          ?= 250m
TRAIN_MEMORY       ?= 768Mi
TRAIN_EPOCHS       ?= 5
TRAIN_LR           ?= 0.01
TRAIN_BATCH_SIZE   ?= 32
PIPELINE_NAME      ?= pytorch-trainer-demo-pipeline
SCHEDULE_CRON      ?= 0 2 * * 0
SCHEDULE_NAME      ?= pytorch-trainer-demo-nightly
MLFLOW_TRACKING_URI       ?=
MLFLOW_EXPERIMENT_NAME    ?= rhoai-pytorch-trainer-demo
USE_MLFLOW         ?= auto
MINIO_STORAGE_SIZE ?= 5Gi
PIPELINE_BUCKET    ?= mlpipeline
MLFLOW_BUCKET      ?= mlflow
CHECKPOINT_BUCKET  ?= checkpoints
DSPA_NAME          ?= dspa
MLFLOW_IMAGE_STREAM_NAME ?= pytorch-trainer-demo-mlflow

export NAMESPACE IMAGE_STREAM_NAME IMAGE_TAG TRAINJOB_NAME TRAIN_NODES MODE GPU_PER_NODE \
	TRAIN_CPU TRAIN_MEMORY TRAIN_EPOCHS TRAIN_LR TRAIN_BATCH_SIZE PIPELINE_NAME \
	SCHEDULE_CRON SCHEDULE_NAME MLFLOW_TRACKING_URI MLFLOW_EXPERIMENT_NAME USE_MLFLOW \
	MINIO_STORAGE_SIZE PIPELINE_BUCKET MLFLOW_BUCKET CHECKPOINT_BUCKET DSPA_NAME \
	MLFLOW_IMAGE_STREAM_NAME

VENV_PYTHON ?= .venv/bin/python
VENV_PIP    ?= .venv/bin/pip

.PHONY: help install preflight bootstrap storage pipeline-server bootstrap-pipelines mlflow \
	build train compile-pipeline pipeline pipeline-upload pipeline-run pipeline-status \
	schedule schedule-status unschedule create-schedule delete-schedule \
	validate security-check status cleanup demo demo-reset notebook lint test-negative

help:
	@echo "RHOAI PyTorch Trainer demo -- available targets:"
	@echo ""
	@echo "  make preflight         Read-only capability check of the current cluster"
	@echo "  make bootstrap         Create the namespace + RBAC (idempotent)"
	@echo "  make storage           Deploy namespace-local MinIO (S3-compatible object storage)"
	@echo "  make pipeline-server   Create the DataSciencePipelinesApplication, wait for Ready"
	@echo "  make mlflow            (optional) Build + deploy a lightweight MLflow tracking server"
	@echo "  make build             Build the training image via an OpenShift Build"
	@echo "  make train             Run a distributed TrainJob directly (MODE=cpu|gpu)"
	@echo "  make compile-pipeline  Compile pipeline/pipeline.py -> pipeline/pipeline.yaml"
	@echo "  make pipeline          Upload + run the AI Pipeline (requires a Pipeline Server)"
	@echo "  make pipeline-status   Check the latest pipeline run's state"
	@echo "  make schedule          Create a recurring (scheduled) pipeline run"
	@echo "  make schedule-status   Show whether the recurring run exists"
	@echo "  make unschedule        Delete the recurring pipeline run"
	@echo "  make validate          Verify namespace/image/TrainJob/pipeline state (PASS/FAIL)"
	@echo "  make security-check    Scan the working tree for tokens/secrets before pushing"
	@echo "  make status            Full CLUSTER/TRAINING/PIPELINES/MLFLOW status table"
	@echo "  make cleanup           Delete ONLY this demo's namespace and resources"
	@echo "  make demo              preflight -> bootstrap -> storage -> pipeline-server -> build -> pipeline -> validate -> status"
	@echo "  make demo-reset        cleanup, then re-run 'make demo' from scratch"
	@echo "  make install           Create a local .venv with dev/tooling dependencies"
	@echo "  make notebook          Launch the exploration notebook locally with Jupyter"
	@echo "  make lint              Lint training/pipeline/scripts with ruff"
	@echo ""
	@echo "All variables have defaults and can be overridden, e.g.:"
	@echo "  export NAMESPACE=my-demo"
	@echo "  TRAIN_NODES=3 MODE=gpu make train"

install:
	python3 -m venv .venv
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r requirements-dev.txt

lint:
	$(VENV_PYTHON) -m ruff check training pipeline scripts

preflight:
	./scripts/preflight.sh

bootstrap:
	./scripts/bootstrap.sh

storage:
	./scripts/storage.sh

pipeline-server: storage
	./scripts/pipeline-server.sh

bootstrap-pipelines: pipeline-server

mlflow:
	./scripts/mlflow.sh

build:
	./scripts/build-training-image.sh

train:
	./scripts/run-trainjob.sh

compile-pipeline:
	$(VENV_PYTHON) pipeline/pipeline.py

pipeline: compile-pipeline
	./scripts/deploy-pipeline.sh run

pipeline-upload: compile-pipeline
	./scripts/deploy-pipeline.sh upload

pipeline-run:
	./scripts/deploy-pipeline.sh run

pipeline-status:
	./scripts/deploy-pipeline.sh status

schedule: compile-pipeline
	./scripts/deploy-pipeline.sh create-schedule

schedule-status:
	./scripts/deploy-pipeline.sh schedule-status

unschedule:
	./scripts/deploy-pipeline.sh delete-schedule

# Kept as aliases: the original names from the first iteration of this demo.
create-schedule: schedule
delete-schedule: unschedule

validate:
	./scripts/validate.sh

security-check:
	./scripts/security-check.sh

test-negative:
	./scripts/negative-tests.sh

status:
	./scripts/status.sh

cleanup:
	./scripts/cleanup.sh

demo:
	./scripts/preflight.sh
	./scripts/bootstrap.sh
	./scripts/build-training-image.sh
	./scripts/run-trainjob.sh
	@echo ""
	@if $(MAKE) --no-print-directory storage && $(MAKE) --no-print-directory pipeline-server; then \
		$(MAKE) --no-print-directory pipeline || echo "WARN Pipeline run did not succeed -- see output above (core TrainJob demo above already succeeded)"; \
	else \
		echo "WARN Object storage / Pipeline Server not available -- skipping the AI Pipeline stage (see README.md 'Pipeline Server')"; \
	fi
	./scripts/validate.sh
	./scripts/status.sh

demo-reset:
	@echo "== demo-reset: deleting this demo's namespace, then rebuilding from scratch =="
	CONFIRM=yes ./scripts/cleanup.sh
	$(MAKE) --no-print-directory demo

notebook:
	$(VENV_PYTHON) -m jupyter notebook notebooks/exploration.ipynb
