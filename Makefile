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
GPU_PER_NODE       ?= 0
TRAIN_CPU          ?= 1
TRAIN_MEMORY       ?= 2Gi
TRAIN_EPOCHS       ?= 5
TRAIN_LR           ?= 0.01
TRAIN_BATCH_SIZE   ?= 32
PIPELINE_NAME      ?= pytorch-trainer-demo-pipeline
SCHEDULE_CRON      ?= 0 2 * * *
SCHEDULE_NAME      ?= pytorch-trainer-demo-nightly
MLFLOW_TRACKING_URI       ?=
MLFLOW_EXPERIMENT_NAME    ?= rhoai-pytorch-trainer-demo

export NAMESPACE IMAGE_STREAM_NAME IMAGE_TAG TRAINJOB_NAME TRAIN_NODES GPU_PER_NODE \
	TRAIN_CPU TRAIN_MEMORY TRAIN_EPOCHS TRAIN_LR TRAIN_BATCH_SIZE PIPELINE_NAME \
	SCHEDULE_CRON SCHEDULE_NAME MLFLOW_TRACKING_URI MLFLOW_EXPERIMENT_NAME

VENV_PYTHON ?= .venv/bin/python
VENV_PIP    ?= .venv/bin/pip

.PHONY: help install preflight bootstrap build train compile-pipeline pipeline \
	pipeline-upload pipeline-run pipeline-status create-schedule delete-schedule \
	validate status cleanup demo notebook lint

help:
	@echo "RHOAI PyTorch Trainer demo -- available targets:"
	@echo ""
	@echo "  make preflight         Read-only capability check of the current cluster"
	@echo "  make bootstrap         Create the namespace + RBAC (idempotent)"
	@echo "  make build             Build the training image via an OpenShift Build"
	@echo "  make train             Run a distributed TrainJob directly (no pipeline)"
	@echo "  make compile-pipeline  Compile pipeline/pipeline.py -> pipeline/pipeline.yaml"
	@echo "  make pipeline          Upload + run the AI Pipeline (requires a Pipeline Server)"
	@echo "  make pipeline-status   Check the latest pipeline run's state"
	@echo "  make create-schedule   Create a recurring (scheduled) pipeline run"
	@echo "  make delete-schedule   Delete the recurring pipeline run"
	@echo "  make validate          Verify namespace/image/TrainJob/pipeline state (PASS/FAIL)"
	@echo "  make status            Quick human-readable snapshot of demo resources"
	@echo "  make cleanup           Delete ONLY this demo's namespace and resources"
	@echo "  make demo              Run preflight -> bootstrap -> build -> train -> pipeline -> validate"
	@echo "  make install           Create a local .venv with dev/tooling dependencies"
	@echo "  make notebook          Launch the exploration notebook locally with Jupyter"
	@echo "  make lint              Lint training/pipeline/scripts with ruff"
	@echo ""
	@echo "All variables have defaults and can be overridden, e.g.:"
	@echo "  export NAMESPACE=my-demo"
	@echo "  TRAIN_NODES=3 GPU_PER_NODE=1 make train"

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

create-schedule: compile-pipeline
	./scripts/deploy-pipeline.sh create-schedule

delete-schedule:
	./scripts/deploy-pipeline.sh delete-schedule

validate:
	./scripts/validate.sh

status:
	@echo "Namespace:  $(NAMESPACE)"
	@oc get namespace $(NAMESPACE) --no-headers 2>/dev/null || echo "  (namespace not found -- run 'make bootstrap')"
	@echo ""
	@echo "Training image:"
	@oc get imagestream $(IMAGE_STREAM_NAME) -n $(NAMESPACE) --no-headers 2>/dev/null || echo "  (not built -- run 'make build')"
	@echo ""
	@echo "TrainJob:"
	@oc get trainjob -n $(NAMESPACE) -l app.kubernetes.io/part-of=rhoai-pytorch-trainer-demo 2>/dev/null || echo "  (none -- run 'make train')"
	@echo ""
	@echo "Pipeline server:"
	@python3 scripts/pipeline_client.py detect 2>/dev/null || echo "  NOT AVAILABLE"

cleanup:
	./scripts/cleanup.sh

demo:
	./scripts/preflight.sh
	./scripts/bootstrap.sh
	./scripts/build-training-image.sh
	./scripts/run-trainjob.sh
	@if python3 scripts/pipeline_client.py detect >/dev/null 2>&1; then \
		$(MAKE) pipeline; \
	else \
		echo "Pipeline Server not available -- skipping the AI Pipeline stage (see README.md 'Pipeline Server')"; \
	fi
	./scripts/validate.sh

notebook:
	$(VENV_PYTHON) -m jupyter notebook notebooks/exploration.ipynb
