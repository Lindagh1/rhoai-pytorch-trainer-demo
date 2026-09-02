# Demo Guide

A presenter walkthrough for the RHOAI PyTorch Trainer demo. This is meant to be read
alongside a terminal and the OpenShift AI Dashboard, on a fresh sandbox that has already
run `make preflight`, `make bootstrap`, `make build`, `make train`, and (if a Pipeline
Server is available) `make pipeline`.

## What I show

1. **The GitHub repository.** Everything is here: `training/train.py`,
   `training/Containerfile`, `manifests/`, `pipeline/pipeline.py`. This repository is the
   only source of truth -- the sandbox can be deleted and rebuilt from it.
2. **The notebook** (`notebooks/exploration.ipynb`). Synthetic dataset, small model, a few
   epochs, a loss curve, a validation check.
3. **`training/train.py`.** The exact same logic, extracted into a script that detects
   `RANK`/`WORLD_SIZE`/`LOCAL_RANK` and runs identically as 1 process or N.
4. **The `TrainJob`** (`oc get trainjob -n $NAMESPACE -o yaml`). Point at `runtimeRef`
   (references the platform-provided `ClusterTrainingRuntime`), `trainer.image` (built from
   this repo, no private registry), `trainer.numNodes`, and `trainer.resourcesPerNode`.
5. **The workers actually created** (`oc get pods -n $NAMESPACE`). Multiple pods, one per
   node, named `<trainjob-name>-node-<i>-0`.
6. **The distributed logs** (`oc logs <pod> -n $NAMESPACE`). Each pod prints its own
   `rank=`/`world_size=`/`epoch=`/`loss=` -- this is real coordination, not one process
   pretending to be many.
7. **The AI Pipeline** (OpenShift AI Dashboard -> Data Science Pipelines -> Runs, or
   `make pipeline-status`). Three real steps: `prepare-data`, `distributed-training`,
   `evaluate-model`.
8. **The pipeline run.** Point out that `distributed-training` creates the *same kind* of
   `TrainJob` shown in step 4-6 -- via the Kubernetes API, waiting for real completion.
9. **Scheduling** (`make create-schedule`, then the Dashboard's Runs -> Schedules view;
   `make delete-schedule` right after so it doesn't keep consuming resources).
10. **MLflow, if available.** Parameters, metrics (loss per epoch), and the model artifact
    for the run just shown.

## What I say

| Artifact | One-line message |
|---|---|
| Notebook | = exploration |
| Training image | = code versioned in GitHub, built reproducibly, never hand-edited on the cluster |
| `TrainJob` | = the *description* of a distributed training workload (image, command, node count, resources) -- not the code itself |
| Kubeflow Trainer | = orchestration of the PyTorch worker processes across nodes |
| AI Pipeline | = orchestration of the ML workflow (prepare -> train -> evaluate) around that `TrainJob` |
| Schedule | = automation over time, created deliberately and removed right after the demo |
| MLflow | = tracking of experiments, complementary to (not a replacement for) the pipeline |

## The diagram

```
GitHub
  |
  v
Training image
  |
  v
OpenShift AI Pipeline
  |
  +--------------------+--------------------+
  v                    v                    v
Prepare data       TrainJob              Evaluation
                       |
                  Kubeflow Trainer
                       |
              +--------+--------+
              v                 v
          Worker 0           Worker 1
           (GPU)               (GPU)
              +---- PyTorch ----+
                       |
                       v
                     Model
                       |
                  MLflow (optional)
```

On a CPU-only sandbox, replace "(GPU)" with "(CPU)" in the walkthrough -- the orchestration
shown is identical; only the requested resource type changes
(`manifests/trainjob-gpu-example.yaml` has the GPU version for reference).

## Suggested timing (10-12 minutes)

1. (1 min) Repository tour: `training/`, `manifests/`, `pipeline/`.
2. (2 min) Notebook: run the last two cells live, point at the `train.py` import.
3. (1 min) `oc get trainjob,pods -n $NAMESPACE` -- show the workers.
4. (2 min) `oc logs` on two different worker pods side by side -- show two different ranks.
5. (2 min) Dashboard: Data Science Pipelines -> the run's graph (3 steps) -> the
   `distributed-training` step logs (same TrainJob creation you just showed by hand).
6. (2 min) `make create-schedule`, show it in the Dashboard's Schedules tab, then
   `make delete-schedule`.
7. (1-2 min) MLflow UI if available: the run's parameters/metrics/artifact.
