# Demo Guide

A presenter walkthrough for the RHOAI PyTorch Trainer demo, written for **OpenShift AI
3.5**. This is meant to be read alongside a terminal and the OpenShift AI Dashboard, on a
sandbox that has already run `make demo` (or at minimum `make preflight`, `make
bootstrap`, `make build`, `make train`).

> **On dashboard menu names:** every click path below is taken from Red Hat's own
> published OpenShift AI **3.5** product documentation (`docs.redhat.com`,
> "Working with AI pipelines" and "Working with distributed workloads" guides), not
> guessed. Two pages in Red Hat's own 3.5 docs disagree on one single label (the page that
> shows live `TrainJob` progress metrics is called **"Model training"** on one doc page and
> **"TrainingJobs"** on another, within the same 3.5 documentation set) -- so treat that one
> item as "look for a left-nav entry along those lines" and confirm the exact wording live
> in your own sandbox before presenting. Every other path below is stated consistently
> across Red Hat's docs.

## The 30-second pitch

> *"What's the best practice after the notebook works?"*

"You extract the training logic from the notebook into a plain Python script, build it
into a container image straight from your Git repository with an OpenShift Build -- no
manual copying, no hand-edited images -- and hand that image to Kubeflow Trainer as a
`TrainJob`, which is the Kubernetes-native way to run it distributed across multiple
GPU or CPU workers. You then wrap `prepare -> train -> evaluate` in an OpenShift AI
Pipeline so it's a repeatable, schedulable, auditable workflow instead of a notebook
someone has to remember to re-run by hand, and you track every run's parameters and
metrics in MLflow so you can compare them later. The notebook stays exactly what it's
good at: exploration. It never becomes the production workload."

## The 5-minute pitch

Suggested prop order: a terminal (with `oc` logged in and this repo checked out) next to
the OpenShift AI Dashboard in a browser tab.

### 0:00 -- Notebook

Open `notebooks/exploration.ipynb`, run the last two or three cells live.

> "We start here, in the exploration phase. Synthetic dataset, a small PyTorch model, a
> few epochs, a loss curve. This is where a data scientist iterates fast, and that's
> exactly what a notebook is good at."

### 0:45 -- GitHub / `training/train.py`

Show `training/train.py` side-by-side with the notebook's last cell.

> "Once the approach is validated, we don't keep training in the notebook -- we extract the
> *exact same* modeling logic into a plain script, `training/train.py`, versioned in this
> GitHub repository. It's parameterized by environment variables (`RANK`, `WORLD_SIZE`,
> `LOCAL_RANK`) so it runs identically as 1 process on my laptop or N processes on the
> cluster."

### 1:30 -- OpenShift Build -> training image -> `TrainJob`

```bash
oc get imagestreamtag pytorch-trainer-demo:latest -n "$NAMESPACE" -o jsonpath='{.metadata.annotations.demo\.git\.sha}'
oc get trainjob -n "$NAMESPACE" -o yaml
```

> "This image was built directly from that GitHub repository by an OpenShift `Build` -- no
> private registry, no hand-pushed image. This annotation is the exact Git commit SHA it
> was built from, so I can always say 'this training corresponds to exactly this commit.'
> The `TrainJob` describes *how* to run it: which image, how many nodes, how much CPU/GPU
> per node. It references a platform-provided `ClusterTrainingRuntime` -- it doesn't
> reimplement `torchrun` wiring itself."

### 2:15 -- Workers / Kubeflow Trainer

```bash
oc get pods -n "$NAMESPACE" -l jobset.sigs.k8s.io/jobset-name=pytorch-trainer-demo
oc logs <worker-0-pod> -n "$NAMESPACE" | grep rank=
oc logs <worker-1-pod> -n "$NAMESPACE" | grep rank=
```

> "Kubeflow Trainer just created these pods -- one per node -- and coordinated them into a
> single PyTorch distributed job. Watch: rank 0 in this pod, rank 1 in that one, same
> `world_size`, same epoch, training the same model together."

### 3:00 -- AI Pipeline

Dashboard: **Projects** -> `rhoai-training-demo` -> **Pipelines** tab, then **Develop &
train** -> **Experiments** -> the experiment for this pipeline -> **Runs**.

> "Now the same three steps -- prepare data, train, evaluate -- are wrapped in an AI
> Pipeline. `distributed-training` in this graph creates the *exact same kind* of `TrainJob`
> I just showed you by hand, through the Kubernetes API, and waits for it to actually
> finish before evaluation runs. Nothing here is simulated -- if I ran `oc get trainjobs`
> right now, in parallel, I'd see it."

### 4:00 -- Schedule

```bash
make schedule
make schedule-status
```

Dashboard: same **Runs** page -> **Schedules** tab.

> "This same pipeline can be triggered on a cron schedule -- here's a weekly one as an
> example -- so retraining happens automatically, without anyone re-running a notebook by
> hand." (Then: `make unschedule` right after, so the sandbox doesn't keep retraining
> itself unattended.)

### 4:30 -- MLflow

> "Every run -- whether triggered by hand, by the pipeline, or by the schedule -- logs its
> learning rate, epochs, batch size, world size, and loss to MLflow, tagged with the
> `TrainJob` name and the Git commit. So here's run A with `lr=0.01`, and here's run B with
> `lr=0.05` -- same code, same commit, different hyperparameters, easy to compare."

### 5:00 -- Summary

> "Notebook for exploration. Script + image for reproducibility. `TrainJob` +  Kubeflow
> Trainer for distributed execution. AI Pipeline for orchestration and scheduling. MLflow
> for tracking. Every piece of this is defined in the GitHub repository -- delete the
> namespace, clone the repo on a new sandbox, run `make demo`, and it all comes back."

## Notebook -> production: what to say

`notebooks/exploration.ipynb` and `training/train.py` are deliberately kept close enough
in shape (same synthetic dataset generator, same tiny model class, same training loop
math) that the diff between them tells the whole story: nothing about the *modeling
logic* changes when it goes to production. What changes is:

| | Notebook | `training/train.py` |
|---|---|---|
| Where it runs | Interactively, one cell at a time | As a container, unattended |
| Parallelism | Single process | `RANK`/`WORLD_SIZE`/`LOCAL_RANK`-aware, N processes |
| Versioning | Cell execution order can drift from file order | A single Git commit, byte for byte |
| Deployment artifact | Not one (a notebook is not a container) | Built into an immutable image by an OpenShift `Build` |

The line to say out loud: **"We don't rewrite the model when it goes to production, we
extract it. The notebook doesn't become a production workload -- it becomes the historical
record of how we got to this script."**

## Dashboard walkthrough: WHAT TO CLICK / WHAT TO SHOW / WHAT TO SAY

### 1. The Pipeline Server

| WHAT TO CLICK | WHAT TO SHOW | WHAT TO SAY |
|---|---|---|
| **Projects** (left nav) -> `rhoai-training-demo` -> **Pipelines** tab | If not yet configured: a prompt with a **Configure pipeline server** button. Once configured: the pipeline list. | "Every project gets its own pipeline server, backed by S3-compatible object storage -- here that's a small MinIO instance this repo deploys just for this namespace, never anything cluster-wide." |

### 2. The pipeline run graph

| WHAT TO CLICK | WHAT TO SHOW | WHAT TO SAY |
|---|---|---|
| Left nav **Develop & train** -> **Experiments** -> select the project -> click the experiment (default name `rhoai-pytorch-trainer-demo` or `Default`) -> **Runs** tab -> click the latest run | The 3-node graph: `prepare-data` -> `distributed-training` -> `evaluate-model`, each with a green checkmark | "Three real steps. Click into `distributed-training` -- its logs show it calling the Kubernetes API to create a `TrainJob`, the same object type you saw me create by hand earlier." |
| Click the `distributed-training` node -> **Logs** tab | Log lines mentioning the `TrainJob` name it just created (e.g. `rhoai-demo-train-<timestamp>-<suffix>`) and its `Complete` condition | "Every run gets a unique `TrainJob` name -- timestamp plus a random suffix -- so a scheduled run firing while a manual run is still going never collides." |

### 3. Schedules

| WHAT TO CLICK | WHAT TO SHOW | WHAT TO SAY |
|---|---|---|
| Same **Runs** page -> **Schedules** tab (or **Develop & train** -> **Pipelines** -> **Runs** -> **Schedules** tab) | The recurring run created by `make schedule`, with its cron string and parameters | "Same pipeline, same parameters, triggered automatically -- this is what turns 'someone remembers to retrain the model' into 'the model retrains itself.'" |

### 4. The `TrainJob` / distributed training itself

Two options, in order of reliability -- **always confirm with `oc` first**, since the
dashboard's live progress view depends on optional progress-tracking annotations this
demo does not enable by default:

| WHAT TO CLICK / RUN | WHAT TO SHOW | WHAT TO SAY |
|---|---|---|
| `oc get trainjobs,jobsets,pods -n "$NAMESPACE"` | The `TrainJob`, its underlying `JobSet`, and N worker pods, all `Running`/`Completed` | "Kubeflow Trainer v2 uses `JobSet` under the hood to actually place and coordinate the worker pods." |
| `oc describe trainjob <name> -n "$NAMESPACE"` | `status.conditions` showing `Created` -> `Complete` | "This is the same status the pipeline step above polled before letting evaluation run." |
| `oc logs <worker-pod> -n "$NAMESPACE"` (x2, different pods) | Distinct `rank=0` / `rank=1` lines | "Two different processes, two different ranks, same job." |
| Left nav -> the page Red Hat's 3.5 docs call **"Model training"** on one page and **"TrainingJobs"** on another (confirm the exact label live) | Real-time progress metrics for the `TrainJob`, IF progress tracking was enabled for it | "OpenShift AI can also render training progress natively, without a terminal, when a job opts into progress tracking." *(If nothing shows here, don't improvise -- just say progress tracking wasn't enabled for this particular run, and fall back to the `oc` commands above, which always work.)* |
| Left nav **Observe & monitor** -> **Workload metrics** -> select the project -> **Distributed workload status** tab | A status table/graph of all distributed workloads (this `TrainJob`'s `JobSet`) in the project | "This is the cluster-wide view of every distributed workload in the project -- useful when there's more than just this one demo running." |

## The architecture diagram

```
GitHub
  |
  v
Training image (OpenShift Build)
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
           (GPU or CPU)       (GPU or CPU)
              +---- PyTorch distributed ----+
                       |
                       v
                     Model + metrics
                       |
                  MLflow (optional)
                       |
                       v
              Schedule (recurring run)
                       |
                       v
              New automatic training
```

On a CPU-only sandbox (or a GPU-sandbox where the GPU is already claimed by another
workload -- `make preflight` tells you which), replace "(GPU)" with "(CPU)" -- the
orchestration is identical; only the requested resource type changes
(`manifests/trainjob-gpu-example.yaml` has the GPU version for reference). This demo never
scales down or evicts another workload to free up a GPU for itself.

## Suggested timing if you have longer (10-12 minutes)

1. (1 min) Repository tour: `training/`, `manifests/`, `pipeline/`.
2. (2 min) Notebook: run the last two cells live, point at the `train.py` import.
3. (1 min) `oc get trainjob,pods -n $NAMESPACE` -- show the workers.
4. (2 min) `oc logs` on two different worker pods side by side -- show two different ranks.
5. (2 min) Dashboard: **Develop & train** -> **Experiments** -> the run's graph (3 steps)
   -> the `distributed-training` step logs (same `TrainJob` creation you just showed by
   hand).
6. (2 min) `make schedule`, show it in the Dashboard's **Schedules** tab, then
   `make unschedule`.
7. (1-2 min) MLflow UI if available: the run's parameters/metrics/artifact.
