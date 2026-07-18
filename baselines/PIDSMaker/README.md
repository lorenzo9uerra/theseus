# Orthrus and Velox Baselines Implementation

> Modified version of the [PIDSMaker framework](https://github.com/ubc-provenance/PIDSMaker/) (commit `2a73886`) used to reproduce the **Orthrus** and **Velox** baselines for our paper. PIDSMaker is the original framework developed by the authors of these models.

## Key Modifications

*   **Removal of Postgres & Docker**: Data processing now uses **Polars** on Parquet/CSV files, removing the need for containerization.
*   **Evaluation Protocol**:
    *   **Process-only Evaluation**: Restricted evaluation to process nodes to align with the REAPR ground truth.
    *   **Checkpoint Selection**: Early stopping and checkpoint selection based on AP on the validation set.
    *   **Validation-based Threshold Setting**: Threshold set to the maximum benign validation score, then applied unchanged to test set.
    *   **Artifact Removal**: Removed CADETS collection artifacts as described in ["What We Talk About When We Talk About Logs"](https://ieeexplore.ieee.org/document/11023260).
*   **Reported Metrics**:
    *   AP, AUROC, MCC, F1, ADP, FPR.
    *   For paper reproduction, use the **Strict Attack Chain** metrics, where contaminated nodes are excluded from metric accounting.

## Prerequisites

| Requirement | Version |
|---|---|
| [uv](https://docs.astral.sh/uv/) | any recent |
| CUDA (optional, for GPU) | 11.7+ |

## Directory Layout

```
PROJECT_ROOT/
├── data/DARPA/                        # Raw DARPA TC E3 data (CSV/Parquet)
│   ├── CADETS_E3/
│   ├── FIVEDIRECTIONS_E3/
│   ├── THEIA_E3/
│   └── TRACE_E3/
├── ground_truth/                      # REAPR ground-truth labels
└── baselines/PIDSMaker/              # <- you are here
    ├── Makefile                       # One-command reproducibility
    ├── pyproject.toml                 # uv project definition + package metadata
    ├── .python-version                # Pins Python 3.9
    ├── ground_truth -> ../../ground_truth  # Symlink (must exist)
    ├── artifacts/                     # Generated: pipeline intermediate files
    └── results/                       # Generated: evaluation logs
```

## Using Zenodo Artifacts (Evaluation-only)

If you downloaded `pidsmaker_artifacts.tar.gz` from our Zenodo reproducibility repository (DOI: https://doi.org/10.5281/zenodo.19844784), you can reproduce the evaluation without re-running the full PIDSMaker pipeline. The bundled `results/` logs are sanitized release copies, with absolute local paths and wall-clock timestamps removed.

From `PROJECT_ROOT/`:

```bash
tar -xzf pidsmaker_artifacts.tar.gz -C baselines/PIDSMaker/
```

Then, from `baselines/PIDSMaker/`:

```bash
make setup
ln -sf ../../data data
ln -sf ../../ground_truth ground_truth

make eval            # Orthrus + Velox, eval-only (skip training)
```

`make eval` still expects the processed E3 tables to be available at
`PROJECT_ROOT/data/DARPA/`, because strict-label reconstruction maps REAPr UUIDs back to
node ids through the node tables even in evaluation-only mode.

For the separate ATLASv2 secondary-benchmark bundle, the same eval-only path works with:

* `CSV_BASE_DIR=../../atlas_bundle/data/ATLASV2`
* `--artifact_dir ../../atlas_bundle/pidsmaker/artifacts`

The repository-level helper `scripts/reproduce_atlasv2_results.sh` wires those paths for the reported ATLASv2 runs.

If the extracted bundle already contains `results/`, you can skip `make eval` and aggregate directly:

```bash
uv run python scripts/aggregate_results.py
```

If you are running on a cluster where `uv` is available on the login node but not on the
worker nodes, create the environment once with `make setup`, then invoke the interpreter
directly from the built virtual environment:

```bash
./.venv/bin/python pidsmaker/main.py orthrus CADETS_E3 --tuned --force_restart evaluation \
    --detection.gnn_training.seed 111 --csv_base_dir ../../data/DARPA
./.venv/bin/python scripts/aggregate_results.py
```

## Quick Start

```bash
# 1. Install (one-time)
make setup

# 2. Create symlinks (if not already present)
ln -sf ../../data data
ln -sf ../../ground_truth ground_truth

# 3. Run everything: Orthrus + Velox, 4 datasets × 5 seeds each
make all
```

## Step-by-Step Reproduction

### 1. Environment Setup

```bash
make setup
```

Or manually:
```bash
uv sync
```

If `uv` is not available on the execution node, use `make setup` or `uv sync` once from
an interactive shell, then run later commands with `./.venv/bin/python ...`.

### 2. Symlinks

The framework needs `ground_truth/` and expects data at `./data/DARPA/<DATASET>/` by default. Create symlinks from the project root:

```bash
ln -sf ../../data data
ln -sf ../../ground_truth ground_truth
```

Or pass `--csv_base_dir` / `CSV_BASE_DIR` to override the data path.

### 3. Run Experiments

**All at once:**
```bash
make all              # Orthrus + Velox, all datasets, all seeds
```

**By system:**
```bash
make orthrus          # All datasets, all seeds
make velox            # All datasets, all seeds
```

**Re-evaluate only (skip training):**
```bash
make eval            # Orthrus + Velox
make eval-orthrus
make eval-velox
```

These evaluation-only targets still require the processed E3 tables at
`PROJECT_ROOT/data/DARPA/`, because the evaluation code reconstructs strict labels from
UUID-to-node-id mappings stored in the node tables.

**Single run:**
```bash
export PYTHONHASHSEED=0
uv run python pidsmaker/main.py orthrus CADETS_E3 --tuned --restart_from_scratch \
    --detection.gnn_training.seed 111  --csv_base_dir ../../data/DARPA
# or, if uv is not on PATH on the execution node:
./.venv/bin/python pidsmaker/main.py orthrus CADETS_E3 --tuned --restart_from_scratch \
    --detection.gnn_training.seed 111 --csv_base_dir ../../data/DARPA
```

`PYTHONHASHSEED=0` ensures deterministic Word2Vec training; without it, Python's hash randomization causes non-deterministic token ordering during vocabulary building.

### 4. Results

Logs are saved to `results/<system>_<dataset>_seed<seed>.log`.

For paper reproduction, use the **Strict Attack Chain** metrics from each log,
with contaminated nodes excluded from metric accounting.

Key metrics: AP, AUROC, MCC, F1, ADP, FPR.

## Aggregating Results

Aggregate the per-seed logs in `results/` into mean ± std tables:

```bash
uv run python scripts/aggregate_results.py
# or: uv run python scripts/aggregate_results.py --log_dir results
# or, if uv is not on PATH on the execution node:
./.venv/bin/python scripts/aggregate_results.py --log_dir results
```

## Customisation

| Variable | Default | Description |
|---|---|---|
| `CSV_BASE_DIR` | `PROJECT_ROOT/data/DARPA` | Raw data location |
| `DATASETS` | `CADETS_E3 FIVEDIRECTIONS_E3 THEIA_E3 TRACE_E3` | Datasets to process |
| `SEEDS` | `111 333 828 0 433` | Seeds for GNN training |
| `DEVICE` | *(unset)* | Optional: set `CUDA_VISIBLE_DEVICES` (e.g., `DEVICE=0`) |
| `USE_CPU` | `0` | Set to `1` to force CPU (`--cpu`) |
| `WANDB` | *(unset)* | Set to 1 to enable Weights & Biases logging |

```bash
make orthrus DATASETS="CADETS_E3" SEEDS="111 333" DEVICE=0 WANDB=1
make orthrus DATASETS="CADETS_E3" SEEDS="111 333" USE_CPU=1 WANDB=1
```

## Troubleshooting

On HPC clusters, if you encounter `CXXABI` errors:
```bash
export LD_LIBRARY_PATH=$(uv run python -c 'import sys; print(sys.prefix)')/lib:$LD_LIBRARY_PATH
```

## Cleanup

```bash
make clean   # Remove results/
```

## Note on Compatibility

Our modifications target **Orthrus** and **Velox** pipelines only. Other PIDSMaker modules/configurations are not guaranteed to work in this fork.
