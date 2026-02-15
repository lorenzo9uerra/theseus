# Orthrus and Velox Baselines Implementation

> Modified version of the [PIDSMaker framework](https://github.com/ubc-provenance/PIDSMaker/) (commit `2a73886`) used to reproduce the **Orthrus** and **Velox** baselines for our paper. PIDSMaker is the original framework developed by the authors of these models.

## Key Modifications

*   **Removal of Postgres & Docker**: Data processing now uses **Polars** on Parquet/CSV files, removing the need for containerization.
*   **Evaluation Protocol**:
    *   **Process-only Evaluation**: Restricted evaluation to process nodes to align with the REAPR ground truth.
    *   **Checkpoint Selection**: Early stopping and checkpoint selection based on PR-AUC on the validation set.
    *   **Validation-based Threshold Setting**: Threshold set to maximize MCC on validation set, then applied unchanged to test set.
    *   **Artifact Removal**: Removed CADETS collection artifacts as described in ["What We Talk About When We Talk About Logs"](https://ieeexplore.ieee.org/document/11023260).
*   **Metrics & Scenarios**:
    *   PR-AUC, MCC, F1, ADP, FNR.
    *   Two evaluation scenarios: **Strict Attack Chain** (contaminated nodes excluded) and **Causal Scope** (contaminated nodes labeled positive).

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
make eval-orthrus
make eval-velox
```

**Single run:**
```bash
export PYTHONHASHSEED=0
uv run python pidsmaker/main.py orthrus CADETS_E3 --tuned --restart_from_scratch \
    --detection.gnn_training.seed 111  --csv_base_dir ../../data/DARPA
```

`PYTHONHASHSEED=0` ensures deterministic Word2Vec training; without it, Python's hash randomization causes non-deterministic token ordering during vocabulary building.

### 4. Results

Logs are saved to `results/<system>_<dataset>_seed<seed>.log`.

Each log contains two evaluation scenarios:
- **Strict Attack Chain**: contaminated nodes excluded.
- **Causal Scope**: contaminated nodes labeled positive.

Key metrics: PR-AUC, MCC, F1, ADP, FNR.

## Aggregating Results

Aggregate the per-seed logs in `results/` into mean ± std tables:

```bash
uv run python scripts/aggregate_results.py
# or: uv run python scripts/aggregate_results.py --log_dir results
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
