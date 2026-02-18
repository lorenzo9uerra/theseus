# Magic Baseline Implementation

> Reproducible pipeline for training and evaluating the Magic baseline on DARPA TC Engagement 3 datasets.

## Prerequisites

| Requirement | Version |
|---|---|
| [uv](https://docs.astral.sh/uv/) | any recent |
| CUDA (optional, for GPU) | 11.6+ |

## Directory Layout

```
PROJECT_ROOT/
├── data/DARPA/                        # Raw DARPA TC E3 data (Parquet/CSV)
│   ├── CADETS_E3/
│   ├── FIVEDIRECTIONS_E3/
│   ├── THEIA_E3/
│   └── TRACE_E3/
├── ground_truth/reapr-ground-truth/
│   └── darpa-tc-engagement3/          # REAPR ground-truth labels
│       ├── cadets_labels.csv
│       ├── fivedirections_labels.csv
│       ├── theia_labels.csv
│       └── trace_labels.csv
└── baselines/MAGIC/                   # <- you are here
    ├── Makefile                       # One-command reproducibility
    ├── pyproject.toml                 # uv project definition
    ├── .python-version                # Pins Python 3.9
    ├── data/                          # Generated: processed DGL graphs
    ├── checkpoints/                   # Generated: trained model weights
    └── results/                       # Generated: evaluation logs
```

## Using Zenodo Artifacts (Evaluation-only)

If you downloaded `magic_artifacts.zip` from our Zenodo reproducibility repository (DOI: https://doi.org/10.5281/zenodo.18489504), you can reproduce the evaluation without re-parsing data or re-training the model.

From `PROJECT_ROOT/`:

```bash
unzip -o magic_artifacts.zip -d baselines/MAGIC/
```

Then, from `baselines/MAGIC/`:

```bash
make setup
make eval
```

If the extracted bundle already contains `results/`, you can skip `make eval` and aggregate directly:

```bash
uv run python utils/aggregate_results.py
```

## Quick Start (One Command)

From the `baselines/MAGIC/` directory:

```bash
# 1. Create the uv environment (one-time)
make setup

# 2. Run the full pipeline: data -> train -> eval
make all
```

This will:

1. Parse the raw DARPA datasets into daily DGL graphs (-> `data/<dataset>/`).
2. Train MAGIC on each dataset with seeds {71, 83, 232, 441, 915} (-> `checkpoints/`).
3. Evaluate every checkpoint and save logs (-> `results/`).

## Step-by-Step Reproduction

### 1. Environment Setup

```bash
make setup
```

Or manually:
```bash
uv sync
```

### 2. Data Preparation

```bash
make data
```

Parses each DARPA TC E3 dataset into daily graph snapshots (DGL format) with two-level ground-truth labels. Expects raw data in `PROJECT_ROOT/data/DARPA/` and ground truth in `PROJECT_ROOT/ground_truth/reapr-ground-truth/darpa-tc-engagement3/`.

Override paths if your data is elsewhere:
```bash
make data DATA_DIR=/path/to/DARPA GROUND_TRUTH_DIR=/path/to/labels
```

To process a single dataset manually:
```bash
uv run python utils/data_parser_daily.py \
    --dataset cadets \
    --data_dir ../../data/DARPA/CADETS_E3 \
    --ground_truth ../../ground_truth/reapr-ground-truth/darpa-tc-engagement3/cadets_labels.csv \
    --label_filter both
```

### 3. Training

```bash
make train           # all datasets, all seeds
make train DEVICE=0  # use GPU 0
```

Or a single run:
```bash
uv run python train.py --dataset cadets --seed 71 --device 0
```

**Arguments:**

| Flag | Default | Description |
|---|---|---|
| `--dataset` | *required* | `cadets`, `fivedirections`, `theia`, `trace` |
| `--seed` | 42 | Random seed |
| `--device` | -1 | GPU id (-1 = CPU) |
| `--lr` | 0.0018 | Learning rate |
| `--weight_decay` | 5e-4 | Weight decay |
| `--patience` | 10 | Early stopping patience (epochs) |
| `--mask_rate` | 0.5 | Masking rate |
| `--alpha_l` | 3 | Power index for SCE loss |
| `--optimizer` | adam | `adam` or `sgd` |
| `--loss_fn` | sce | `sce` or `bce` |
| `--pooling` | mean | `mean` or `sum` |
| `--wandb` | off | Enable W&B logging |

### 4. Evaluation

```bash
make eval            # all datasets, all seeds
make eval DEVICE=0   # use GPU 0
```

Or a single run:
```bash
uv run python eval.py --dataset cadets --seed 71 --device 0
```

Results are written to `results/<dataset>_seed<seed>.log`.

## Expected Output

Each evaluation log contains two-level metrics:

- **Level 1 (Causal Scope):** attack + contaminated nodes as positives.
- **Level 2 (Strict Attack Chain):** only attack nodes as positives.

Key metrics: F1, PR-AUC, FPR, MCC, ADP (Attack Detection Precision).

## Aggregating Results

Aggregate the per-seed logs in `results/` into mean ± std tables:

```bash
uv run python utils/aggregate_results.py
# or: uv run python utils/aggregate_results.py --log_dir results
```

## Customisation

| Variable | Default | Description |
|---|---|---|
| `DATA_DIR` | `PROJECT_ROOT/data/DARPA` | Raw data location |
| `GROUND_TRUTH_DIR` | `PROJECT_ROOT/ground_truth/...` | REAPR labels |
| `DEVICE` | -1 | GPU device id (-1 = CPU) |
| `SEEDS` | 71 83 232 441 915 | Seeds to train/eval |
| `DATASETS` | cadets fivedirections theia trace | Datasets to process |
| `WANDB` | *(unset)* | Set to 1 to enable Weights & Biases logging |

Example:
```bash
make train DATASETS=cadets SEEDS="71 83" DEVICE=0 WANDB=1
```

## Cleanup

```bash
make clean   # Remove data/, checkpoints/, results/, eval_result/
```
