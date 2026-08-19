# Magic Baseline Implementation

This fork adapts [MAGIC](https://github.com/FDUDSDE/MAGIC) at commit `aa0b647`
for the shared DARPA TC Engagement 3 evaluation protocol.

## Prerequisites

| Requirement | Version |
|---|---|
| [uv](https://docs.astral.sh/uv/) | any recent |
| CUDA (optional, for GPU) | 11.6+ |

## Directory Layout

```
PROJECT_ROOT/
├── data/DARPA/                        # Processed E3 node/event tables
│   ├── CADETS_E3/
│   ├── FIVEDIRECTIONS_E3/
│   ├── THEIA_E3/
│   └── TRACE_E3/
├── ground_truth/reapr-ground-truth/
│   └── darpa-tc-engagement3/          # REAPr ground-truth labels
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

## Evaluation from Zenodo Artifacts

The Zenodo [reproducibility artifacts](https://doi.org/10.5281/zenodo.21427594) provide processed Magic graphs, checkpoints, and canonical result logs in `magic_artifacts.tar.gz`.

From `PROJECT_ROOT/`:

```bash
tar -xzf magic_artifacts.tar.gz -C baselines/MAGIC/
```

Then, from `baselines/MAGIC/`:

```bash
make setup
make eval RESULT_DIR=results_rerun
uv run python utils/aggregate_results.py --log_dir results_rerun
```

This writes fresh evaluation logs to `results_rerun/` and preserves the canonical logs under `results/`.

If the extracted bundle already contains `results/`, you can skip `make eval` and aggregate directly:

```bash
uv run python utils/aggregate_results.py
```

If you are running on a cluster where `uv` is available on the login node but not on the
worker nodes, create the environment once with `make setup`, then invoke the interpreter
directly from the built virtual environment:

```bash
./.venv/bin/python eval.py --dataset cadets --seed 71
./.venv/bin/python utils/aggregate_results.py
```

## Full Pipeline

From the `baselines/MAGIC/` directory:

```bash
make setup
make all
```

This will:

1. Convert the processed E3 tables into daily DGL graphs.
2. Train Magic with seeds {71, 83, 232, 441, 915}.
3. Evaluate each checkpoint and write logs under `results/`.

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

### 2. Data Preparation

```bash
make data
```

Converts each processed E3 dataset into daily DGL graph snapshots with attack and contaminated labels. It expects node and event tables in `PROJECT_ROOT/data/DARPA/` and ground truth in `PROJECT_ROOT/ground_truth/reapr-ground-truth/darpa-tc-engagement3/`.

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

Evaluation defaults to CPU (`DEVICE=-1`), matching the original MAGIC
implementation. Set `DEVICE` explicitly only to use a compatible GPU.

```bash
make eval RESULT_DIR=results_rerun           # all datasets, all seeds
make eval RESULT_DIR=results_rerun DEVICE=0  # use GPU 0
```

Or a single run:
```bash
uv run python eval.py --dataset cadets --seed 71
# or, if uv is not on PATH on the execution node:
./.venv/bin/python eval.py --dataset cadets --seed 71
```

With the commands above, results are written to
`results_rerun/<dataset>_seed<seed>.log`.

## Expected Output

For paper reproduction, use the **strict attack-chain** metrics reported in each
evaluation log, where contaminated nodes are excluded from metric accounting.

Key metrics: F1, AP, AUROC, FPR, MCC, ADP (Attack Detection Precision).

ADP is rank-based and can be sensitive to tied or nearly tied anomaly scores.
On Cadets, ADP varied across CPU environments while AP, AUROC, and the
operating-point metrics were unchanged.

## Aggregating Results

Aggregate the canonical logs in `results/` into mean ± std tables:

```bash
uv run python utils/aggregate_results.py
# or: uv run python utils/aggregate_results.py --log_dir results
# or, if uv is not on PATH on the execution node:
./.venv/bin/python utils/aggregate_results.py --log_dir results
```

Use `--log_dir results_rerun` to aggregate a fresh evaluation.

## Customisation

| Variable | Default | Description |
|---|---|---|
| `DATA_DIR` | `PROJECT_ROOT/data/DARPA` | Raw data location |
| `GROUND_TRUTH_DIR` | `PROJECT_ROOT/ground_truth/...` | REAPr labels |
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
