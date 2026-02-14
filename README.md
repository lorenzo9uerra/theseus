# Theseus

This repository provides the official code for the paper "Theseus: Navigating the Labyrinth of Evaluation Bias in Provenance-based Intrusion Detection".

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

Sync the project dependencies:

```bash
uv sync
```

## Data Setup

To facilitate reproducibility, we provide pre-processed datasets and artifacts. While it is possible to process the raw data from scratch (see [Manual Data Processing](#appendix-manual-data-processing) at the end of this document), we recommend using the Zenodo artifacts for immediate access.

### 1. Download Artifacts

Download the following two archives from Zenodo:

* **Processed Datasets:** Contains the Parquet and CSV files derived from the raw JSON logs. We highly recommend using the Parquet version for faster loading times and stricter type handling.
    * URL: [https://zenodo.org/records/18450779](https://zenodo.org/records/18450779)


* **Reproducibility Artifacts:** Contains the graph construction cache, Theseus model checkpoints, and Word2Vec embeddings required to reproduce the exact results reported in the paper without retraining.
    * URL: [https://zenodo.org/records/18489505](https://zenodo.org/records/18489505)



### 2. Extract and Configure

Place the downloaded `.zip` files in the project root and extract them.

```bash
# Extracts the DARPA/ folder (with its subdirectories) into the data/ directory
unzip DARPA-TC-E3-Parquet.zip -d data/

# Extracts 'checkpoints/' and 'cache/' folders
unzip theseus_artifacts.zip
```

**Ground Truth Data**
This project uses the [REAPr](https://bitbucket.org/sts-lab/reapr-ground-truth) ground truth (Liu et al.). The necessary files are already included in this repository under the `ground_truth/` directory (Commit ID: `b67da6b`). No further action is required.

**Expected Directory Structure**
Ensure your project structure looks like this before proceeding:

```text
theseus/
├── data/
│   └── DARPA/
│       ├── CADETS_E3/
│       │   ├── process_node_table.parquet
│       │   └── ...
│       └── ...
├── checkpoints/
│   └── theseus/
│       └── ...
├── cache/
│   └── graph_cadets_e3_...
├── ground_truth/
└── main.py

```

## Reproducing Paper Results

### Erratum: Table 2 Correction

>Note: The final published version of the paper has been updated to reflect the correct values produced by this artifact.

In the originally submitted manuscript, the False Positive Rate (FPR) reported for the **Velox** baseline on the **Cadets** dataset was erroneously calculated. This error affected the mean and standard deviation for that specific entry, resulting in a value outside the valid range. The correct value is:

* **FPR of Velox on CADETS:** 0.0031 ± 0.0028

### 1. Evaluation

To reproduce the precise results reported in the paper, use the provided script. This script **does not perform training**; it loads the pre-built graph cache and evaluates the provided checkpoints to generate the metrics.

Ensure you have extracted the `cache` and `checkpoints` folders as described above, then run:

```bash
./scripts/reproduce_results.sh
```

This process requires approximately 5 GB of RAM and 6 GB of GPU memory.

### 2. Retraining from Scratch

To retrain all models from scratch across all seeds used in the paper:

```bash
./scripts/retrain_from_scratch.sh
```

This will train on all four datasets (CADETS_E3, FIVEDIRECTIONS_E3, THEIA_E3, TRACE_E3) with seeds 65129, 923457, 56604, 9382, and 58371. Results are logged to Weights & Biases.

If you prefer to retrain a single configuration manually, set the Python hash seed to ensure deterministic vocabulary building:

```bash
export PYTHONHASHSEED=0
uv run main.py CADETS_E3
```

## General Usage

To run training and evaluation on specific datasets using the default configuration:

```bash
uv run main.py CADETS_E3
```

**Arguments:**

* `--dataset`: The target dataset (`CADETS_E3`, `THEIA_E3`, `FIVEDIRECTIONS_E3`, `TRACE_E3`).
* `--config PATH`: Path to a custom configuration file (overrides defaults).
* `--test`: Run evaluation only (skips training; requires valid checkpoints).
* `--wandb`: Enable logging to Weights & Biases.
* `--force_restart`: Force a rebuild of graph snapshots, ignoring the cache.

### Custom Configuration

You can override default hyperparameters by providing a custom YAML configuration file:

```bash
uv run main.py CADETS_E3 --config configs/examples/custom_theseus_example.yml
```

## Baselines (Magic, Orthrus and Velox)

Reproduction instructions for the baseline systems are provided in their respective directories:

* **Magic:** See [baselines/MAGIC/](baselines/MAGIC/)
* **Orthrus and Velox:** See [baselines/PIDSMaker/](baselines/PIDSMaker/)

Both baselines use conda for dependency management and include Makefiles for one-command reproducibility.

## Development

For researchers intending to modify or extend the code, install the development dependencies:

```bash
uv sync --extra dev
```

This installs `ruff` for linting and formatting, along with `pre-commit` hooks.

**Setup Git Hooks:**

```bash
uv run pre-commit install
```

**Manual Linting:**

```bash
uv run ruff check .
uv run ruff format .
```

## Appendix: Manual Data Processing

> **Note:** This section is only necessary if you wish to process the raw DARPA TC Engagement 3 data from scratch. This process requires approximately **500GB of free disk space** and may take several days.

The `scripts/process_datasets.sh` script automates the pipeline:

1. **Download:** Fetches raw datasets (TRACE, CADETS, FIVEDIRECTIONS, THEIA) via `gdown`.
2. **Cleanup:** Removes unnecessary files (e.g., binaries) to conserve space.
3. **Extraction:** Iteratively extracts and deletes `.tar.gz` archives.
4. **Preprocessing:** Parses raw JSON logs into the Parquet/CSV format used by Theseus.

To run the full processing pipeline:

```bash
./scripts/process_datasets.sh
```

## License

All original components of this repository are licensed under the [Apache License 2.0](./LICENSE). Third-party components are used in compliance with their respective licenses.
