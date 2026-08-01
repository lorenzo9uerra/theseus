# Theseus

This repository provides the official code for the paper "How Benchmarks and Evaluation Protocols Shape Conclusions in Provenance-Based Intrusion Detection".

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

Sync the project dependencies:

```bash
uv sync
```

## Data Setup

We provide processed datasets and evaluation artifacts on Zenodo. Raw-data processing instructions are included in [Manual Data Processing](#appendix-manual-data-processing).

### 1. Download Artifacts

Download the following archives from Zenodo:

* **Processed datasets:** Parquet and CSV node and event tables derived from the raw JSON logs. The Parquet archive is recommended.
    * URL: [https://doi.org/10.5281/zenodo.18450778](https://doi.org/10.5281/zenodo.18450778)
    * Files: `DARPA-TC-E3-Parquet.zip`, `DARPA-TC-E3-CSV.zip`

* **Reproducibility artifacts:** Cache files, model checkpoints, retained baseline artifacts, and sanitized evaluation logs for the reported results.
    * URL: [https://doi.org/10.5281/zenodo.21427594](https://doi.org/10.5281/zenodo.21427594)
    * Files: `theseus-source.tar.gz`, `theseus_artifacts.tar.gz`, `magic_artifacts.tar.gz`, `pidsmaker_artifacts.tar.gz`, `atlasv2_artifacts.tar.gz`

### 2. Extract and Configure

Place the downloaded archives in the project root and extract them.

```bash
# Extracts the DARPA/ folder (with its subdirectories) into the data/ directory
unzip -o DARPA-TC-E3-Parquet.zip -d data/

# Theseus: extracts 'checkpoints/' and 'cache/' into the project root
tar -xzf theseus_artifacts.tar.gz -C .

# MAGIC baseline: extracts 'data/', 'checkpoints/', 'results/' into baselines/MAGIC/
tar -xzf magic_artifacts.tar.gz -C baselines/MAGIC/

# Orthrus/Velox baselines (PIDSMaker): extracts 'artifacts/' and 'results/' into baselines/PIDSMaker/
tar -xzf pidsmaker_artifacts.tar.gz -C baselines/PIDSMaker/

# ATLASv2 secondary benchmark: extract into a staging directory
mkdir -p atlas_bundle
tar -xzf atlasv2_artifacts.tar.gz -C atlas_bundle/
```

### Ground Truth

This project uses [REAPr](https://bitbucket.org/sts-lab/reapr-ground-truth) ground truth at commit `e726c01`. The E3 labels under `ground_truth/` normalize one duplicated Cadets attack-chain identifier without changing process UUIDs or label assignments. The ATLASv2 artifact contains UUID-based labels under `atlas_bundle/ground_truth/atlasv2/`.

### Expected Directory Structure

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

If you also extracted `atlasv2_artifacts.tar.gz`, the staged ATLAS payload will look like:

```text
theseus/
├── atlas_bundle/
│   ├── data/ATLASV2/
│   ├── ground_truth/atlasv2/
│   ├── pidsmaker/
│   └── theseus/
└── ...
```

## Reproducing Paper Results

### 1. E3 Main Results

The main E3 table combines Theseus, Magic, Orthrus, Velox, and the allowlist
diagnostic under the strict attack-chain scope. The commands in this section use
the released artifacts in evaluation-only mode; they **do not perform training**.

For the Theseus rows, ensure you have extracted the `cache` and `checkpoints`
folders as described above, then run:

```bash
RESULT_DIR=results_rerun ./scripts/reproduce_results.sh
uv run python scripts/aggregate_results.py --log_dir results_rerun
```

Fresh logs are written to
`results_rerun/theseus_<dataset>_seed<seed>.log`, preserving the canonical
sanitized logs under `results/`.

For the allowlist row, regenerate and aggregate the deterministic diagnostic:

```bash
for ds in CADETS_E3 FIVEDIRECTIONS_E3 THEIA_E3 TRACE_E3; do
  uv run python scripts/allowlist_diagnostic.py "$ds" \
    --cmd-mode executable \
    --output "outputs/${ds,,}_allowlist_diagnostic.csv"
done
uv run python scripts/aggregate_allowlist_results.py
```

For the Magic, Orthrus, and Velox rows, use the eval-only baseline commands in
[Baselines](#baselines-magic-orthrus-and-velox).

The Theseus eval-only path requires approximately 5 GB of RAM and 6 GB of GPU
memory.

### 1.1 ATLASv2 Secondary Benchmark

The ATLASv2 artifact is packaged separately because it bundles the processed ATLAS tables
used for the paper's secondary-benchmark analysis.

After extracting `atlasv2_artifacts.tar.gz` into `atlas_bundle/` as shown above:

* processed ATLAS parquet tables under `atlas_bundle/data/ATLASV2/`
* UUID-based process labels under `atlas_bundle/ground_truth/atlasv2/`
* canonical result logs under `atlas_bundle/results/`
* deterministic allowlist outputs under `atlas_bundle/outputs/`
* Theseus caches and checkpoints under `atlas_bundle/theseus/`
* eval-only Velox artifacts under `atlas_bundle/pidsmaker/`

To reconstruct the exact paper table from the packaged canonical results:

```bash
./.venv/bin/python scripts/aggregate_atlasv2_secondary_results.py \
  --results-dir atlas_bundle/results \
  --allowlist-dir atlas_bundle/outputs
```

To rerun the ATLASv2 secondary results end to end:

```bash
RESULT_DIR=results_rerun/atlas \
ALLOWLIST_DIR=outputs/atlas_ae \
./scripts/reproduce_atlasv2_results.sh
```

This script:

* re-runs `Theseus` in eval-only mode from the staged cache/checkpoints
* re-runs `Velox` in eval-only mode from the staged `PIDSMaker` artifacts
* regenerates the deterministic allowlist diagnostic CSVs
* aggregates everything into:
  * `outputs/atlas_ae/atlasv2_secondary_benchmark_summary.json`
  * `outputs/atlas_ae/atlasv2_secondary_benchmark_summary.md`

### 1.2 Runtime Profile

The runtime appendix can be regenerated with the profiling script:

```bash
uv run python scripts/profile_runtime.py THEIA_E3 --size avg
```

### 2. Retraining from Scratch

The top-level retraining helper covers Theseus. To retrain Theseus from scratch
across all seeds used in the paper:

```bash
./scripts/retrain_from_scratch.sh
```

This will train on all four datasets (CADETS_E3, FIVEDIRECTIONS_E3, THEIA_E3, TRACE_E3) with seeds 65129, 923457, 56604, 9382, and 58371.

Logs are written to `results/theseus_<dataset>_seed<seed>.log`. To also log to Weights & Biases, set `WANDB=1` before running the script.

If you prefer to retrain a single configuration manually, set the Python hash seed to ensure deterministic vocabulary building:

```bash
export PYTHONHASHSEED=0
uv run main.py CADETS_E3
```

For Magic, Orthrus, and Velox retraining, use the baseline commands in
[Baselines](#baselines-magic-orthrus-and-velox). Those commands live with the
baseline forks because they use separate environments, seeds, and intermediate
artifacts.

## General Usage

To run training and evaluation on specific datasets using the default configuration:

```bash
uv run main.py CADETS_E3
```

**Arguments:**

* `dataset`: Positional target dataset (`CADETS_E3`, `THEIA_E3`, `FIVEDIRECTIONS_E3`, `TRACE_E3`).
* `--config PATH`: Path to a custom configuration file (overrides defaults).
* `--test`: Run evaluation only (skips training; requires valid checkpoints).
* `--wandb`: Enable logging to Weights & Biases.
* `--force_restart`: Force a rebuild of graph snapshots, ignoring the cache.

### Custom Configuration

You can override default hyperparameters by providing a custom YAML configuration file:

```bash
uv run main.py CADETS_E3 --config configs/examples/custom_theseus_example.yml
```

## Baselines (Magic, Orthrus, and Velox)

Reproduction instructions for the baseline systems are provided in their respective directories:

* **Magic:** See [baselines/MAGIC/](baselines/MAGIC/)
* **Orthrus and Velox:** See [baselines/PIDSMaker/](baselines/PIDSMaker/)

Both baselines use separate `uv` environments and provide Makefiles for evaluation and retraining.
Magic evaluation can run on CPU. PIDSMaker evaluation requires a 128 GB RAM
allocation and one CUDA-capable GPU on the larger E3 datasets.

After extracting `magic_artifacts.tar.gz` / `pidsmaker_artifacts.tar.gz`, these
commands reproduce the Magic row and the Orthrus/Velox rows of the E3 main
table:

```bash
cd baselines/MAGIC
make setup
make eval RESULT_DIR=results_rerun
uv run python utils/aggregate_results.py --log_dir results_rerun

cd ../PIDSMaker
make setup
ln -sf ../../data data
ln -sf ../../ground_truth ground_truth
make eval RESULT_DIR=results_rerun
uv run python scripts/aggregate_results.py --log_dir results_rerun
```

Using a separate result directory forces a fresh evaluation while preserving the
canonical logs bundled with the artifacts. To reconstruct the paper table directly
from those bundled logs, skip `make eval` and run each aggregation command without
`--log_dir`.

To retrain the baselines from scratch instead of using the released checkpoints:

```bash
cd baselines/MAGIC
make setup
make all
uv run python utils/aggregate_results.py

cd ../PIDSMaker
make setup
ln -sf ../../data data
ln -sf ../../ground_truth ground_truth
make all
uv run python scripts/aggregate_results.py
```

## Development

For researchers intending to modify or extend the code, install the development dependencies:

```bash
uv sync --extra dev
```

This installs `pytest`, `ruff`, `ty`, and `pre-commit`.

**Run Tests:**

```bash
uv run pytest tests
```

These tests are synthetic unit and smoke tests for the core pipeline and do not require dataset downloads or cluster access.

**Setup Git Hooks:**

```bash
uv run pre-commit install
```

**Manual Linting:**

```bash
uv run ruff check --exclude baselines --exclude MAGIC .
uv run ruff format --check --exclude baselines --exclude MAGIC .
uv run ty check --exclude baselines --exclude MAGIC .
```

## Container

An optional Docker recipe for the main Theseus E3 evaluation is provided in [containers/](./containers/). Magic, PIDSMaker, and the ATLASv2 evaluation use the native environments documented above.

## Appendix: Manual Data Processing

> **Note:** This section is only necessary if you wish to process the raw DARPA TC Engagement 3 data from scratch. This process requires approximately **500GB of free disk space** and may take multiple days.

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

The Theseus code and other original components of this repository are licensed under the [Apache License 2.0](./LICENSE). Modified third-party baseline forks under `baselines/` retain their upstream licenses: Magic is MIT-licensed ([baselines/MAGIC/LICENSE](baselines/MAGIC/LICENSE)), and PIDSMaker/Orthrus/Velox is Apache-2.0-licensed ([baselines/PIDSMaker/LICENSE](baselines/PIDSMaker/LICENSE)).
