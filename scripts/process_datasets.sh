#!/bin/bash

set -euo pipefail

./scripts/download_raw_darpa_e3.sh

for dataset in trace cadets fivedirections theia; do
    echo "Processing dataset: $dataset"
    dataset_upper=$(echo "$dataset" | tr '[:lower:]' '[:upper:]')_E3
    ./.venv/bin/python scripts/create_csv_files_e3.py \
        --raw_dir "data/raw/$dataset" \
        --out_dir "data/DARPA/$dataset_upper"
    echo "Finished processing $dataset"
    echo "----------------------------------------"
done
