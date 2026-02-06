#!/bin/bash

set -e

uv run gdown --folder "https://drive.google.com/drive/folders/1fOCY3ERsEmXmvDekG-LUUSjfWs6TRdp-" -O data/raw/

# Move extracted files up one level and clean up unused directories
mv data/raw/data/* data/raw/
rmdir data/raw/data/
rm data/raw/clearscope/*
rmdir data/raw/clearscope/
# Clean up any bin files that may have been downloaded.
# Those are the raw avro binary files that were used to extract the JSONs.
# However, the authors already provide the JSON files, so we don't need the bin files.
find data/raw/ -type f -name "*\.bin\.*" -delete

for dataset in trace cadets fivedirections theia; do
    echo "Processing dataset: $dataset"
    for file in data/raw/$dataset/*.tar.gz; do
        echo "  Extracting file: $file"
        tar xvf "$file" -C "data/raw/$dataset/"
        rm "$file"
    done
    dataset_upper=$(echo "$dataset" | tr '[:lower:]' '[:upper:]')_E3
    uv run scripts/create_csv_files_e3.py \
        --raw_dir "data/raw/$dataset" \
        --out_dir "data/DARPA/$dataset_upper"
    echo "Finished processing $dataset"
    echo "----------------------------------------"
done
