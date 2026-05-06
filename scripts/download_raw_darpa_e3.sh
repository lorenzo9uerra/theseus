#!/bin/bash

set -euo pipefail

RAW_ROOT="data/raw"
GDRIVE_FOLDER="https://drive.google.com/drive/folders/1fOCY3ERsEmXmvDekG-LUUSjfWs6TRdp-"
MANIFEST_PATH="$RAW_ROOT/download_manifest.sha256"

mkdir -p "$RAW_ROOT"
./.venv/bin/python -m gdown --folder "$GDRIVE_FOLDER" -O "$RAW_ROOT/"

if [ -d "$RAW_ROOT/data" ]; then
    mv "$RAW_ROOT/data/"* "$RAW_ROOT/"
    rmdir "$RAW_ROOT/data/"
fi

find "$RAW_ROOT" -type f -name '*.tar.gz' -print0 | sort -z | xargs -0 sha256sum > "$MANIFEST_PATH"

for dataset in trace cadets fivedirections theia; do
    dataset_dir="$RAW_ROOT/$dataset"
    if [ ! -d "$dataset_dir" ]; then
        continue
    fi
    for file in "$dataset_dir"/*.json.tar.gz; do
        [ -e "$file" ] || continue
        echo "Extracting $file"
        tar xvf "$file" -C "$dataset_dir/"
        rm -f "$file"
    done
done

echo "Raw DARPA TC E3 sources are available under $RAW_ROOT/"
echo "Archive manifest written to $MANIFEST_PATH"
echo "JSON archives were extracted; *.bin.tar.gz archives were left untouched."
