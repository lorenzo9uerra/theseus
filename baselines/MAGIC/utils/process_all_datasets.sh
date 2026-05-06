#!/bin/bash

# Process all DARPA TC E3 datasets with data parser (daily splits)
# This script creates graphs with two-level ground truth labels:
#   - Level 1 (Causal Scope): attack + contaminated
#   - Level 2 (Attack Chain): attack only, contaminated masked
#
# Dataset splits (from data/configs/*.yml):
#   CADETS:        train=[2,3,4,5,7,8,9], val=[6,10], test=[11,12,13]
#   FIVEDIRECTIONS: train=[2,3,4,5,6,7], val=[8,9], test=[10,11,12,13]
#   THEIA:         train=[11], val=[10], test=[12,13]
#   TRACE:         train=[9,11], val=[10], test=[12,13]

set -e  # Exit on error

# Resolve paths relative to the project root, regardless of CWD
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAGIC_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$MAGIC_ROOT/../.." && pwd)"

# Default paths - can be overridden via environment variables
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data/DARPA}"
GROUND_TRUTH_DIR="${GROUND_TRUTH_DIR:-$PROJECT_ROOT/ground_truth/reapr-ground-truth/darpa-tc-engagement3}"

echo "============================================================"
echo "DARPA TC E3 Dataset Processing with Two-Level Ground Truth"
echo "============================================================"
echo "Data directory: $DATA_DIR"
echo "Ground truth directory: $GROUND_TRUTH_DIR"
echo ""

# Process CADETS
if [ -d "$DATA_DIR/CADETS_E3" ]; then
    echo "Processing CADETS dataset..."
    python "$SCRIPT_DIR/data_parser_daily.py" \
        --dataset cadets \
        --data_dir "$DATA_DIR/CADETS_E3" \
        --ground_truth "$GROUND_TRUTH_DIR/cadets_labels.csv" \
        --label_filter both
    echo "CADETS done."
    echo ""
else
    echo "WARNING: CADETS_E3 directory not found at $DATA_DIR/CADETS_E3, skipping..."
fi

# Process FIVEDIRECTIONS
if [ -d "$DATA_DIR/FIVEDIRECTIONS_E3" ]; then
    echo "Processing FIVEDIRECTIONS dataset..."
    python "$SCRIPT_DIR/data_parser_daily.py" \
        --dataset fivedirections \
        --data_dir "$DATA_DIR/FIVEDIRECTIONS_E3" \
        --ground_truth "$GROUND_TRUTH_DIR/fivedirections_labels.csv" \
        --label_filter both
    echo "FIVEDIRECTIONS done."
    echo ""
else
    echo "WARNING: FIVEDIRECTIONS_E3 directory not found at $DATA_DIR/FIVEDIRECTIONS_E3, skipping..."
fi

# Process THEIA
if [ -d "$DATA_DIR/THEIA_E3" ]; then
    echo "Processing THEIA dataset..."
    python "$SCRIPT_DIR/data_parser_daily.py" \
        --dataset theia \
        --data_dir "$DATA_DIR/THEIA_E3" \
        --ground_truth "$GROUND_TRUTH_DIR/theia_labels.csv" \
        --label_filter both
    echo "THEIA done."
    echo ""
else
    echo "WARNING: THEIA_E3 directory not found at $DATA_DIR/THEIA_E3, skipping..."
fi

# Process TRACE
if [ -d "$DATA_DIR/TRACE_E3" ]; then
    echo "Processing TRACE dataset..."
    python "$SCRIPT_DIR/data_parser_daily.py" \
        --dataset trace \
        --data_dir "$DATA_DIR/TRACE_E3" \
        --ground_truth "$GROUND_TRUTH_DIR/trace_labels.csv" \
        --label_filter both
    echo "TRACE done."
    echo ""
else
    echo "WARNING: TRACE_E3 directory not found at $DATA_DIR/TRACE_E3, skipping..."
fi

echo "============================================================"
echo "All available datasets processed successfully!"
echo "============================================================"
echo ""
echo "Output files created in $MAGIC_ROOT/data/<dataset>/:"
echo "  - train{0..n}.pkl: Training graphs (DGL format)"
echo "  - val{0..n}.pkl: Validation graphs (DGL format)"
echo "  - test{0..n}.pkl: Test graphs (DGL format)"
echo "  - ground_truth.pkl: Two-level labels (attack + contaminated)"
echo "  - metadata.json: Dataset metadata"
echo ""
echo "Next steps:"
echo "  1. Train: python train.py --dataset <dataset>"
echo "  2. Evaluate: python eval.py --dataset <dataset> --wandb"
