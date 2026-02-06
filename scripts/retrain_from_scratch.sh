#!/bin/bash
set -e

if [ ! -f "main.py" ]; then
    echo "Run this script from the project root directory."
    exit 1
fi

export PYTHONHASHSEED=0

COMMON_SEEDS=(65129 923457 56604 9382 58371)

# 1. CADETS_E3
for SEED in "${COMMON_SEEDS[@]}"; do
    echo "Training CADETS_E3 (Seed: $SEED)..."
    uv run main.py CADETS_E3 \
        --seed "$SEED" \
        --wandb \
        --tags retrain,reproduction
done

# 2. FIVEDIRECTIONS_E3
for SEED in "${COMMON_SEEDS[@]}"; do
    echo "Training FIVEDIRECTIONS_E3 (Seed: $SEED)..."
    uv run main.py FIVEDIRECTIONS_E3 \
        --seed "$SEED" \
        --wandb \
        --tags retrain,reproduction
done

# 3. THEIA_E3
for SEED in "${COMMON_SEEDS[@]}"; do
    echo "Training THEIA_E3 (Seed: $SEED)..."
    uv run main.py THEIA_E3 \
        --seed "$SEED" \
        --wandb \
        --tags retrain,reproduction
done

# 4. TRACE_E3
for SEED in "${COMMON_SEEDS[@]}"; do
    echo "Training TRACE_E3 (Seed: $SEED)..."
    uv run main.py TRACE_E3 \
        --seed "$SEED" \
        --wandb \
        --tags retrain,reproduction
done

echo "All retraining experiments finished."
