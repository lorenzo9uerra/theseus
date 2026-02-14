#!/bin/bash
set -e

if [ ! -f "main.py" ]; then
    echo "Run this script from the project root directory."
    exit 1
fi

export PYTHONHASHSEED=0

COMMON_SEEDS=(65129 923457 56604 9382 58371)

# Directory where checkpoint_theseus_*.pt files are located
CKPT_DIR="checkpoints/theseus"

RESULT_DIR="results"
mkdir -p "$RESULT_DIR"

# 1. CADETS_E3
for SEED in "${COMMON_SEEDS[@]}"; do
    echo "Running CADETS_E3 (Seed: $SEED)..."
    LOG_FILE="$RESULT_DIR/theseus_CADETS_E3_seed${SEED}.log"
    uv run main.py CADETS_E3 \
        --seed "$SEED" \
        --wandb \
        --test \
        --checkpoint "$CKPT_DIR/checkpoint_theseus_cadets_e3_seed_${SEED}_paper.pt" \
        > "${LOG_FILE}.tmp" 2>&1 \
    && mv "${LOG_FILE}.tmp" "${LOG_FILE}" \
    || { rm -f "${LOG_FILE}.tmp"; exit 1; }
    tail -20 "${LOG_FILE}"
done

# 2. FIVEDIRECTIONS_E3
for SEED in "${COMMON_SEEDS[@]}"; do
    echo "Running FIVEDIRECTIONS_E3 (Seed: $SEED)..."
    LOG_FILE="$RESULT_DIR/theseus_FIVEDIRECTIONS_E3_seed${SEED}.log"
    uv run main.py FIVEDIRECTIONS_E3 \
        --seed "$SEED" \
        --wandb \
        --test \
        --checkpoint "$CKPT_DIR/checkpoint_theseus_fivedirections_e3_seed_${SEED}_paper.pt" \
        > "${LOG_FILE}.tmp" 2>&1 \
    && mv "${LOG_FILE}.tmp" "${LOG_FILE}" \
    || { rm -f "${LOG_FILE}.tmp"; exit 1; }
    tail -20 "${LOG_FILE}"
done

# 3. THEIA_E3
for SEED in "${COMMON_SEEDS[@]}"; do
    echo "Running THEIA_E3 (Seed: $SEED)..."
    LOG_FILE="$RESULT_DIR/theseus_THEIA_E3_seed${SEED}.log"
    uv run main.py THEIA_E3 \
        --seed "$SEED" \
        --wandb \
        --test \
        --checkpoint "$CKPT_DIR/checkpoint_theseus_theia_e3_seed_${SEED}_paper.pt" \
        > "${LOG_FILE}.tmp" 2>&1 \
    && mv "${LOG_FILE}.tmp" "${LOG_FILE}" \
    || { rm -f "${LOG_FILE}.tmp"; exit 1; }
    tail -20 "${LOG_FILE}"
done

# 4. TRACE_E3
for SEED in "${COMMON_SEEDS[@]}"; do
    echo "Running TRACE_E3 (Seed: $SEED)..."
    LOG_FILE="$RESULT_DIR/theseus_TRACE_E3_seed${SEED}.log"
    uv run main.py TRACE_E3 \
        --seed "$SEED" \
        --wandb \
        --test \
        --checkpoint "$CKPT_DIR/checkpoint_theseus_trace_e3_seed_${SEED}_paper.pt" \
        > "${LOG_FILE}.tmp" 2>&1 \
    && mv "${LOG_FILE}.tmp" "${LOG_FILE}" \
    || { rm -f "${LOG_FILE}.tmp"; exit 1; }
    tail -20 "${LOG_FILE}"
done

echo "All reproduction experiments finished."
