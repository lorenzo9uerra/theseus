#!/bin/bash
set -euo pipefail

if [ ! -f "main.py" ]; then
    echo "Run this script from the project root directory."
    exit 1
fi

export PYTHONHASHSEED=0

COMMON_SEEDS=(65129 923457 56604 9382 58371)

# Directory where checkpoint_theseus_*.pt files are located
CKPT_DIR="checkpoints/theseus"

RESULT_DIR="${RESULT_DIR:-results}"
mkdir -p "$RESULT_DIR"

WANDB_FLAG=()
if [ "${WANDB:-0}" = "1" ]; then
    WANDB_FLAG=(--wandb)
fi

run_eval() {
    local dataset="$1"
    local seed="$2"
    local ckpt="$3"

    echo "Running ${dataset} (Seed: ${seed})..."

    if [ ! -f "${ckpt}" ]; then
        echo "ERROR: checkpoint not found: ${ckpt}"
        echo "Make sure you extracted 'theseus_artifacts.tar.gz' into the project root:"
        echo "  tar -xzf theseus_artifacts.tar.gz -C ."
        exit 1
    fi

    local log_file="${RESULT_DIR}/theseus_${dataset}_seed${seed}.log"
    uv run main.py "${dataset}" \
        --seed "${seed}" \
        "${WANDB_FLAG[@]}" \
        --test \
        --checkpoint "${ckpt}" \
        > "${log_file}.tmp" 2>&1 \
    && mv "${log_file}.tmp" "${log_file}" \
    || { echo "FAILED: ${dataset} seed=${seed}"; tail -80 "${log_file}.tmp" || true; exit 1; }

    tail -39 "${log_file}"
}

# 1. CADETS_E3
for SEED in "${COMMON_SEEDS[@]}"; do
    run_eval "CADETS_E3" "${SEED}" "$CKPT_DIR/checkpoint_theseus_cadets_e3_seed_${SEED}_paper.pt"
done

# 2. FIVEDIRECTIONS_E3
for SEED in "${COMMON_SEEDS[@]}"; do
    run_eval "FIVEDIRECTIONS_E3" "${SEED}" "$CKPT_DIR/checkpoint_theseus_fivedirections_e3_seed_${SEED}_paper.pt"
done

# 3. THEIA_E3
for SEED in "${COMMON_SEEDS[@]}"; do
    run_eval "THEIA_E3" "${SEED}" "$CKPT_DIR/checkpoint_theseus_theia_e3_seed_${SEED}_paper.pt"
done

# 4. TRACE_E3
for SEED in "${COMMON_SEEDS[@]}"; do
    run_eval "TRACE_E3" "${SEED}" "$CKPT_DIR/checkpoint_theseus_trace_e3_seed_${SEED}_paper.pt"
done

echo "All reproduction experiments finished."
