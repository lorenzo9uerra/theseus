#!/bin/bash
set -euo pipefail

if [ ! -f "main.py" ]; then
    echo "Run this script from the project root directory."
    exit 1
fi

export PYTHONHASHSEED=0

COMMON_SEEDS=(65129 923457 56604 9382 58371)

CKPT_DIR="${CKPT_DIR:-checkpoints/theseus}"
RESULT_DIR="${RESULT_DIR:-results}"
mkdir -p "$CKPT_DIR" "$RESULT_DIR"

WANDB_FLAG=()
if [ "${WANDB:-0}" = "1" ]; then
    WANDB_FLAG=(--wandb)
fi

FORCE_RESTART_FLAG=()
if [ "${FORCE_RESTART:-0}" = "1" ]; then
    FORCE_RESTART_FLAG=(--force_restart)
fi

run_train() {
    local dataset="$1"
    local seed="$2"
    local dataset_lower
    dataset_lower="$(echo "$dataset" | tr '[:upper:]' '[:lower:]')"

    echo "Training ${dataset} (Seed: ${seed})..."

    local ckpt="${CKPT_DIR}/checkpoint_theseus_${dataset_lower}_seed_${seed}_scratch.pt"
    local log_file="${RESULT_DIR}/theseus_${dataset}_seed${seed}.log"

    uv run main.py "${dataset}" \
        --seed "${seed}" \
        --checkpoint "${ckpt}" \
        "${WANDB_FLAG[@]}" \
        --tags retrain,reproduction \
        "${FORCE_RESTART_FLAG[@]}" \
        > "${log_file}.tmp" 2>&1 \
    && mv "${log_file}.tmp" "${log_file}" \
    || { echo "FAILED: ${dataset} seed=${seed}"; tail -80 "${log_file}.tmp" || true; exit 1; }
}

# 1. CADETS_E3
for SEED in "${COMMON_SEEDS[@]}"; do
    run_train "CADETS_E3" "${SEED}"
done

# 2. FIVEDIRECTIONS_E3
for SEED in "${COMMON_SEEDS[@]}"; do
    run_train "FIVEDIRECTIONS_E3" "${SEED}"
done

# 3. THEIA_E3
for SEED in "${COMMON_SEEDS[@]}"; do
    run_train "THEIA_E3" "${SEED}"
done

# 4. TRACE_E3
for SEED in "${COMMON_SEEDS[@]}"; do
    run_train "TRACE_E3" "${SEED}"
done

echo "All retraining experiments finished."
