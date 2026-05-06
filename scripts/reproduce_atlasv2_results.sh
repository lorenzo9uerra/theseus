#!/bin/bash
set -euo pipefail

if [ ! -f "main.py" ]; then
    echo "Run this script from the project root directory."
    exit 1
fi

ROOT_DIR="$(pwd)"
ATLAS_BUNDLE_DIR="${ATLAS_BUNDLE_DIR:-atlas_bundle}"
RESULT_DIR="${RESULT_DIR:-results}"
ALLOWLIST_DIR="${ALLOWLIST_DIR:-outputs}"
THESEUS_SEEDS=(65129 923457 56604 9382 58371)
VELOX_SEEDS=(111 333 828 0 433)
ATLAS_DATASETS=(atlasv2_h1 atlasv2_h2)

resolve_path() {
    local path="$1"
    if [[ "${path}" = /* ]]; then
        printf '%s\n' "${path}"
    else
        printf '%s/%s\n' "${ROOT_DIR}" "${path}"
    fi
}

ATLAS_BUNDLE_DIR="$(resolve_path "${ATLAS_BUNDLE_DIR}")"
RESULT_DIR="$(resolve_path "${RESULT_DIR}")"
ALLOWLIST_DIR="$(resolve_path "${ALLOWLIST_DIR}")"
ATLAS_DATA_DIR="${ATLAS_BUNDLE_DIR}/data/ATLASV2"
THESEUS_CACHE_DIR="${ATLAS_BUNDLE_DIR}/theseus/cache"
THESEUS_CHECKPOINT_DIR="${ATLAS_BUNDLE_DIR}/theseus/checkpoints"
PIDS_ARTIFACT_DIR="${ATLAS_BUNDLE_DIR}/pidsmaker/artifacts"

mkdir -p "${RESULT_DIR}" "${ALLOWLIST_DIR}"

if [ ! -d "${ATLAS_DATA_DIR}" ]; then
    echo "Missing ATLAS data directory: ${ATLAS_DATA_DIR}"
    echo "Extract atlasv2_artifacts.tar.gz into ${ATLAS_BUNDLE_DIR}/ first."
    exit 1
fi

if [ ! -d "${THESEUS_CACHE_DIR}" ] || [ ! -d "${THESEUS_CHECKPOINT_DIR}" ]; then
    echo "Missing staged Theseus ATLAS artifacts under ${ATLAS_BUNDLE_DIR}/theseus/"
    exit 1
fi

if [ ! -d "${PIDS_ARTIFACT_DIR}" ]; then
    echo "Missing staged PIDSMaker ATLAS artifacts under ${ATLAS_BUNDLE_DIR}/pidsmaker/"
    exit 1
fi

run_theseus_eval() {
    local dataset="$1"
    local seed="$2"
    local checkpoint="${THESEUS_CHECKPOINT_DIR}/checkpoint_theseus_${dataset}_seed_${seed}_paper.pt"
    local config="configs/tuned/theseus_${dataset}.yml"
    local log_file="${RESULT_DIR}/theseus_${dataset}_seed${seed}.log"

    echo "=== Theseus eval | ${dataset} | seed=${seed} ==="
    ./.venv/bin/python main.py "${dataset}" \
        --config "${config}" \
        --seed "${seed}" \
        --test \
        --cache_dir "${THESEUS_CACHE_DIR}" \
        --checkpoint_dir "${THESEUS_CHECKPOINT_DIR}" \
        --checkpoint "${checkpoint}" \
        --data_dir "${ATLAS_DATA_DIR}" \
        --outputs_dir "${ROOT_DIR}/outputs/${dataset}_seed_${seed}" \
        > "${log_file}.tmp" 2>&1 \
    && mv "${log_file}.tmp" "${log_file}" \
    || { echo "FAILED: Theseus ${dataset} seed=${seed}"; tail -80 "${log_file}.tmp" || true; exit 1; }
}

run_velox_eval() {
    local dataset="$1"
    local seed="$2"
    local log_file="${RESULT_DIR}/velox_${dataset}_seed${seed}.log"

    echo "=== Velox eval | ${dataset} | seed=${seed} ==="
    (
        cd "${ROOT_DIR}/baselines/PIDSMaker"
        PYTHONHASHSEED=0 ./.venv/bin/python pidsmaker/main.py \
            velox "${dataset}" \
            --tuned \
            --force_restart evaluation \
            --csv_base_dir "${ATLAS_DATA_DIR}" \
            --artifact_dir "${PIDS_ARTIFACT_DIR}" \
            --detection.gnn_training.seed "${seed}" \
            > "${log_file}.tmp" 2>&1
    ) \
    && mv "${log_file}.tmp" "${log_file}" \
    || { echo "FAILED: Velox ${dataset} seed=${seed}"; tail -80 "${log_file}.tmp" || true; exit 1; }
}

for dataset in "${ATLAS_DATASETS[@]}"; do
    for seed in "${THESEUS_SEEDS[@]}"; do
        run_theseus_eval "${dataset}" "${seed}"
    done
done

for dataset in "${ATLAS_DATASETS[@]}"; do
    for seed in "${VELOX_SEEDS[@]}"; do
        run_velox_eval "${dataset}" "${seed}"
    done
done

for dataset in "${ATLAS_DATASETS[@]}"; do
    ./.venv/bin/python scripts/allowlist_diagnostic.py "${dataset}" \
        --data-dir "${ATLAS_DATA_DIR}" \
        --ground-truth-dir "${ROOT_DIR}/ground_truth/reapr-ground-truth/atlasv2" \
        --output "${ALLOWLIST_DIR}/${dataset}_allowlist_diagnostic.csv"
done

./.venv/bin/python scripts/aggregate_atlasv2_secondary_results.py \
    --results-dir "${RESULT_DIR}" \
    --allowlist-dir "${ALLOWLIST_DIR}" \
    --json-out "${ALLOWLIST_DIR}/atlasv2_secondary_benchmark_summary.json" \
    --markdown-out "${ALLOWLIST_DIR}/atlasv2_secondary_benchmark_summary.md"

echo "ATLASv2 evaluation-only reproduction finished."
