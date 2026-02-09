# ============================================================
# PIDSMaker - Reproducible Artifact Pipeline
# ============================================================
# Usage:
#   make setup              Install uv environment + package
#   make orthrus            Train & evaluate Orthrus (all seeds)
#   make velox              Train & evaluate Velox (all seeds)
#   make all                Run both Orthrus and Velox
#   make eval-orthrus       Eval-only for Orthrus (all seeds)
#   make eval-velox         Eval-only for Velox (all seeds)
#
# Override variables:
#   make all SEEDS="111 333" DEVICE=0
#   make orthrus DATASETS="CADETS_E3 THEIA_E3"
#   make all CSV_BASE_DIR=/path/to/data/DARPA
#   make all WANDB=1   Enable Weights & Biases logging
# ============================================================

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

# ---------- Paths ----------
PIDS_ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
PROJECT_ROOT := $(abspath $(PIDS_ROOT)/../..)

CSV_BASE_DIR ?= $(PROJECT_ROOT)/data/DARPA

DATASETS := CADETS_E3 FIVEDIRECTIONS_E3 THEIA_E3 TRACE_E3
SEEDS    := 111 333 828 0 433
DEVICE   ?=
WANDB    ?=

# Build --wandb flag when WANDB is set (e.g. make orthrus WANDB=1)
ifdef WANDB
  WANDB_FLAG := --wandb
else
  WANDB_FLAG :=
endif

# Build --cpu flag if no GPU requested
ifdef DEVICE
  DEVICE_FLAG :=
else
  DEVICE_FLAG := --cpu
endif

RESULT_DIR := $(PIDS_ROOT)/results/paper

# ---------- Targets ----------

.PHONY: all setup verify-env verify-links orthrus velox \
        eval-orthrus eval-velox clean help

all: orthrus velox

help:
	@echo "PIDSMaker Artifact Evaluation"
	@echo "============================="
	@echo ""
	@echo "  make setup          - Create uv env & install package"
	@echo "  make orthrus        - Full Orthrus pipeline (all datasets, all seeds)"
	@echo "  make velox          - Full Velox pipeline (all datasets, all seeds)"
	@echo "  make all            - Run both Orthrus and Velox"
	@echo "  make eval-orthrus   - Re-evaluate Orthrus only (skip training)"
	@echo "  make eval-velox     - Re-evaluate Velox only (skip training)"
	@echo "  make clean          - Remove result logs"
	@echo ""
	@echo "Variables:"
	@echo "  CSV_BASE_DIR=$(CSV_BASE_DIR)"
	@echo "  DATASETS=$(DATASETS)"
	@echo "  SEEDS=$(SEEDS)"
	@echo "  DEVICE=$(DEVICE)  (unset = CPU)"
	@echo "  WANDB=$(WANDB)  (set to 1 to enable W&B logging)"

# ---------- Environment ----------

setup:
	@echo "=== Setting up uv environment ==="
	cd $(PIDS_ROOT) && uv sync
	@echo ""
	@echo "=== Environment ready. Use 'uv run' or 'source .venv/bin/activate' ==="

verify-env:
	@cd $(PIDS_ROOT) && uv run python -c "import torch; import pidsmaker; print('Environment OK: torch=' + torch.__version__)" \
		|| (echo "ERROR: Required packages not found. Run 'make setup'." && exit 1)

verify-links:
	@test -d "$(CSV_BASE_DIR)" || (echo "ERROR: Data directory not found: $(CSV_BASE_DIR)" && exit 1)
	@test -d "$(PIDS_ROOT)/ground_truth" || \
		(echo "ERROR: ground_truth symlink missing. Run:" && \
		 echo "  ln -s $(PROJECT_ROOT)/ground_truth $(PIDS_ROOT)/ground_truth" && exit 1)

# ---------- Orthrus ----------

orthrus: verify-env verify-links
	@mkdir -p $(RESULT_DIR)
	@for ds in $(DATASETS); do \
		for seed in $(SEEDS); do \
			echo "=== Orthrus | $$ds | seed=$$seed ==="; \
			PYTHONHASHSEED=0 cd $(PIDS_ROOT) && uv run python $(PIDS_ROOT)/pidsmaker/main.py \
				orthrus $$ds \
				--tuned \
				--restart_from_scratch \
				--csv_base_dir "$(CSV_BASE_DIR)" \
				$(DEVICE_FLAG) \
				$(WANDB_FLAG) \
				--detection.gnn_training.seed $$seed \
				--tags orthrus,reproduction \
				> "$(RESULT_DIR)/orthrus_$${ds}_seed$${seed}.log.tmp" 2>&1 \
			&& mv "$(RESULT_DIR)/orthrus_$${ds}_seed$${seed}.log.tmp" "$(RESULT_DIR)/orthrus_$${ds}_seed$${seed}.log" \
			|| (rm -f "$(RESULT_DIR)/orthrus_$${ds}_seed$${seed}.log.tmp" && echo "  FAILED - see previous log" && continue); \
			echo "  Done. Log: $(RESULT_DIR)/orthrus_$${ds}_seed$${seed}.log"; \
			tail -5 "$(RESULT_DIR)/orthrus_$${ds}_seed$${seed}.log"; \
		done; \
	done
	@echo "=== Orthrus complete ==="

eval-orthrus: verify-env verify-links
	@mkdir -p $(RESULT_DIR)
	@for ds in $(DATASETS); do \
		for seed in $(SEEDS); do \
			echo "=== Orthrus eval | $$ds | seed=$$seed ==="; \
			PYTHONHASHSEED=0 cd $(PIDS_ROOT) && uv run python $(PIDS_ROOT)/pidsmaker/main.py \
				orthrus $$ds \
				--tuned \
				--force_restart evaluation \
				--csv_base_dir "$(CSV_BASE_DIR)" \
				$(DEVICE_FLAG) \
				$(WANDB_FLAG) \
				--detection.gnn_training.seed $$seed \
				--tags orthrus,reproduction \
				> "$(RESULT_DIR)/orthrus_$${ds}_seed$${seed}.log.tmp" 2>&1 \
			&& mv "$(RESULT_DIR)/orthrus_$${ds}_seed$${seed}.log.tmp" "$(RESULT_DIR)/orthrus_$${ds}_seed$${seed}.log" \
			|| (rm -f "$(RESULT_DIR)/orthrus_$${ds}_seed$${seed}.log.tmp" && echo "  FAILED - see previous log" && continue); \
			tail -5 "$(RESULT_DIR)/orthrus_$${ds}_seed$${seed}.log"; \
		done; \
	done
	@echo "=== Orthrus eval complete ==="

# ---------- Velox ----------

velox: verify-env verify-links
	@mkdir -p $(RESULT_DIR)
	@for ds in $(DATASETS); do \
		for seed in $(SEEDS); do \
			echo "=== Velox | $$ds | seed=$$seed ==="; \
			PYTHONHASHSEED=0 cd $(PIDS_ROOT) && uv run python $(PIDS_ROOT)/pidsmaker/main.py \
				velox $$ds \
				--tuned \
				--restart_from_scratch \
				--csv_base_dir "$(CSV_BASE_DIR)" \
				$(DEVICE_FLAG) \
				$(WANDB_FLAG) \
				--detection.gnn_training.seed $$seed \
				--tags velox,reproduction \
				> "$(RESULT_DIR)/velox_$${ds}_seed$${seed}.log.tmp" 2>&1 \
			&& mv "$(RESULT_DIR)/velox_$${ds}_seed$${seed}.log.tmp" "$(RESULT_DIR)/velox_$${ds}_seed$${seed}.log" \
			|| (rm -f "$(RESULT_DIR)/velox_$${ds}_seed$${seed}.log.tmp" && echo "  FAILED - see previous log" && continue); \
			echo "  Done. Log: $(RESULT_DIR)/velox_$${ds}_seed$${seed}.log"; \
			tail -5 "$(RESULT_DIR)/velox_$${ds}_seed$${seed}.log"; \
		done; \
	done
	@echo "=== Velox complete ==="

eval-velox: verify-env verify-links
	@mkdir -p $(RESULT_DIR)
	@for ds in $(DATASETS); do \
		for seed in $(SEEDS); do \
			echo "=== Velox eval | $$ds | seed=$$seed ==="; \
			PYTHONHASHSEED=0 cd $(PIDS_ROOT) && uv run python $(PIDS_ROOT)/pidsmaker/main.py \
				velox $$ds \
				--tuned \
				--force_restart evaluation \
				--csv_base_dir "$(CSV_BASE_DIR)" \
				$(DEVICE_FLAG) \
				$(WANDB_FLAG) \
				--detection.gnn_training.seed $$seed \
				--tags velox,reproduction \
				> "$(RESULT_DIR)/velox_$${ds}_seed$${seed}.log.tmp" 2>&1 \
			&& mv "$(RESULT_DIR)/velox_$${ds}_seed$${seed}.log.tmp" "$(RESULT_DIR)/velox_$${ds}_seed$${seed}.log" \
			|| (rm -f "$(RESULT_DIR)/velox_$${ds}_seed$${seed}.log.tmp" && echo "  FAILED - see previous log" && continue); \
			tail -5 "$(RESULT_DIR)/velox_$${ds}_seed$${seed}.log"; \
		done; \
	done
	@echo "=== Velox eval complete ==="

# ---------- Cleanup ----------

clean:
	rm -rf $(RESULT_DIR)
