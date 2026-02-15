# ============================================================
# MAGIC - Reproducible Artifact Pipeline
# ============================================================
# Usage:
#   make setup           Install uv environment
#   make data            Parse raw DARPA TC E3 -> DGL graphs
#   make train           Train all datasets (all seeds, no eval)
#   make eval            Evaluate all datasets (all seeds)
#   make all             Full pipeline: data -> train+eval each seed
#   make clean           Remove generated data/checkpoints/results
#
# Override paths / options:
#   make data DATA_DIR=/path/to/DARPA GROUND_TRUTH_DIR=/path/to/gt
#   make train WANDB=1   Enable Weights & Biases logging
# ============================================================

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

# ---------- Paths (relative to this Makefile / MAGIC root) ----------
MAGIC_ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
PROJECT_ROOT := $(abspath $(MAGIC_ROOT)/../..)

DATA_DIR       ?= $(PROJECT_ROOT)/data/DARPA
GROUND_TRUTH_DIR ?= $(PROJECT_ROOT)/ground_truth/reapr-ground-truth/darpa-tc-engagement3

	DATASETS := cadets fivedirections theia trace
	SEEDS    := 71 83 232 441 915
	DEVICE   ?= -1
	WANDB    ?=

# Build --wandb flag when WANDB is set (e.g. make train WANDB=1)
ifdef WANDB
  WANDB_FLAG := --wandb
else
  WANDB_FLAG :=
endif

# ---------- Targets ----------

.PHONY: all setup data train eval clean verify-env help train-all-seeds

all: data train-all-seeds

help:
	@echo "MAGIC"
	@echo "========================="
	@echo "  make setup    - Create uv environment"
	@echo "  make data     - Parse DARPA TC E3 datasets into DGL graphs"
	@echo "  make train    - Train all datasets / seeds (no evaluation)"
	@echo "  make eval     - Evaluate trained models"
	@echo "  make all      - Full pipeline: data -> train+eval each seed"
	@echo "  make clean    - Remove generated artifacts"
	@echo ""
	@echo "Configurable variables:"
	@echo "  DATA_DIR=$(DATA_DIR)"
	@echo "  GROUND_TRUTH_DIR=$(GROUND_TRUTH_DIR)"
	@echo "  DEVICE=$(DEVICE)  (GPU id, -1 for CPU)"
	@echo "  SEEDS=$(SEEDS)"
	@echo "  WANDB=$(WANDB)  (set to 1 to enable W&B logging)"

# ---------- Environment ----------

setup:
	@echo "=== Setting up uv environment ==="
	cd $(MAGIC_ROOT) && uv sync
	@echo "=== Environment ready. Use 'uv run' or 'source .venv/bin/activate' ==="

verify-env:
	@cd $(MAGIC_ROOT) && uv run python -c "import torch; import dgl; import polars; print('Environment OK: torch=' + torch.__version__ + ', dgl=' + dgl.__version__)" \
		|| (echo "ERROR: Required packages not found. Run 'make setup'." && exit 1)

# ---------- Data Preparation ----------

# Sentinel files that mark a dataset as fully parsed
DATA_SENTINELS := $(foreach ds,$(DATASETS),$(MAGIC_ROOT)/data/$(ds)/metadata.json)

data: verify-env $(DATA_SENTINELS)

# Map dataset name -> raw directory name and ground truth filename
cadets_RAW         := CADETS_E3
fivedirections_RAW := FIVEDIRECTIONS_E3
theia_RAW          := THEIA_E3
trace_RAW          := TRACE_E3

cadets_GT         := cadets_labels.csv
fivedirections_GT := fivedirections_labels.csv
theia_GT          := theia_labels.csv
trace_GT          := trace_labels.csv

define parse_dataset_rule
$(MAGIC_ROOT)/data/$(1)/metadata.json:
	@echo "=== Parsing $(1) ==="
	@test -d "$(DATA_DIR)/$($(1)_RAW)" || (echo "ERROR: $(DATA_DIR)/$($(1)_RAW) not found" && exit 1)
	@test -f "$(GROUND_TRUTH_DIR)/$($(1)_GT)" || (echo "ERROR: $(GROUND_TRUTH_DIR)/$($(1)_GT) not found" && exit 1)
	cd $(MAGIC_ROOT) && uv run python $(MAGIC_ROOT)/utils/data_parser_daily.py \
		--dataset $(1) \
		--data_dir "$(DATA_DIR)/$($(1)_RAW)" \
		--ground_truth "$(GROUND_TRUTH_DIR)/$($(1)_GT)" \
		--label_filter both \
		--output_dir "$(MAGIC_ROOT)/data/$(1)"
	@echo "=== $(1) done ==="
endef

$(foreach ds,$(DATASETS),$(eval $(call parse_dataset_rule,$(ds))))

# ---------- Training ----------

# One checkpoint per (dataset, seed)
define train_rule
$(MAGIC_ROOT)/checkpoints/checkpoint-$(1)-seed$(2).pt: $(MAGIC_ROOT)/data/$(1)/metadata.json
	@echo "=== Training $(1) seed=$(2) device=$(DEVICE) ==="
	cd $(MAGIC_ROOT) && uv run python $(MAGIC_ROOT)/train.py --dataset $(1) --seed $(2) --device $(DEVICE) $(WANDB_FLAG)
endef

$(foreach ds,$(DATASETS),$(foreach s,$(SEEDS),$(eval $(call train_rule,$(ds),$(s)))))

TRAIN_TARGETS := $(foreach ds,$(DATASETS),$(foreach s,$(SEEDS),$(MAGIC_ROOT)/checkpoints/checkpoint-$(ds)-seed$(s).pt))

train: verify-env $(TRAIN_TARGETS)

# Train and evaluate immediately for each dataset × seed combination
train-all-seeds: verify-env data
	@mkdir -p $(RESULT_DIR)
	@for ds in $(DATASETS); do \
		for seed in $(SEEDS); do \
			echo ""; \
			echo "=== Training $$ds seed=$$seed device=$(DEVICE) ==="; \
			cd $(MAGIC_ROOT) && uv run python $(MAGIC_ROOT)/train.py --dataset $$ds --seed $$seed --device $(DEVICE) $(WANDB_FLAG) || exit 1; \
			echo ""; \
			echo "=== Evaluating $$ds seed=$$seed ==="; \
			cd $(MAGIC_ROOT) && uv run python $(MAGIC_ROOT)/eval.py --dataset $$ds --seed $$seed --device $(DEVICE) $(WANDB_FLAG) \
				> "$(RESULT_DIR)/$${ds}_seed$${seed}.log.tmp" 2>&1 \
			&& mv "$(RESULT_DIR)/$${ds}_seed$${seed}.log.tmp" "$(RESULT_DIR)/$${ds}_seed$${seed}.log" \
		|| { rm -f "$(RESULT_DIR)/$${ds}_seed$${seed}.log.tmp"; exit 1; }; \
			tail -10 "$(RESULT_DIR)/$${ds}_seed$${seed}.log"; \
		done; \
	done
	@echo ""
	@echo "=== All training and evaluation complete. Results in $(RESULT_DIR)/ ==="

# ---------- Evaluation ----------

RESULT_DIR := $(MAGIC_ROOT)/results

define eval_rule
$(RESULT_DIR)/$(1)_seed$(2).log: $(MAGIC_ROOT)/checkpoints/checkpoint-$(1)-seed$(2).pt
	@mkdir -p $(RESULT_DIR)
	@echo "=== Evaluating $(1) seed=$(2) ==="
	cd $(MAGIC_ROOT) && uv run python $(MAGIC_ROOT)/eval.py --dataset $(1) --seed $(2) --device $(DEVICE) $(WANDB_FLAG) \
		> "$(RESULT_DIR)/$(1)_seed$(2).log.tmp" 2>&1 \
		&& mv "$(RESULT_DIR)/$(1)_seed$(2).log.tmp" "$(RESULT_DIR)/$(1)_seed$(2).log"
	@tail -10 "$(RESULT_DIR)/$(1)_seed$(2).log"
endef

$(foreach ds,$(DATASETS),$(foreach s,$(SEEDS),$(eval $(call eval_rule,$(ds),$(s)))))

EVAL_TARGETS := $(foreach ds,$(DATASETS),$(foreach s,$(SEEDS),$(RESULT_DIR)/$(ds)_seed$(s).log))

eval: verify-env $(EVAL_TARGETS)
	@echo ""
	@echo "=== All evaluations complete. Results in $(RESULT_DIR)/ ==="

# ---------- Cleanup ----------

clean:
	rm -rf $(MAGIC_ROOT)/data/cadets $(MAGIC_ROOT)/data/fivedirections \
		$(MAGIC_ROOT)/data/theia $(MAGIC_ROOT)/data/trace
	rm -rf $(MAGIC_ROOT)/checkpoints/checkpoint-*.pt
	rm -rf $(RESULT_DIR)
	rm -rf $(MAGIC_ROOT)/eval_result
