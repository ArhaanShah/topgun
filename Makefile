PROFILE ?= fp8_offload
RUN_ID ?=
RUNS_DIR ?= $(if $(PHASE_A_RUNS_DIR),$(PHASE_A_RUNS_DIR),.phase_a_runs)
CACHE_DIR ?= $(if $(PHASE_A_CACHE_DIR),$(PHASE_A_CACHE_DIR),.phase_a_cache)
COMMON = --profile $(PROFILE) --runs-dir $(RUNS_DIR) --cache-dir $(CACHE_DIR) --run-id $(RUN_ID)

.PHONY: test check format preflight download verify-offline prepare health-check canary smoke judge-smoke reproduce judge-reproduction export-audit finalize export-run mock-e2e 2x2-prepare 2x2-run 2x2-export-audit 2x2-import-audit 2x2-analyze 2x2-recover-lock followup-prepare followup-run followup-status followup-export-audit followup-import-audit followup-analyze followup-export-run followup-verify-run followup-recover-lock followup-mock-e2e

test:
	python -m pytest -q

check:
	python -m ruff check src tests scripts
	python -m ruff format --check src tests scripts

format:
	python -m ruff check --fix src tests scripts
	python -m ruff format src tests scripts

preflight:
	python -m phase_a.cli --command preflight --profile $(PROFILE) --cache-dir $(CACHE_DIR) --runs-dir $(RUNS_DIR)

download:
	python -m phase_a.cli --command download --profile $(PROFILE) --cache-dir $(CACHE_DIR) --runs-dir $(RUNS_DIR)

verify-offline:
	python -m phase_a.cli --command verify-offline --profile $(PROFILE) --cache-dir $(CACHE_DIR) --runs-dir $(RUNS_DIR)

prepare:
	python -m phase_a.cli --command prepare $(COMMON)

health-check:
	python -m phase_a.cli --command health-check $(COMMON)

canary:
	python -m phase_a.cli --command canary $(COMMON) --resume

smoke:
	python -m phase_a.cli --command smoke $(COMMON) --resume

judge-smoke:
	python -m phase_a.cli --command judge $(COMMON) --split smoke --resume

reproduce:
	python -m phase_a.cli --command reproduce $(COMMON) --resume

judge-reproduction:
	python -m phase_a.cli --command judge $(COMMON) --split reproduction --resume

export-audit:
	python -m phase_a.cli --command audit $(COMMON)

finalize:
	python -m phase_a.cli --command finalize $(COMMON)

export-run:
	python scripts/export_run.py $(RUNS_DIR)/$(RUN_ID)

mock-e2e:
	python -m phase_a.cli --command prepare $(COMMON) --mock
	python -m phase_a.cli --command health-check $(COMMON) --mock
	python -m phase_a.cli --command smoke $(COMMON) --mock
	python -m phase_a.cli --command judge $(COMMON) --split smoke --mock

2X2_COMMON = $(if $(strip $(RUN_ID)),,$(error RUN_ID is required))$(if $(filter command line environment,$(origin CACHE_DIR)),,$(error CACHE_DIR must be supplied explicitly))$(if $(filter command line environment,$(origin RUNS_DIR)),,$(error RUNS_DIR must be supplied explicitly))--run-id "$(RUN_ID)" --cache-dir "$(CACHE_DIR)" --runs-dir "$(RUNS_DIR)"
2X2_DOWNLOAD = $(if $(filter 1,$(DOWNLOAD)),--download,)

2x2-prepare:
	python -m phase_a.understand_2x2 prepare $(2X2_COMMON) $(2X2_DOWNLOAD)

2x2-run:
	python -m phase_a.understand_2x2 run $(2X2_COMMON) --resume

2x2-export-audit:
	python -m phase_a.understand_2x2 export-audit $(2X2_COMMON)

2x2-import-audit:
	python -m phase_a.understand_2x2 import-audit $(2X2_COMMON) --labels "$(if $(strip $(LABELS)),$(LABELS),$(error LABELS is required))"

2x2-analyze:
	python -m phase_a.understand_2x2 analyze $(2X2_COMMON)

2x2-recover-lock:
	python -m phase_a.understand_2x2 recover-lock $(2X2_COMMON)

# Evidence followup experiment (144-response design)
FOLLOWUP_COMMON = $(if $(strip $(RUN_ID)),,$(error RUN_ID is required))$(if $(filter command line environment,$(origin CACHE_DIR)),,$(error CACHE_DIR must be supplied explicitly))$(if $(filter command line environment,$(origin RUNS_DIR)),,$(error RUNS_DIR must be supplied explicitly))--run-id "$(RUN_ID)" --cache-dir "$(CACHE_DIR)" --runs-dir "$(RUNS_DIR)"
FOLLOWUP_DOWNLOAD = $(if $(filter 1,$(DOWNLOAD)),--download,)

followup-prepare:
	python -m phase_a.evidence_followup prepare $(FOLLOWUP_COMMON) $(FOLLOWUP_DOWNLOAD)

followup-run:
	python -m phase_a.evidence_followup run $(FOLLOWUP_COMMON) --resume

followup-status:
	python -m phase_a.evidence_followup status $(FOLLOWUP_COMMON)

followup-export-audit:
	python -m phase_a.evidence_followup export-audit $(FOLLOWUP_COMMON)

followup-import-audit:
	python -m phase_a.evidence_followup import-audit $(FOLLOWUP_COMMON) --labels "$(if $(strip $(LABELS)),$(LABELS),$(error LABELS is required))"

followup-analyze:
	python -m phase_a.evidence_followup analyze $(FOLLOWUP_COMMON)

followup-export-run:
	python -m phase_a.evidence_followup export-run $(FOLLOWUP_COMMON)

followup-verify-run:
	python -m phase_a.evidence_followup verify-run $(FOLLOWUP_COMMON)

followup-recover-lock:
	python -m phase_a.evidence_followup recover-lock $(FOLLOWUP_COMMON)

followup-mock-e2e:
	python -m phase_a.evidence_followup mock-e2e $(FOLLOWUP_COMMON)
