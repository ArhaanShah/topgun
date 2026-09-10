PROFILE ?= fp8_offload
RUN_ID ?=
RUNS_DIR ?= $(if $(PHASE_A_RUNS_DIR),$(PHASE_A_RUNS_DIR),.phase_a_runs)
CACHE_DIR ?= $(if $(PHASE_A_CACHE_DIR),$(PHASE_A_CACHE_DIR),.phase_a_cache)
COMMON = --profile $(PROFILE) --runs-dir $(RUNS_DIR) --cache-dir $(CACHE_DIR) --run-id $(RUN_ID)

.PHONY: test check format preflight download verify-offline prepare health-check canary smoke judge-smoke reproduce judge-reproduction export-audit finalize export-run mock-e2e

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
