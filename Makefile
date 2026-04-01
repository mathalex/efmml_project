# Корень репозитория — рабочая директория для всех целей.
MAKEFLAGS += --no-print-directory

VENV         := .venv
PY           := $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,$(shell command -v python3 2>/dev/null || echo python))
BENCH_OUT    ?= report/data/exp01_results.json
WARMUP       ?= 5
ITERS        ?= 20

.PHONY: help test bench report all

help:
	@echo "Цели:"
	@echo "  make test    — pytest (настройки в pyproject.toml)"
	@echo "  make bench   — exp01 бенчмарк (CUDA), пишет $(BENCH_OUT)"
	@echo "  make report  — report/build.py → report/report.md"
	@echo "  make all     — test, затем bench, затем report"
	@echo "Переменные: PY=$(PY) WARMUP=$(WARMUP) ITERS=$(ITERS) BENCH_OUT=..."

test:
	$(PY) -m pytest tests/ -v

bench:
	PYTHONPATH=src $(PY) experiments/exp01_bench_matmul.py \
		--warmup $(WARMUP) --iters $(ITERS) --output $(BENCH_OUT)

report:
	$(PY) report/build.py

all: test bench report
