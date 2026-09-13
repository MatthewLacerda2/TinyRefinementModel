# The front door (#169): the incantations live here, not in a session's memory.
#
#   make lint                    what CI's ruff job runs
#   make test                    what CI's pytest job runs (core, then apparatus — separately,
#                                since one process for both has been OOM-killed on this box)
#   make test-affected           only the tests a change can reach; fails open to `make test`
#   make gate                    VRAM headroom of the config a launch would train (#161)
#   make launch BUDGET=4e9       a supervised base run; refuses without BUDGET
#   make report RUN=run_...      the terminal report and plots for a run
#
# `make launch` does not refuse a config that does not fit yet: that gate is #168.

PY ?= venv/bin/python

.PHONY: lint test test-affected gate launch report

lint:
	$(PY) -m ruff check .

test:
	$(PY) -m pytest tests/core -q
	$(PY) -m pytest tests/apparatus -q

test-affected:
	@sel="$$($(PY) tests/affected.py)"; \
	if [ -z "$$sel" ]; then echo "nothing a test can reach changed"; \
	elif [ "$$sel" = "tests/core tests/apparatus" ]; then $(MAKE) --no-print-directory test; \
	else $(PY) -m pytest -q $$sel; fi

gate:
	$(PY) -m instruments.vram_headroom_smoke

launch:
	@test -n "$(BUDGET)" || { echo "no BUDGET, no launch: make launch BUDGET=4e9 [ISSUE=157]"; exit 2; }
	$(PY) -m trm.runtime.launch --budget $(BUDGET) $(if $(ISSUE),--issue $(ISSUE),)

report:
	@test -n "$(RUN)" || { echo "which run? make report RUN=run_YYYYMMDD_HHMMSS"; exit 2; }
	$(PY) -m instruments.report --log runs/$(RUN)/metrics.csv
	$(PY) -m instruments.plots --log runs/$(RUN)/metrics.csv --out runs/$(RUN)
