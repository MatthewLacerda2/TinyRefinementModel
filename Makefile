# The front door (#169): the incantations live here, not in a session's memory.
#
#   make lint                    what CI's ruff job runs
#   make test                    what CI's pytest job runs (core, then apparatus — separately,
#                                since one process for both has been OOM-killed on this box)
#   make test-affected           only the tests a change can reach; fails open to `make test`
#   make audit                   the validity audit over the specs a change can reach (#304);
#                                `make audit ALL=1` sweeps every spec and finding
#   make gate                    VRAM headroom of the config a launch would train (#161)
#   make launch SPEC=… BUDGET=…  a supervised base run; refuses without a committed spec or BUDGET
#   make report RUN=run_...      the terminal report and plots for a run
#
# `make launch` is gated: the supervisor first runs the real trainer for ~5 minutes and
# refuses a config that does not survive an apply, a probe and a checkpoint (#168).

PY ?= venv/bin/python

.PHONY: lint test test-affected audit gate launch report

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

audit:
	$(PY) -m instruments.audit $(if $(ALL),--all,)

gate:
	$(PY) -m instruments.vram_headroom_smoke

launch:
	@test -n "$(BUDGET)" || { echo "no BUDGET, no launch: make launch SPEC=experiments/base/specs/<id>.toml BUDGET=4e9 [ISSUE=157]"; exit 2; }
	@test -n "$(SPEC)" || { echo "no SPEC, no launch: the base run is pre-registered (#294)"; exit 2; }
	$(PY) -m trm.runtime.launch --budget $(BUDGET) --spec $(SPEC) $(if $(ISSUE),--issue $(ISSUE),)

report:
	@test -n "$(RUN)" || { echo "which run? make report RUN=run_YYYYMMDD_HHMMSS"; exit 2; }
	$(PY) -m instruments.report --log runs/$(RUN)/metrics.csv
	$(PY) -m instruments.plots --log runs/$(RUN)/metrics.csv --out runs/$(RUN)
