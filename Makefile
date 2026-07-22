PYTHON ?= /home/shunsukenaito/.conda/envs/adv/bin/python
export PYTHONPATH := $(CURDIR)/src

.PHONY: lint test-changed smoke verify-milestone

lint:
	$(PYTHON) -m ruff format --check src scripts tests
	$(PYTHON) -m ruff check src scripts tests
	MYPYPATH=src $(PYTHON) -m mypy src/ard
	$(PYTHON) -m pytest -q tests/unit/test_imports.py
	$(PYTHON) -m ard.cli.train --help
	$(PYTHON) -m ard.cli.evaluate --help

test-changed:
	$(PYTHON) scripts/verify.py --changed

smoke:
	$(PYTHON) scripts/verify.py --smoke

verify-milestone:
	$(PYTHON) scripts/verify.py --changed --force --non-scientific
