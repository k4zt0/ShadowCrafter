.PHONY: install test lint format validate-configs

install:
	python -m pip install -e '.[data,dev]'

test:
	pytest

lint:
	ruff check src tests
	mypy src

format:
	ruff format src tests
	ruff check --fix src tests

validate-configs:
	shadowcrafter config validate configs/models/shadowcrafter-9b.yaml
