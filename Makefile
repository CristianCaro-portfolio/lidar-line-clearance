.PHONY: install lint test demo smoke benchmark diagram

install:
	pip install -e ".[dev]"

lint:
	ruff check src tests
	ruff format --check src tests

test:
	pytest

demo:
	lineclear demo --workdir demo -c configs/default.yaml

smoke:
	lineclear demo --workdir demo -c configs/smoke.yaml

benchmark:
	lineclear benchmark -c configs/default.yaml --seeds 5

diagram:
	lineclear diagram docs/architecture.png
