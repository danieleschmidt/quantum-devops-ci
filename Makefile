.PHONY: test demo install clean

PYTHON := $(shell command -v ~/anaconda3/bin/python3 2>/dev/null || echo python3)

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests/ -v

demo:
	$(PYTHON) demo.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache dist build *.egg-info
