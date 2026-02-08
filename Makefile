.PHONY: install install-dev test lint format clean run-api run-dashboard ingest demo help

# Default target
help:
	@echo "SupportMind - Available commands:"
	@echo ""
	@echo "  make install       Install production dependencies"
	@echo "  make install-dev   Install development dependencies"
	@echo "  make test          Run tests"
	@echo "  make lint          Run linters"
	@echo "  make format        Format code"
	@echo "  make clean         Clean artifacts"
	@echo "  make ingest        Run data ingestion"
	@echo "  make demo          Run demo"
	@echo "  make run-api       Start FastAPI server"
	@echo "  make run-dashboard Start Streamlit dashboard"
	@echo ""

install:
	pip install -e .

install-dev:
	pip install -e ".[all]"
	@if [ -z "$$CI" ]; then pre-commit install; fi

install-ci:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=supportmind

lint:
	flake8 supportmind tests
	mypy supportmind
	black --check supportmind tests
	isort --check-only supportmind tests

format:
	black supportmind tests
	isort supportmind tests

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

ingest:
	python scripts/ingest_data.py

demo:
	python scripts/demo.py

run-api:
	uvicorn supportmind.api.endpoints:app --reload --host 0.0.0.0 --port 8000

run-dashboard:
	streamlit run app/streamlit_app.py
