# Makefile for quantum-devops-ci development
# Run 'make help' to see available commands

.PHONY: help install install-dev test test-all lint format clean build docker docs serve-docs
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
PIP := pip3
NPM := npm
DOCKER := docker
DOCKER_COMPOSE := docker-compose
PROJECT_NAME := quantum-devops-ci

# Colors for output
BOLD := \033[1m
RED := \033[31m
GREEN := \033[32m  
YELLOW := \033[33m
BLUE := \033[34m
RESET := \033[0m

help: ## Show this help message
	@echo "$(BOLD)$(PROJECT_NAME) Development Commands$(RESET)"
	@echo ""
	@echo "$(BOLD)Setup:$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-15s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST) | grep -E "(install|setup)"
	@echo ""
	@echo "$(BOLD)Development:$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-15s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST) | grep -E "(test|lint|format|build|run)"
	@echo ""
	@echo "$(BOLD)Docker:$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-15s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST) | grep -E "(docker|compose)"
	@echo ""
	@echo "$(BOLD)Utilities:$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-15s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST) | grep -E "(clean|docs|serve)"

# Setup and Installation
install: ## Install production dependencies
	@echo "$(BLUE)Installing production dependencies...$(RESET)"
	$(NPM) ci --only=production
	$(PIP) install -e .

install-dev: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(RESET)"
	$(NPM) ci
	$(PIP) install -e .[dev,all]
	pre-commit install
	@echo "$(GREEN)Development environment ready!$(RESET)"

setup: install-dev ## Alias for install-dev

# Testing
test: ## Run basic tests
	@echo "$(BLUE)Running basic tests...$(RESET)"
	$(NPM) test
	pytest tests/ -v

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(RESET)"
	pytest tests/unit/ -v

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(RESET)"
	pytest tests/integration/ -v

test-quantum: ## Run quantum-specific tests
	@echo "$(BLUE)Running quantum tests...$(RESET)"
	quantum-test run --verbose

test-all: ## Run all tests including quantum tests
	@echo "$(BLUE)Running all tests...$(RESET)"
	$(NPM) test
	pytest tests/ -v --cov=quantum_devops_ci --cov-report=html --cov-report=term
	quantum-test run --verbose

test-coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(RESET)"
	pytest tests/ --cov=quantum_devops_ci --cov-report=html --cov-report=term --cov-report=xml
	@echo "$(GREEN)Coverage report generated in htmlcov/$(RESET)"

# Code Quality
lint: ## Run all linters
	@echo "$(BLUE)Running linters...$(RESET)"
	$(NPM) run lint
	flake8 src/ tests/ quantum-tests/
	mypy src/quantum_devops_ci
	bandit -r src/ -f txt
	quantum-lint check src/

lint-py: ## Run Python linters only
	@echo "$(BLUE)Running Python linters...$(RESET)"
	flake8 src/ tests/ quantum-tests/
	mypy src/quantum_devops_ci
	bandit -r src/

lint-js: ## Run JavaScript linters only
	@echo "$(BLUE)Running JavaScript linters...$(RESET)"
	$(NPM) run lint

lint-fix: ## Run linters with auto-fix
	@echo "$(BLUE)Running linters with auto-fix...$(RESET)"
	$(NPM) run lint:fix
	black src/ tests/ quantum-tests/
	isort src/ tests/ quantum-tests/

format: ## Format all code
	@echo "$(BLUE)Formatting code...$(RESET)"
	black src/ tests/ quantum-tests/
	isort src/ tests/ quantum-tests/
	$(NPM) run format
	@echo "$(GREEN)Code formatted!$(RESET)"

# Security
security: ## Run security checks
	@echo "$(BLUE)Running security checks...$(RESET)"
	bandit -r src/ -f json -o bandit-report.json
	safety check --json --output safety-report.json
	$(NPM) audit --audit-level high
	@echo "$(GREEN)Security checks completed!$(RESET)"

# Build and Package
build: ## Build the project
	@echo "$(BLUE)Building project...$(RESET)"
	$(PYTHON) -m build
	$(NPM) run build
	@echo "$(GREEN)Build completed!$(RESET)"

build-docs: ## Build documentation
	@echo "$(BLUE)Building documentation...$(RESET)"
	cd docs && make html
	@echo "$(GREEN)Documentation built in docs/_build/html/$(RESET)"

# Docker Commands
docker-build: ## Build Docker images
	@echo "$(BLUE)Building Docker images...$(RESET)"
	$(DOCKER) build -t $(PROJECT_NAME):dev --target development .
	$(DOCKER) build -t $(PROJECT_NAME):prod --target production .
	@echo "$(GREEN)Docker images built!$(RESET)"

docker-run: ## Run development container
	@echo "$(BLUE)Starting development container...$(RESET)"
	$(DOCKER) run -it --rm -v $(PWD):/workspace $(PROJECT_NAME):dev

docker-test: ## Run tests in Docker container
	@echo "$(BLUE)Running tests in Docker...$(RESET)"
	$(DOCKER) run --rm -v $(PWD):/workspace $(PROJECT_NAME):dev quantum-test run --ci

compose-up: ## Start all services with docker-compose
	@echo "$(BLUE)Starting all services...$(RESET)"
	$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)All services started!$(RESET)"
	@echo "Jupyter Lab: http://localhost:8889"
	@echo "Docs: http://localhost:8080"
	@echo "Grafana: http://localhost:3001"

compose-down: ## Stop all services
	@echo "$(BLUE)Stopping all services...$(RESET)"
	$(DOCKER_COMPOSE) down
	@echo "$(GREEN)All services stopped!$(RESET)"

compose-logs: ## View logs from all services
	$(DOCKER_COMPOSE) logs -f

# Development Servers
serve-docs: ## Serve documentation locally
	@echo "$(BLUE)Starting documentation server...$(RESET)"
	cd docs && $(PYTHON) -m http.server 8000
	@echo "$(GREEN)Documentation available at http://localhost:8000$(RESET)"

jupyter: ## Start Jupyter Lab
	@echo "$(BLUE)Starting Jupyter Lab...$(RESET)"
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Quantum Development
quantum-demo: ## Run quantum demo
	@echo "$(BLUE)Running quantum demo...$(RESET)"
	$(PYTHON) examples/basic/demo.py

quantum-benchmark: ## Run quantum benchmarks
	@echo "$(BLUE)Running quantum benchmarks...$(RESET)"
	quantum-benchmark run --output-format json

# Cleanup
clean: ## Clean build artifacts and cache
	@echo "$(BLUE)Cleaning build artifacts...$(RESET)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf node_modules/.cache/
	rm -rf .mypy_cache/
	rm -rf __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)Cleaned!$(RESET)"

clean-all: clean ## Clean everything including node_modules
	@echo "$(BLUE)Deep cleaning...$(RESET)"
	rm -rf node_modules/
	rm -rf .venv/
	rm -rf docs/_build/
	$(DOCKER) system prune -f
	@echo "$(GREEN)Deep clean completed!$(RESET)"

# Release and Publishing
version: ## Show current version
	@echo "$(BLUE)Current version:$(RESET)"
	@$(PYTHON) -c "import json; print('Python:', json.load(open('pyproject.toml'))['project']['version'])" 2>/dev/null || echo "Python version not found"
	@$(NPM) -s run version 2>/dev/null || echo "Node.js version not found"

bump-patch: ## Bump patch version
	@echo "$(BLUE)Bumping patch version...$(RESET)"
	$(NPM) version patch --no-git-tag-version
	# Update Python version manually or with bumpversion tool

bump-minor: ## Bump minor version
	@echo "$(BLUE)Bumping minor version...$(RESET)"
	$(NPM) version minor --no-git-tag-version

bump-major: ## Bump major version
	@echo "$(BLUE)Bumping major version...$(RESET)"
	$(NPM) version major --no-git-tag-version

# CI/CD Helpers
ci-setup: install-dev ## Setup for CI environment
	@echo "$(BLUE)Setting up CI environment...$(RESET)"
	pre-commit install-hooks

ci-test: ## Run CI test suite
	@echo "$(BLUE)Running CI test suite...$(RESET)"
	$(MAKE) lint
	$(MAKE) test-all
	$(MAKE) security
	$(MAKE) docker-build

# Utility Commands
check-deps: ## Check for outdated dependencies
	@echo "$(BLUE)Checking Python dependencies...$(RESET)"
	$(PIP) list --outdated
	@echo "$(BLUE)Checking Node.js dependencies...$(RESET)"
	$(NPM) outdated

update-deps: ## Update dependencies
	@echo "$(BLUE)Updating dependencies...$(RESET)"
	$(NPM) update
	$(PIP) install --upgrade -e .[dev,all]

pre-commit: ## Run pre-commit hooks on all files
	@echo "$(BLUE)Running pre-commit hooks...$(RESET)"
	pre-commit run --all-files

validate: ## Validate project configuration
	@echo "$(BLUE)Validating project configuration...$(RESET)"
	$(PYTHON) -m json.tool package.json > /dev/null && echo "✅ package.json valid"  
	$(PYTHON) -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))" && echo "✅ pyproject.toml valid"
	$(NPM) run validate 2>/dev/null || echo "⚠️  NPM validation not configured"
	@echo "$(GREEN)Configuration validation completed!$(RESET)"]