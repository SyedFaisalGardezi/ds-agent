.PHONY: help init build up down services shell jupyter test test-watch lint typecheck clean logs ollama-check

COMPOSE := docker compose -f deploy/docker-compose.yml

init:
	@test -f .env || (cp .env.example .env && echo "Created .env from .env.example — edit it before starting services.")

help:
	@echo "ds-agent dev commands (hybrid: Ollama on host, everything else in Docker)"
	@echo ""
	@echo "  make build         - Build the dev image"
	@echo "  make up            - Start services: mlflow, chromadb, postgres, redis"
	@echo "  make down          - Stop all services"
	@echo "  make shell         - Open bash in dev container"
	@echo "  make jupyter       - Start JupyterLab on http://localhost:8888 (token: dsagent)"
	@echo "  make test          - Run unit tests in container"
	@echo "  make test-watch    - Run unit tests with --lf on file change"
	@echo "  make lint          - ruff check + mypy"
	@echo "  make typecheck     - mypy only"
	@echo "  make ollama-check  - Verify host Ollama reachable from container"
	@echo "  make logs S=mlflow - Tail logs for one service"
	@echo "  make clean         - Remove volumes and containers"

build: init
	$(COMPOSE) build dev

up:
	$(COMPOSE) up -d mlflow chromadb postgres redis

down:
	$(COMPOSE) down

shell: up
	$(COMPOSE) run --rm dev bash

jupyter: up
	$(COMPOSE) up jupyter

test: up
	$(COMPOSE) run --rm dev pytest tests/unit/ --no-cov -v

test-watch: up
	$(COMPOSE) run --rm dev pytest tests/unit/ --no-cov --lf -v

lint: up
	$(COMPOSE) run --rm dev bash -c "ruff check . && mypy agent/ --ignore-missing-imports"

typecheck: up
	$(COMPOSE) run --rm dev mypy agent/ --ignore-missing-imports

ollama-check: up
	$(COMPOSE) run --rm dev curl -s http://host.docker.internal:11434/api/tags | head -c 200

logs:
	$(COMPOSE) logs -f $(S)

clean:
	$(COMPOSE) down -v
