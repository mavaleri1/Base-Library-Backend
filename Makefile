.PHONY: help build up down restart logs clean

help:
	@echo "NewWeb3 - core AI Backend"
	@echo ""
	@echo "Available commands:"
	@echo "  make build          - Build all Docker images"
	@echo "  make up             - Start all services"
	@echo "  make up-langfuse    - Start all services with LangFuse"
	@echo "  make down           - Stop all services"
	@echo "  make restart        - Restart all services"
	@echo "  make logs           - View logs from all services"
	@echo "  make logs-core - View logs from core service"
	@echo "  make clean          - Remove all containers, volumes, and data"
	@echo "  make health         - Check health of all services"

build:
	docker compose build

up:
	docker compose up -d

up-langfuse:
	docker compose --profile langfuse up -d

down:
	docker compose down

restart:
	docker compose restart

logs:
	docker compose logs -f

logs-core:
	docker compose logs -f core

logs-artifacts:
	docker compose logs -f artifacts-service

logs-prompts:
	docker compose logs -f prompt-config-service

clean:
	docker compose down -v
	rm -rf data/* logs/*

health:
	@echo "Checking service health..."
	@curl -s http://localhost:8000/health || echo "core: DOWN"
	@curl -s http://localhost:8001/health || echo "Artifacts: DOWN"
	@curl -s http://localhost:8002/health || echo "Prompt Config: DOWN"
