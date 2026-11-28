.PHONY: install db test lint api worker corpus dataset eval eval-smoke

install:
	uv sync --extra local-rerank

db:
	docker compose up -d db

test:
	uv run pytest -q

lint:
	uv run ruff check . && uv run ruff format --check .

api:
	uv run uvicorn app.api:app --reload --port 8000

worker:
	uv run python -m app.worker

corpus:
	uv run python -m scripts.ingest_local evals/corpus/*.pdf --strategies fixed,recursive,semantic

dataset:
	uv run python -m evals.build_dataset --per-doc 12

eval:
	uv run python -m evals.run_eval --split dev

eval-smoke:
	uv run python -m evals.run_eval --limit 15 --rerankers none
