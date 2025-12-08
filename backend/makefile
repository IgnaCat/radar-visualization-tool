all:

# Development with hot-reload (single worker for debugging)
run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Production with multiple workers (better performance)
run-prod:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4 --limit-concurrency 2000 --backlog 4096 --timeout-keep-alive 120