# Finalex backend

Finalex uses LiteLLM for model calls and MLflow for local or remote experiment
tracking. It does not depend on the OpenAI Agents SDK.

## Local setup

```bash
cd backend
cp .env.example .env
uv sync
uv run mlflow ui --backend-store-uri ./mlruns
```

In another terminal, run a service from the backend directory so the shared
`finalex_runtime.py` module is on `PYTHONPATH`, for example:

```bash
uv run --project researcher uvicorn researcher.server:app --reload
```

Set `MLFLOW_TRACKING_URI=http://localhost:5000` to send runs to the UI server.
Without it MLflow writes runs to `backend/mlruns`.
