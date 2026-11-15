# Deployment guide

The repo ships its processed data and trained models (about 9 MB) so both services run straight from GitHub. Raw data is not committed.

## 0. Push to GitHub
```bash
git init && git add . && git commit -m "RetailIQ" && git branch -M main
git remote add origin https://github.com/<you>/retailiq.git && git push -u origin main
```

## 1. Dashboard - Streamlit Community Cloud (free)
1. <https://share.streamlit.io> -> **New app** -> pick the repo, branch `main`, main file `app.py`.
2. **Advanced settings** -> Python **3.11** (or the version you trained with).
3. Deploy. It installs `requirements.txt` (runtime only; no notebook/SHAP/MLflow). If artefacts are missing, the app builds a small demo set on first launch.

## 2. API - Render (free web service)
1. Render dashboard -> **New + > Blueprint** -> select the repo; `render.yaml` defines a Docker web service with `/health` as the health check.
2. Set `RETAILIQ_API_KEY` in the service's environment to require an `X-API-Key` header (the health check stays public).
3. Swagger UI: `https://<service>.onrender.com/docs`. Free instances sleep when idle: the first request takes about a minute.

## 3. Run locally with Docker
```bash
docker build -t retailiq-api . && docker run -p 8000:8000 -e RETAILIQ_API_KEY=secret retailiq-api
curl localhost:8000/health
```

## Environment variables
| Variable | Default | Purpose |
|---|---|---|
| `RETAILIQ_ROOT` | repo root | Folder containing `data/processed` and `models` |
| `RETAILIQ_API_KEY` | unset | Enables API-key auth |
| `RETAILIQ_LOG_LEVEL` | INFO | Log verbosity |

## Important: artefacts are version-sensitive
`models/*.joblib` are pickles. They load reliably only with the library versions in `requirements-lock.txt` (the Docker image applies it as a constraints file). If you train on another Python/library version, run `make pipeline` and commit the regenerated `models/` and `data/processed/`.

## Retraining
`make data && make pipeline && make reports && make test`, commit the changed artefacts, redeploy.
