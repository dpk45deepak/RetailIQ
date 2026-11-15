# Architecture

```
                   data/raw (UCI csv/xlsx)
                            |
        src/data/load_data.py  ->  validate_data.py  ->  cleaning.py
                            |   data/processed/cleaned_transactions.parquet (+ cleaning_report.json)
                            v
        src/features/rfm.py + customer_features.py   (snapshot-date, leakage-safe)
                            |   data/processed/customer_features.parquet
          +-----------------+------------------+----------------------+
          v                                    v                      v
 models/customer_segmentation.py     models/value_prediction.py    recommendations/
 (log -> scale -> K-Means -> PCA)    (temporal splits, 5 models,   (popularity, item-CF,
  models/kmeans_*.joblib              RBF network from scratch)     TF-IDF, hybrid, eval)
          |                          models/value_model.joblib             |
          +----------------+-------------------+---------------------------+
                           v
              src/analytics/ (eda, explain, business_insights)  ->  models/*.json
                           |
          +----------------+-----------------+
          v                                  v
   Streamlit (app.py + pages/)        FastAPI (api/)  -> Swagger /docs
   src/ui (theme, cached loaders)     ArtifactStore loads artefacts once
```

## Design principles
* **Notebooks explore, `src/` ships.** Notebooks import the same functions the pipeline, dashboard and API use, so there is one implementation of every transformation.
* **One orchestrator**: `src/pipeline.py` (`python scripts/run_pipeline.py`) produces every artefact deterministically (`RANDOM_STATE = 42`) in about a minute on the full data.
* **Artefacts are plain files**: parquet for tables, joblib for models, JSON for metrics. The dashboard and API only *read* them; they never train.
* **Leakage is designed out**: all feature builders take a snapshot date and use only rows strictly before it; labels come only from the following 90 days; tests assert this.
* **Fail soft on optional tools**: SHAP and MLflow are optional; the pipeline falls back to permutation importance / no tracking.

## Modules
| Package | Responsibility |
|---|---|
| `src/data` | load (xlsx/csv/older UCI layout), schema checks, cleaning with a step-by-step report, synthetic data for tests/demos |
| `src/features` | RFM and the 14 customer features |
| `src/models` | K-Means segmentation (Hungarian label assignment), `RBFNetworkClassifier`, value-prediction experiment |
| `src/recommendations` | interaction matrix, popularity, item-based CF, TF-IDF content, hybrid with explanations, time-aware evaluation |
| `src/analytics` | EDA helpers, permutation/SHAP importance, data-driven business insights |
| `src/ui` | dashboard theme (CSS, Plotly template, segment colours) and cached data access |
| `api` | FastAPI app, routers, Pydantic schemas, API-key dependency, request-ID logging |

## API surface
`GET /health`, `GET /customers/{id}`, `GET /customers/{id}/segment`, `POST /customers/{id}/predict-value`, `POST /recommendations`, `GET /analytics/segments`. Validation errors -> 422, unknown customer -> 404, missing/invalid API key (when `RETAILIQ_API_KEY` is set) -> 401. Responses contain derived metrics only - no invoices, names or purchase histories.
