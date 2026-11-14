"""End-to-end training pipeline: raw data -> cleaned data -> features -> models -> artefacts.

Run:  python scripts/run_pipeline.py            (full)
      python scripts/run_pipeline.py --fast     (small hyper-parameter grids, for CI/tests)
"""
from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone

import joblib
import pandas as pd

from src.analytics.business_insights import generate_insights, product_affinity
from src.analytics.explain import permutation_table, shap_table
from src.data.cleaning import run_cleaning
from src.data.load_data import load_raw_transactions
from src.data.validate_data import check_schema, profile_dataframe
from src.features.customer_features import FEATURE_COLUMNS, build_customer_features, validate_features
from src.features.rfm import default_snapshot
from src.models.customer_segmentation import assign_segments, fit_segmentation, save_bundle, segment_profile
from src.models.value_prediction import make_windows, save_metrics, score_customers, train_and_select
from src.recommendations.evaluation import evaluate_recommenders
from src.utils.config import get_logger, get_paths

log = get_logger("pipeline")


def _mlflow_tracker(paths):
    """Return a callable(name, params, result) that logs to MLflow, or None if MLflow is unavailable."""
    try:
        import mlflow
        mlflow.set_tracking_uri(f"sqlite:///{paths.root / 'mlflow.db'}")
        mlflow.set_experiment("retailiq-value-prediction")

        def track(name: str, params: dict, result: dict) -> None:
            with mlflow.start_run(run_name=name):
                mlflow.log_params({k: str(v) for k, v in params.items()})
                for split in ("validation", "test"):
                    for m in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
                        mlflow.log_metric(f"{split}_{m}", result[split][m])
        return track
    except Exception as exc:  # tracking is optional
        log.warning("MLflow tracking disabled: %s", exc)
        return None


def run_pipeline(root=None, fast: bool = False, n_customers: int = 2000, track: bool = True) -> dict:
    t0 = time.time()
    paths = get_paths(root).ensure()

    log.info("1/8 Load + validate")
    raw, source = load_raw_transactions(paths, synthetic_customers=n_customers)
    assert not check_schema(raw.assign(InvoiceDate=pd.to_datetime(raw["InvoiceDate"]))), "schema check failed"
    (paths.processed / "raw_profile.json").write_text(json.dumps(profile_dataframe(raw), indent=2, default=str))

    log.info("2/8 Clean")
    clean, report = run_cleaning(raw, paths, source)

    log.info("3/8 Customer features")
    snapshot = default_snapshot(clean)
    feats = build_customer_features(clean, snapshot)
    issues = validate_features(feats)
    assert not issues, issues
    feats.to_parquet(paths.processed / "customer_features.parquet")

    log.info("4/8 Segmentation")
    seg_bundle = fit_segmentation(feats)
    save_bundle(seg_bundle, paths.models / "kmeans_customer_segments.joblib")
    seg = assign_segments(seg_bundle, feats)
    profile = segment_profile(feats, seg["segment"])
    profile.to_parquet(paths.processed / "segment_profile.parquet")

    log.info("5/8 Value prediction")
    res = train_and_select(clean, fast=fast, tracker=_mlflow_tracker(paths) if track else None)
    bundle, metrics = res["bundle"], res["metrics"]
    joblib.dump(bundle, paths.models / "value_model.joblib")
    val = res["data"]["val"]
    perm = permutation_table(bundle["pipeline"], val[FEATURE_COLUMNS], val["high_value"].to_numpy(), n_repeats=5 if fast else 10)
    shap_df = shap_table(res["fitted"]["XGBoost"], res["prep"], res["data"]["test"][FEATURE_COLUMNS])
    metrics["explainability"] = {"permutation": perm.round(5).to_dict("records"),
                                 "shap": None if shap_df is None else shap_df.round(5).to_dict("records")}
    save_metrics(metrics, paths.models / "value_metrics.json")

    scored = feats.join(seg).join(score_customers(bundle, feats))
    scored["predicted_high_value"] = (scored["p_high_value"] >= bundle["threshold"]).astype(int)
    scored.to_parquet(paths.processed / "customer_scores.parquet")

    log.info("6/8 Recommender evaluation")
    split = clean["InvoiceDate"].max().normalize() - pd.Timedelta(days=60)
    reco_eval = evaluate_recommenders(clean, split)
    (paths.models / "recommender_eval.json").write_text(json.dumps(reco_eval, indent=2))

    log.info("7/8 Business insights")
    affinity = product_affinity(clean)
    affinity.to_parquet(paths.processed / "product_affinity.parquet")
    importance = shap_df if shap_df is not None else perm.rename(columns={"importance": "mean_abs_shap"})
    insights = generate_insights(clean, scored, profile, importance, affinity, reco_eval)
    (paths.models / "business_insights.json").write_text(json.dumps(insights, indent=2))

    log.info("8/8 Metadata")
    meta = {"data_source": source, "is_synthetic": source == "synthetic", "snapshot_date": str(snapshot),
            "rows_raw": len(raw), "rows_clean": len(clean), "customers": len(feats),
            "n_segments": seg_bundle["k"], "silhouette": round(seg_bundle["silhouette"], 4),
            "selected_model": metrics["selected_model"], "windows": make_windows(clean).as_dict(),
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "python": platform.python_version(),
            "seconds": round(time.time() - t0, 1)}
    (paths.models / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    log.info("Done in %.0fs. Source=%s, selected=%s", time.time() - t0, source, metrics["selected_model"])
    return meta
