"""Lazily-loaded model/data store shared by all routes (loaded once per process)."""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pandas as pd
from fastapi import Header, HTTPException, status

from src.recommendations.hybrid import HybridRecommender
from src.utils.config import get_logger, get_paths

log = get_logger("retailiq.api")


@dataclass
class ArtifactStore:
    scores: pd.DataFrame
    value_bundle: dict
    seg_bundle: dict
    profile: pd.DataFrame
    recommender: HybridRecommender
    meta: dict
    descriptions: dict = field(default_factory=dict)


_store: ArtifactStore | None = None
_lock = threading.Lock()


def load_store(root: str | Path | None = None) -> ArtifactStore:
    paths = get_paths(root)
    needed = [paths.processed / "customer_scores.parquet", paths.processed / "cleaned_transactions.parquet",
              paths.models / "value_model.joblib", paths.models / "kmeans_customer_segments.joblib", paths.models / "run_metadata.json"]
    missing = [p.name for p in needed if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing artefacts {missing}. Run `python scripts/run_pipeline.py` first.")
    clean = pd.read_parquet(paths.processed / "cleaned_transactions.parquet", columns=[
        "InvoiceNo", "StockCode", "Description", "RecordType", "HasCustomer", "CustomerID"])
    store = ArtifactStore(
        scores=pd.read_parquet(paths.processed / "customer_scores.parquet"),
        value_bundle=joblib.load(paths.models / "value_model.joblib"),
        seg_bundle=joblib.load(paths.models / "kmeans_customer_segments.joblib"),
        profile=pd.read_parquet(paths.processed / "segment_profile.parquet"),
        recommender=HybridRecommender.from_transactions(clean),
        meta=json.loads((paths.models / "run_metadata.json").read_text()))
    log.info("Artefacts loaded: %d customers, %d products", len(store.scores), store.recommender.data.n_items)
    return store


def get_store() -> ArtifactStore:
    global _store
    with _lock:
        if _store is None:
            try:
                _store = load_store()
            except FileNotFoundError as exc:
                raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
        return _store


def reset_store() -> None:
    global _store
    _store = None


def require_api_key(x_api_key: str | None = Header(default=None, description="Required only if RETAILIQ_API_KEY is set")) -> None:
    expected = os.environ.get("RETAILIQ_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing X-API-Key header.")
