"""Cached data access for the dashboard. Builds demo artefacts automatically on first launch if they are missing."""
from __future__ import annotations

import json

import joblib
import pandas as pd
import streamlit as st

from src.recommendations.hybrid import HybridRecommender
from src.utils.config import get_paths

REQUIRED = ["data/processed/cleaned_transactions.parquet", "data/processed/customer_scores.parquet",
            "models/value_metrics.json", "models/run_metadata.json"]


def artifacts_ready() -> bool:
    root = get_paths().root
    return all((root / r).exists() for r in REQUIRED)


def ensure_artifacts() -> None:
    """First launch (e.g. fresh Streamlit Cloud deploy without committed artefacts): build a small demo set."""
    if artifacts_ready():
        return
    from src.pipeline import run_pipeline
    with st.spinner("First launch: building demo data and models (about a minute)..."):
        run_pipeline(fast=True, n_customers=1200, track=False)
    st.cache_data.clear()


@st.cache_data(show_spinner=False)
def load_clean() -> pd.DataFrame:
    return pd.read_parquet(get_paths().processed / "cleaned_transactions.parquet")


@st.cache_data(show_spinner=False)
def load_scores() -> pd.DataFrame:
    return pd.read_parquet(get_paths().processed / "customer_scores.parquet")


@st.cache_data(show_spinner=False)
def load_profile() -> pd.DataFrame:
    return pd.read_parquet(get_paths().processed / "segment_profile.parquet")


@st.cache_data(show_spinner=False)
def load_affinity() -> pd.DataFrame:
    return pd.read_parquet(get_paths().processed / "product_affinity.parquet")


@st.cache_data(show_spinner=False)
def load_json(name: str) -> dict | list:
    return json.loads((get_paths().models / name).read_text())


@st.cache_resource(show_spinner=False)
def load_seg_bundle() -> dict:
    return joblib.load(get_paths().models / "kmeans_customer_segments.joblib")


@st.cache_resource(show_spinner="Preparing the recommendation engine...")
def load_recommender() -> HybridRecommender:
    c = load_clean()
    return HybridRecommender.from_transactions(c[["InvoiceNo", "StockCode", "Description", "RecordType", "HasCustomer", "CustomerID"]])
