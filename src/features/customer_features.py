"""Customer-level feature table (one row per customer) at a given snapshot date.

Why customer-level? ML models and segmentation need ONE row per customer, but the
raw data has one row per product line. Aggregation (groupby) turns the second
into the first.

OBSERVATION WINDOW: all history before ``snapshot_date`` (the full available past).
LEAKAGE GUARD:     nothing on/after ``snapshot_date`` is ever read.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.rfm import compute_rfm, history_before

FEATURE_COLUMNS = [
    "recency_days", "frequency", "monetary", "avg_order_value", "total_items", "product_diversity",
    "active_months", "avg_basket_size", "spend_30d", "spend_60d", "spend_90d",
    "avg_days_between_purchases", "tenure_days", "return_rate",
]


def build_customer_features(clean: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    """Build features for every customer with at least one sale before ``snapshot_date``."""
    snapshot_date = pd.Timestamp(snapshot_date)
    tx = history_before(clean, snapshot_date)
    sales, rets = tx[tx["RecordType"] == "sale"], tx[tx["RecordType"] == "return"]
    feats = compute_rfm(clean, snapshot_date)

    g = sales.groupby("CustomerID")
    gross = g["Revenue"].sum()
    feats["avg_order_value"] = gross / feats["frequency"]                    # gross spend per order
    feats["total_items"] = g["Quantity"].sum()
    feats["product_diversity"] = g["StockCode"].nunique()
    feats["active_months"] = sales["InvoiceDate"].dt.to_period("M").groupby(sales["CustomerID"]).nunique()
    lines_per_invoice = sales.groupby(["CustomerID", "InvoiceNo"])["StockCode"].nunique()
    feats["avg_basket_size"] = lines_per_invoice.groupby("CustomerID").mean()  # distinct products per order

    for days in (30, 60, 90):                                                # recent net spend
        recent = tx[tx["InvoiceDate"] >= snapshot_date - pd.Timedelta(days=days)]
        feats[f"spend_{days}d"] = recent.groupby("CustomerID")["Revenue"].sum()
    feats[["spend_30d", "spend_60d", "spend_90d"]] = feats[["spend_30d", "spend_60d", "spend_90d"]].fillna(0.0)

    # Average gap between consecutive orders (NaN for one-time buyers: undefined, imputed later)
    orders = sales.groupby(["CustomerID", "InvoiceNo"])["InvoiceDate"].min().reset_index().sort_values(["CustomerID", "InvoiceDate"])
    gaps = orders.groupby("CustomerID")["InvoiceDate"].diff().dt.total_seconds() / 86400
    feats["avg_days_between_purchases"] = gaps.groupby(orders["CustomerID"]).mean()

    feats["tenure_days"] = (snapshot_date - g["InvoiceDate"].min()).dt.days
    ret_value = -rets.groupby("CustomerID")["Revenue"].sum()
    feats["return_rate"] = (ret_value.reindex(feats.index).fillna(0) / gross.reindex(feats.index)).clip(0, 1)

    country = (sales.groupby("CustomerID")["Country"].agg(lambda s: s.value_counts().index[0]))
    feats = feats.join(country.rename("country"))
    feats["snapshot_date"] = snapshot_date
    out = feats[FEATURE_COLUMNS + ["country", "snapshot_date"]].sort_index()
    out.index = out.index.astype("int64")          # same dtype whether `clean` came from memory or from parquet
    out.index.name = "CustomerID"
    return out


def validate_features(feats: pd.DataFrame) -> list[str]:
    """Sanity checks that must hold for any feature table."""
    issues = []
    if feats.index.has_duplicates: issues.append("duplicate CustomerID in index")
    if (feats["recency_days"] < 0).any(): issues.append("negative recency (future data leaked in?)")
    if (feats["frequency"] < 1).any(): issues.append("customer with zero orders")
    if (feats["tenure_days"] < feats["recency_days"]).any(): issues.append("tenure shorter than recency")
    if feats[["recency_days", "frequency", "monetary"]].isna().any().any(): issues.append("NaN in RFM")
    return issues
