"""RFM analysis: Recency, Frequency, Monetary.

* Recency   - days since the customer's last purchase (SMALL is good)
* Frequency - number of distinct sale invoices (BIG is good)
* Monetary  - net spend = sales minus returns (BIG is good)

Everything is computed strictly from transactions BEFORE ``snapshot_date``,
so the numbers are exactly what the business could have known on that day.
"""
from __future__ import annotations

import pandas as pd


def default_snapshot(clean: pd.DataFrame) -> pd.Timestamp:
    """Day after the last transaction: 'today' from the dataset's point of view."""
    return clean["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)


def history_before(clean: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    """Identified-customer transactions strictly before the snapshot (the leak guard)."""
    mask = (clean["InvoiceDate"] < snapshot_date) & clean["HasCustomer"]
    return clean.loc[mask]


def compute_rfm(clean: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    """Return one row per customer with recency_days, frequency, monetary."""
    tx = history_before(clean, snapshot_date)
    sales = tx[tx["RecordType"] == "sale"]
    g = sales.groupby("CustomerID")
    rfm = pd.DataFrame({
        "recency_days": (snapshot_date - g["InvoiceDate"].max()).dt.days,
        "frequency": g["InvoiceNo"].nunique(),
    })
    rfm["monetary"] = tx.groupby("CustomerID")["Revenue"].sum().reindex(rfm.index)  # net of returns
    rfm.index = rfm.index.astype("int64")
    rfm.index.name = "CustomerID"
    return rfm


def add_rfm_scores(rfm: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    """Classic 1-5 quintile scores (5 = best). Ranks first so ties never break qcut."""
    out = rfm.copy()
    out["R_score"] = pd.qcut(out["recency_days"].rank(method="first", ascending=False), bins, labels=range(1, bins + 1)).astype(int)
    out["F_score"] = pd.qcut(out["frequency"].rank(method="first"), bins, labels=range(1, bins + 1)).astype(int)
    out["M_score"] = pd.qcut(out["monetary"].rank(method="first"), bins, labels=range(1, bins + 1)).astype(int)
    out["RFM_score"] = out["R_score"].astype(str) + out["F_score"].astype(str) + out["M_score"].astype(str)
    return out
