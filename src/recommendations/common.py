"""Shared data structures for all recommenders."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass
class RecoData:
    """Everything a recommender needs, built ONLY from the transactions it is given.

    items:    one row per product (StockCode, Description, n_orders = popularity)
    matrix:   binary customer x product interaction matrix (1 = customer bought the product)
    history:  customer_id -> sorted array of product indices already bought
    """
    items: pd.DataFrame
    code_to_idx: dict[str, int]
    matrix: sparse.csr_matrix
    customer_ids: np.ndarray
    history: dict[int, np.ndarray]

    @property
    def n_items(self) -> int:
        return len(self.items)

    @property
    def popularity(self) -> np.ndarray:
        return self.items["n_orders"].to_numpy(dtype=float)


def build_reco_data(sales: pd.DataFrame) -> RecoData:
    """Build interaction data from SALE rows (returns excluded). Guest rows count toward popularity only."""
    sales = sales[sales["RecordType"] == "sale"]
    items = (sales.groupby("StockCode").agg(Description=("Description", "first"), n_orders=("InvoiceNo", "nunique"))
             .reset_index().sort_values("StockCode").reset_index(drop=True))
    code_to_idx = {c: i for i, c in enumerate(items["StockCode"])}
    known = sales[sales["HasCustomer"]]
    pairs = known[["CustomerID", "StockCode"]].drop_duplicates()
    cust_ids = np.sort(pairs["CustomerID"].astype("int64").unique())
    cust_idx = {c: i for i, c in enumerate(cust_ids)}
    rows = pairs["CustomerID"].astype("int64").map(cust_idx).to_numpy()
    cols = pairs["StockCode"].map(code_to_idx).to_numpy()
    matrix = sparse.csr_matrix((np.ones(len(pairs), dtype=np.float32), (rows, cols)), shape=(len(cust_ids), len(items)))
    history = {int(c): matrix.indices[matrix.indptr[i]:matrix.indptr[i + 1]].copy() for c, i in cust_idx.items()}
    return RecoData(items, code_to_idx, matrix, cust_ids, history)


def minmax(scores: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Scale scores to [0, 1] over the candidate set (mask=True). Constant scores -> all zeros."""
    out = np.zeros_like(scores, dtype=float)
    cand = np.ones(len(scores), bool) if mask is None else mask
    if cand.sum() == 0:
        return out
    lo, hi = scores[cand].min(), scores[cand].max()
    if hi > lo:
        out[cand] = (scores[cand] - lo) / (hi - lo)
    return out
