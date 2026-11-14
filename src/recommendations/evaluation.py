"""Time-aware offline evaluation of recommenders.

Protocol (no peeking at the future)
-----------------------------------
1. Pick a split date. Fit every recommender ONLY on sales before it.
2. For each evaluation customer, the ground truth is the set of NEW products (never bought before, and present in
   the training catalogue) they bought AFTER the split date.
3. Recommend 5 products; score them against the ground truth.

Metrics @k: Precision (how many of the k were right), Recall (how many of the true products were found),
NDCG (rewards putting hits near the top), Coverage (share of the catalogue ever recommended) and popularity bias
(do we only recommend best-sellers?).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.recommendations.common import build_reco_data
from src.recommendations.hybrid import HybridRecommender
from src.utils.config import RANDOM_STATE


def precision_at_k(hits: int, k: int) -> float:
    return hits / k


def recall_at_k(hits: int, n_relevant: int) -> float:
    return hits / n_relevant if n_relevant else 0.0


def ndcg_at_k(is_hit: np.ndarray, n_relevant: int) -> float:
    """Binary-relevance NDCG: DCG of our ranking / DCG of a perfect ranking."""
    k = len(is_hit)
    dcg = float((is_hit / np.log2(np.arange(2, k + 2))).sum())
    ideal = float((1 / np.log2(np.arange(2, min(n_relevant, k) + 2))).sum())
    return dcg / ideal if ideal > 0 else 0.0


def evaluate_recommenders(clean: pd.DataFrame, split_date: pd.Timestamp, k: int = 5, max_customers: int = 800,
                          min_history: int = 3) -> dict:
    train = clean[clean["InvoiceDate"] < split_date]
    future = clean[(clean["InvoiceDate"] >= split_date) & (clean["RecordType"] == "sale") & clean["HasCustomer"]]
    model = HybridRecommender.from_transactions(train)
    data = model.data

    truth: dict[int, set[int]] = {}
    for cid, codes in future.groupby("CustomerID")["StockCode"].unique().items():
        hist = data.history.get(int(cid))
        if hist is None or len(hist) < min_history:
            continue
        new = {data.code_to_idx[c] for c in codes if c in data.code_to_idx} - set(hist)
        if new:
            truth[int(cid)] = new
    rng = np.random.default_rng(RANDOM_STATE)
    cids = sorted(truth)
    if len(cids) > max_customers:
        cids = sorted(rng.choice(cids, max_customers, replace=False).tolist())

    pop_pct = pd.Series(data.popularity).rank(pct=True).to_numpy()           # 0..1 popularity percentile per item
    out = {}
    for strategy in ("popularity", "collaborative", "content", "hybrid"):
        p, r, nd, recommended, pct = [], [], [], [], []
        for cid in cids:
            hist = data.history[cid]
            top = np.argsort(-model.score_items(hist, strategy))[:k]
            hit = np.array([t in truth[cid] for t in top], dtype=float)
            p.append(precision_at_k(int(hit.sum()), k)); r.append(recall_at_k(int(hit.sum()), len(truth[cid])))
            nd.append(ndcg_at_k(hit, len(truth[cid])))
            recommended.extend(top.tolist()); pct.extend(pop_pct[top].tolist())
        out[strategy] = {
            f"precision@{k}": round(float(np.mean(p)), 4), f"recall@{k}": round(float(np.mean(r)), 4),
            f"ndcg@{k}": round(float(np.mean(nd)), 4),
            "coverage": round(len(set(recommended)) / data.n_items, 4),
            "avg_popularity_percentile": round(float(np.mean(pct)), 4),
            "top10pct_item_share": round(float(np.mean(np.array(pct) >= 0.9)), 4),
        }
    actual_pct = float(np.mean([pop_pct[list(truth[c])].mean() for c in cids]))
    return {"split_date": str(split_date), "k": k, "n_eval_customers": len(cids), "n_train_items": data.n_items,
            "actual_purchase_popularity_percentile": round(actual_pct, 4), "models": out,
            "weights": model.weights}
