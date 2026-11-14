"""Hybrid recommender: blend collaborative, content and popularity signals.

    final_score = 0.60 * collaborative + 0.30 * content + 0.10 * popularity

Each component is first min-max normalised to [0, 1] over the candidate products (otherwise the component with
the biggest raw numbers would silently dominate). Products the customer already bought are never recommended.

Cold start: a customer with no purchase history gets the most popular products (flagged ``fallback=True``).
A short list of basket items can be passed for anonymous visitors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.recommendations.collaborative import CollaborativeRecommender, PopularityRecommender
from src.recommendations.common import RecoData, build_reco_data, minmax
from src.recommendations.content_based import ContentRecommender

WEIGHTS = {"collaborative": 0.60, "content": 0.30, "popularity": 0.10}
STRATEGIES = ("hybrid", "collaborative", "content", "popularity")


class HybridRecommender:
    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = weights or dict(WEIGHTS)
        if abs(sum(self.weights.values()) - 1.0) > 1e-9:
            raise ValueError("hybrid weights must sum to 1")

    @classmethod
    def from_transactions(cls, sales: pd.DataFrame, weights: dict[str, float] | None = None) -> "HybridRecommender":
        return cls(weights).fit(build_reco_data(sales))

    def fit(self, data: RecoData) -> "HybridRecommender":
        self.data = data
        self.pop = PopularityRecommender().fit(data)
        self.cf = CollaborativeRecommender().fit(data)
        self.cb = ContentRecommender().fit(data)
        return self

    # -- scoring -------------------------------------------------------------------------------
    def components(self, history: np.ndarray) -> dict[str, np.ndarray]:
        mask = np.ones(self.data.n_items, bool)
        mask[history] = False                                  # candidates = not yet bought
        raw = {"collaborative": self.cf.score_items(history), "content": self.cb.score_items(history),
               "popularity": self.pop.score_items(history)}
        return {k: minmax(v, mask) for k, v in raw.items()} | {"_mask": mask}

    def score_items(self, history: np.ndarray, strategy: str = "hybrid") -> np.ndarray:
        comp = self.components(history)
        if strategy == "hybrid":
            s = sum(self.weights[k] * comp[k] for k in self.weights)
        elif strategy in comp:
            s = comp[strategy]
        else:
            raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")
        return np.where(comp["_mask"], s, -np.inf)

    def history_for(self, customer_id: int | None = None, basket: list[str] | None = None) -> np.ndarray:
        idx = set(self.data.history.get(int(customer_id), [])) if customer_id is not None else set()
        idx |= {self.data.code_to_idx[c] for c in (basket or []) if c in self.data.code_to_idx}
        return np.array(sorted(idx), dtype=int)

    # -- public API ----------------------------------------------------------------------------
    def recommend(self, customer_id: int | None = None, n: int = 5, strategy: str = "hybrid",
                  basket: list[str] | None = None) -> pd.DataFrame:
        """Top-n recommendations with component scores and a human-readable reason."""
        history = self.history_for(customer_id, basket)
        fallback = len(history) == 0
        if fallback:
            strategy = "popularity"
        scores = self.score_items(history, strategy)
        top = np.argsort(-scores)[:n]
        comp = self.components(history)
        rows = []
        for i in top:
            if not np.isfinite(scores[i]):
                continue
            rows.append({
                "stock_code": self.data.items.at[i, "StockCode"], "description": self.data.items.at[i, "Description"],
                "score": float(scores[i]), "collaborative": float(comp["collaborative"][i]),
                "content": float(comp["content"][i]), "popularity": float(comp["popularity"][i]),
                "reason": self._reason(i, history, comp, strategy, fallback), "fallback": fallback})
        return pd.DataFrame(rows)

    def _reason(self, i: int, history: np.ndarray, comp: dict, strategy: str, fallback: bool) -> str:
        if fallback:
            return "No purchase history yet, so we show what is popular with all customers."
        weights = self.weights if strategy == "hybrid" else {strategy: 1.0}
        main = max(weights, key=lambda k: weights[k] * comp[k][i])
        desc = lambda j: self.data.items.at[j, "Description"].title()
        if main == "collaborative":
            j = self.cf.most_similar_owned(i, history)
            return f"Customers who bought '{desc(j)}' often buy this too." if j is not None else "Often bought together with your products."
        if main == "content":
            j = self.cb.most_similar_owned(i, history)
            return f"Similar in style to '{desc(j)}', which you bought." if j is not None else "Similar to products you bought."
        return "A best-seller across the whole customer base."
