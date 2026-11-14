"""Popularity baseline and item-based collaborative filtering.

Collaborative filtering = "learn from the crowd's behaviour, ignore what the product IS".
Two products are SIMILAR when largely the same customers buy both (cosine similarity of their
customer columns). To recommend, add up the similarity of each candidate to everything the
customer already bought: "customers who bought A also bought B".
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.recommendations.common import RecoData


class PopularityRecommender:
    """Baseline: recommend whatever sells in the most orders. Also the cold-start fallback."""

    def fit(self, data: RecoData) -> "PopularityRecommender":
        self.data = data
        self.pop = data.popularity
        return self

    def score_items(self, history: np.ndarray) -> np.ndarray:
        return self.pop.copy()


class CollaborativeRecommender:
    def fit(self, data: RecoData) -> "CollaborativeRecommender":
        self.data = data
        # item x item cosine similarity from the customer x item matrix (float32 keeps memory small)
        self.sim = cosine_similarity(data.matrix.T.tocsr(), dense_output=True).astype(np.float32)
        np.fill_diagonal(self.sim, 0.0)          # a product is not "similar to itself" for recommendation purposes
        return self

    def score_items(self, history: np.ndarray) -> np.ndarray:
        if len(history) == 0:
            return np.zeros(self.data.n_items, dtype=float)
        return self.sim[:, history].sum(axis=1).astype(float)

    def most_similar_owned(self, item_idx: int, history: np.ndarray) -> int | None:
        """Which already-bought product makes ``item_idx`` look relevant (for explanations)."""
        if len(history) == 0:
            return None
        return int(history[np.argmax(self.sim[item_idx, history])])
