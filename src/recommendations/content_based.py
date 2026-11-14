"""Content-based filtering from product descriptions.

TF-IDF turns every description into a vector: words that are rare across the catalogue (e.g. "BOTANICAL")
get high weight, words in almost every product get low weight. Cosine similarity then measures how alike
two descriptions are. A customer profile = the average vector of the products they bought.
Needs no other customers' data, so it also works for brand-new products.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.recommendations.common import RecoData


class ContentRecommender:
    def fit(self, data: RecoData) -> "ContentRecommender":
        self.data = data
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.tfidf = self.vectorizer.fit_transform(data.items["Description"])   # rows are L2-normalised
        return self

    def score_items(self, history: np.ndarray) -> np.ndarray:
        if len(history) == 0:
            return np.zeros(self.data.n_items, dtype=float)
        profile = np.asarray(self.tfidf[history].mean(axis=0)).ravel()         # customer taste vector
        return np.asarray(self.tfidf @ profile).ravel()                        # cosine up to a constant per customer

    def most_similar_owned(self, item_idx: int, history: np.ndarray) -> int | None:
        if len(history) == 0:
            return None
        sims = (self.tfidf[history] @ self.tfidf[item_idx].T).toarray().ravel()
        return int(history[np.argmax(sims)])

    def similar_items(self, item_idx: int, n: int = 5) -> list[int]:
        """Pure item-to-item lookup: 'products that read like this one'."""
        sims = (self.tfidf @ self.tfidf[item_idx].T).toarray().ravel()
        sims[item_idx] = -1
        return [int(i) for i in np.argsort(-sims)[:n]]
