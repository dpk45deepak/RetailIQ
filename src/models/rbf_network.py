"""A from-scratch RBF (Radial Basis Function) neural network classifier.

How it works (3 steps)
----------------------
1. HIDDEN LAYER CENTERS: run K-Means on the training data; each cluster centre c_j
   becomes one hidden neuron ("a prototype customer").
2. GAUSSIAN ACTIVATIONS: each hidden neuron answers "how close is this customer to me?"
       phi_j(x) = exp(-gamma * ||x - c_j||^2)      (1 = identical, ~0 = far away)
   ``gamma`` controls how wide each neuron's "neighbourhood" is.
3. OUTPUT LAYER: a regularised logistic regression on the activations phi(x).
   ``C`` is the inverse regularisation strength (small C = stronger regularisation).

Only the output layer is trained by gradient methods; the hidden layer is fixed by
K-Means. That makes RBF networks fast, stable and easy to explain.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.validation import check_array, check_is_fitted


class RBFNetworkClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, n_centers: int = 25, gamma: float = 0.1, C: float = 1.0, random_state: int | None = 42):
        self.n_centers = n_centers
        self.gamma = gamma
        self.C = C
        self.random_state = random_state

    def _activations(self, X: np.ndarray) -> np.ndarray:
        sq_dist = euclidean_distances(X, self.centers_, squared=True)
        return np.exp(-self.gamma * sq_dist)

    def fit(self, X, y):
        X = check_array(X)
        y = np.asarray(y)
        k = min(self.n_centers, len(X))
        self.centers_ = KMeans(n_clusters=k, n_init=5, random_state=self.random_state).fit(X).cluster_centers_
        self.output_layer_ = LogisticRegression(C=self.C, max_iter=2000).fit(self._activations(X), y)
        self.classes_ = self.output_layer_.classes_
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self, ["centers_", "output_layer_"])
        return self.output_layer_.predict_proba(self._activations(check_array(X)))

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
