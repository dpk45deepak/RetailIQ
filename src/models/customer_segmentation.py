"""Unsupervised customer segmentation with K-Means.

Pipeline: select RFM -> log1p (tames skew) -> StandardScaler (equal footing)
          -> K-Means -> profile clusters -> name them with business labels.

Unsupervised = there is no "right answer" column. The algorithm only looks for
customers who behave alike. We then interpret the groups ourselves.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.utils.config import RANDOM_STATE

SEGMENT_FEATURES = ["recency_days", "frequency", "monetary"]

# Target profile of each business segment as percentile scores in [0, 1] of the
# whole customer base (recency is inverted: 1.0 = bought very recently).
ARCHETYPES: dict[str, dict[str, float]] = {
    "Champions":             {"recency": 0.95, "frequency": 0.95, "monetary": 0.95, "tenure": 0.85},
    "Loyal Customers":       {"recency": 0.70, "frequency": 0.80, "monetary": 0.75, "tenure": 0.80},
    "Potential Loyalists":   {"recency": 0.80, "frequency": 0.40, "monetary": 0.45, "tenure": 0.45},
    "New Customers":         {"recency": 0.95, "frequency": 0.10, "monetary": 0.15, "tenure": 0.05},
    "At-Risk Customers":     {"recency": 0.25, "frequency": 0.70, "monetary": 0.70, "tenure": 0.80},
    "Hibernating Customers": {"recency": 0.05, "frequency": 0.15, "monetary": 0.15, "tenure": 0.50},
}

SEGMENT_PLAYBOOK: dict[str, dict[str, str]] = {
    "Champions": {"who": "Buy often, spend the most, bought very recently.",
                  "action": "Reward and recruit: early access, loyalty perks, referral asks. Avoid discounting - they already buy."},
    "Loyal Customers": {"who": "Regular, dependable buyers with solid spend.",
                        "action": "Upsell and cross-sell with bundles; invite into the loyalty programme."},
    "Potential Loyalists": {"who": "Recent buyers with moderate frequency and spend.",
                            "action": "Nudge a second and third order: replenishment reminders and product-affinity emails."},
    "New Customers": {"who": "Very recent first-timers.",
                      "action": "Onboard well: welcome series, a clear reason to return within 30 days."},
    "At-Risk Customers": {"who": "Used to spend a lot but have gone quiet.",
                          "action": "Win-back now: personalised outreach, 'we miss you' offer, ask for feedback."},
    "Hibernating Customers": {"who": "Low spend, long inactive.",
                              "action": "Low-cost reactivation only (one email wave); suppress from paid campaigns afterwards."},
}


def prepare_matrix(feats: pd.DataFrame) -> np.ndarray:
    """log1p-transform the RFM columns (net spend floored at 0) before scaling."""
    x = feats[SEGMENT_FEATURES].astype(float).copy()
    x["monetary"] = x["monetary"].clip(lower=0)
    return np.log1p(x.to_numpy())


def evaluate_k(x_scaled: np.ndarray, k_range=range(2, 11), random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Elbow (inertia) and silhouette for each candidate number of clusters."""
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(x_scaled)
        sil = silhouette_score(x_scaled, km.labels_, sample_size=min(3000, len(x_scaled)), random_state=random_state)
        rows.append({"k": k, "inertia": float(km.inertia_), "silhouette": float(sil)})
    return pd.DataFrame(rows)


def choose_k(metrics: pd.DataFrame, k_min: int = 3, k_max: int = 6, tolerance: float = 0.02) -> int:
    """Best silhouette in [k_min, k_max] (max 6 = number of business labels); among near-ties (within ``tolerance``) prefer the larger k.

    Documented rule: 2 clusters are rarely useful to a marketing team, and a slightly lower
    silhouette is an acceptable price for a more actionable split. We never force six.
    """
    m = metrics[(metrics.k >= k_min) & (metrics.k <= k_max)]
    best = m["silhouette"].max()
    return int(m[m["silhouette"] >= best - tolerance]["k"].max())


def _percentile(series: pd.Series, value: float) -> float:
    return float((series <= value).mean())


def name_clusters(feats: pd.DataFrame, labels: np.ndarray) -> dict[int, str]:
    """Match each cluster to the closest business archetype (Hungarian assignment => unique names)."""
    df = feats.assign(cluster=labels)
    med = df.groupby("cluster")[["recency_days", "frequency", "monetary", "tenure_days"]].median()
    names = list(ARCHETYPES)
    cols = names + [f"{n} II" for n in names]          # extra slots if k > 6
    cost = np.zeros((len(med), len(cols)))
    for i, (_, row) in enumerate(med.iterrows()):
        score = {
            "recency": 1 - _percentile(feats["recency_days"], row["recency_days"]),
            "frequency": _percentile(feats["frequency"], row["frequency"]),
            "monetary": _percentile(feats["monetary"], row["monetary"]),
            "tenure": _percentile(feats["tenure_days"], row["tenure_days"]),
        }
        for j, name in enumerate(cols):
            target = ARCHETYPES[name.replace(" II", "")]
            cost[i, j] = sum((score[k] - target[k]) ** 2 for k in score) + (0.15 if name.endswith(" II") else 0)
    rows, assigned = linear_sum_assignment(cost)
    return {int(med.index[r]): cols[c] for r, c in zip(rows, assigned)}


def fit_segmentation(feats: pd.DataFrame, k: int | None = None, random_state: int = RANDOM_STATE) -> dict:
    """Fit the full segmentation bundle. ``k=None`` selects k automatically."""
    x_log = prepare_matrix(feats)
    scaler = StandardScaler().fit(x_log)
    x = scaler.transform(x_log)
    metrics = evaluate_k(x, random_state=random_state)
    k = k or choose_k(metrics)
    km = KMeans(n_clusters=k, n_init=20, random_state=random_state).fit(x)
    cluster_to_name = name_clusters(feats, km.labels_)
    pca = PCA(n_components=2, random_state=random_state).fit(x)
    return {"scaler": scaler, "kmeans": km, "pca": pca, "k": k, "features": SEGMENT_FEATURES,
            "cluster_to_name": cluster_to_name, "k_metrics": metrics.to_dict("records"),
            "silhouette": float(metrics.loc[metrics.k == k, "silhouette"].iloc[0])}


def assign_segments(bundle: dict, feats: pd.DataFrame) -> pd.DataFrame:
    """Return cluster id, segment name and 2-D PCA coordinates for each customer."""
    x = bundle["scaler"].transform(prepare_matrix(feats))
    cluster = bundle["kmeans"].predict(x)
    coords = bundle["pca"].transform(x)
    return pd.DataFrame({"cluster": cluster, "segment": pd.Series(cluster, index=feats.index).map(bundle["cluster_to_name"]),
                         "pc1": coords[:, 0], "pc2": coords[:, 1]}, index=feats.index)


def segment_profile(feats: pd.DataFrame, segments: pd.Series) -> pd.DataFrame:
    """One row per segment: size, medians of R/F/M and revenue share."""
    df = feats.assign(segment=segments)
    prof = df.groupby("segment").agg(
        customers=("frequency", "size"), recency_median=("recency_days", "median"),
        frequency_median=("frequency", "median"), monetary_median=("monetary", "median"),
        avg_order_value=("avg_order_value", "mean"), revenue=("monetary", "sum"),
        tenure_median=("tenure_days", "median"), p_return=("return_rate", "mean"))
    prof["customer_share"] = prof["customers"] / prof["customers"].sum()
    prof["revenue_share"] = prof["revenue"] / prof["revenue"].sum()
    return prof.sort_values("revenue", ascending=False)


def save_bundle(bundle: dict, path: Path) -> None:
    joblib.dump(bundle, path)


def load_bundle(path: Path) -> dict:
    return joblib.load(path)
