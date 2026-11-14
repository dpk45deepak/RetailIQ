"""Customer value prediction: "Will this customer be a HIGH-VALUE customer in the next 90 days?"

LEAKAGE-SAFE DESIGN (the most important idea in this file)
----------------------------------------------------------
Time is split into three consecutive 90-day windows. For each window we pretend to stand on its first
day (the *cutoff*), compute features ONLY from the past, and label customers using the 90 days AFTER:

    train cutoff T1 --90d--> val cutoff T2 --90d--> test cutoff T3 --90d--> end of data
       features(<T1)  label(T1..T2)   features(<T2)  label(T2..T3)   features(<T3)  label(T3..end)

* Features never see the label period.  * The test period is touched once, at the very end.
* Hyper-parameters, the decision threshold and the winning model are chosen on VALIDATION only.

LABEL: ``high_value`` = customer is in the top 20% by NET revenue (sales minus returns) in the next 90 days,
among customers who had bought before the cutoff. A per-window quantile (rather than a fixed GBP value) keeps
the positive rate comparable across seasons (the test window is Christmas, the train window is not).
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBClassifier

from src.features.customer_features import FEATURE_COLUMNS, build_customer_features
from src.features.rfm import default_snapshot
from src.models.rbf_network import RBFNetworkClassifier
from src.utils.config import HIGH_VALUE_TOP_PCT, HORIZON_DAYS, RANDOM_STATE, get_logger

log = get_logger(__name__)

# Business requirement for model selection: when we contact a customer list it must be (a) reasonably pure and
# (b) cover a meaningful share of the true high-value customers.
MIN_PRECISION, MIN_RECALL = 0.40, 0.50


def signed_log1p(x: np.ndarray) -> np.ndarray:
    """log1p that also works for negative numbers (net spend can be negative). Tames heavy right tails."""
    return np.sign(x) * np.log1p(np.abs(x))


def make_preprocessor() -> Pipeline:
    """Impute (median + missing indicator) -> signed log -> standardise. Fitted on TRAIN only."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("log", FunctionTransformer(signed_log1p, validate=False)),
        ("scale", StandardScaler()),
    ])


@dataclass
class Windows:
    train: pd.Timestamp
    val: pd.Timestamp
    test: pd.Timestamp
    end: pd.Timestamp
    horizon: int

    def as_dict(self) -> dict:
        return {k: str(v) for k, v in self.__dict__.items() if k != "horizon"} | {"horizon_days": self.horizon}


def make_windows(clean: pd.DataFrame, horizon: int = HORIZON_DAYS) -> Windows:
    end = default_snapshot(clean)
    h = pd.Timedelta(days=horizon)
    return Windows(train=end - 3 * h, val=end - 2 * h, test=end - h, end=end, horizon=horizon)


def future_net_revenue(clean: pd.DataFrame, cutoff: pd.Timestamp, horizon: int) -> pd.Series:
    """Net revenue per customer in [cutoff, cutoff + horizon). This is the LABEL source, never a feature."""
    win = clean[(clean["InvoiceDate"] >= cutoff) & (clean["InvoiceDate"] < cutoff + pd.Timedelta(days=horizon)) & clean["HasCustomer"]]
    s = win.groupby("CustomerID")["Revenue"].sum()
    s.index = s.index.astype("int64")
    return s


def build_split(clean: pd.DataFrame, cutoff: pd.Timestamp, horizon: int, top_pct: float = HIGH_VALUE_TOP_PCT) -> pd.DataFrame:
    """Features at ``cutoff`` + label from the following ``horizon`` days."""
    feats = build_customer_features(clean, cutoff)
    fut = future_net_revenue(clean, cutoff, horizon).reindex(feats.index).fillna(0.0)
    thr = float(np.quantile(fut, 1 - top_pct))
    out = feats[FEATURE_COLUMNS].copy()
    out["future_revenue"] = fut
    out["high_value"] = (fut >= max(thr, 0.01)).astype(int)
    out.attrs["threshold_gbp"] = thr
    return out


def param_grids(fast: bool = False) -> dict[str, tuple[type, dict, dict]]:
    """name -> (estimator class, fixed params, grid of params to search on validation data)."""
    rs = {"random_state": RANDOM_STATE}
    g = {
        "Logistic Regression": (LogisticRegression, {"max_iter": 2000, **rs}, {"C": [0.01, 0.1, 1, 10]}),
        "Random Forest": (RandomForestClassifier, {"n_estimators": 300, "n_jobs": -1, **rs},
                          {"max_depth": [4, 8, 12], "min_samples_leaf": [3, 10]}),
        "XGBoost": (XGBClassifier, {"subsample": 0.8, "colsample_bytree": 0.8, "eval_metric": "logloss", "n_jobs": 4, **rs},
                    {"n_estimators": [150, 300], "max_depth": [2, 3, 4], "learning_rate": [0.05, 0.1]}),
        "RBF Network": (RBFNetworkClassifier, rs, {"n_centers": [10, 25, 50], "gamma": [0.01, 0.03, 0.1, 0.3], "C": [0.1, 1, 10]}),
        "MLP Classifier": (MLPClassifier, {"max_iter": 500, "early_stopping": True, **rs},
                           {"hidden_layer_sizes": [(32,), (64, 32)], "alpha": [1e-4, 1e-2]}),
    }
    if fast:
        g["Random Forest"] = (RandomForestClassifier, {"n_estimators": 100, "n_jobs": -1, **rs}, {"max_depth": [6], "min_samples_leaf": [5]})
        g["XGBoost"] = (XGBClassifier, g["XGBoost"][1], {"n_estimators": [100], "max_depth": [3], "learning_rate": [0.1]})
        g["RBF Network"] = (RBFNetworkClassifier, rs, {"n_centers": [15], "gamma": [0.05, 0.2], "C": [1]})
        g["MLP Classifier"] = (MLPClassifier, g["MLP Classifier"][1], {"hidden_layer_sizes": [(16,)], "alpha": [1e-3]})
    return g


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[int(np.argmax(f1))])


def _thin(a: np.ndarray, n: int = 120) -> list[float]:
    idx = np.unique(np.linspace(0, len(a) - 1, min(n, len(a))).astype(int))
    return [round(float(v), 4) for v in a[idx]]


def evaluate(y: np.ndarray, p: np.ndarray, threshold: float, with_curves: bool = False) -> dict:
    pred = (p >= threshold).astype(int)
    out = {
        "threshold": round(threshold, 4), "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0), "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }
    out = {k: (round(float(v), 4) if not isinstance(v, (list,)) else v) for k, v in out.items()}
    if with_curves:
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        out["roc_curve"] = {"fpr": _thin(fpr), "tpr": _thin(tpr)}
        out["pr_curve"] = {"precision": _thin(prec), "recall": _thin(rec)}
    return out


def train_and_select(clean: pd.DataFrame, fast: bool = False, horizon: int = HORIZON_DAYS, tracker=None) -> dict:
    """Run the whole experiment. Returns {bundle, metrics}; ``bundle`` is what gets saved for serving."""
    w = make_windows(clean, horizon)
    train, val, test = (build_split(clean, c, horizon) for c in (w.train, w.val, w.test))
    prep = make_preprocessor().fit(train[FEATURE_COLUMNS])            # fit on TRAIN only
    Xtr, Xva, Xte = (prep.transform(d[FEATURE_COLUMNS]) for d in (train, val, test))
    ytr, yva, yte = (d["high_value"].to_numpy() for d in (train, val, test))
    log.info("Split sizes train/val/test = %d/%d/%d | positive rates %.2f/%.2f/%.2f", len(train), len(val), len(test), ytr.mean(), yva.mean(), yte.mean())

    results: dict[str, dict] = {}
    fitted: dict[str, object] = {}
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    for name, (cls, fixed, grid) in param_grids(fast).items():
        best = None
        for combo in itertools.product(*grid.values()):
            params = dict(zip(grid, combo))
            est = cls(**fixed, **params).fit(Xtr, ytr)
            score = average_precision_score(yva, est.predict_proba(Xva)[:, 1])   # tuned on VALIDATION PR-AUC
            if best is None or score > best[0]:
                best = (score, params, est)
        _, params, est = best
        p_val, p_te = est.predict_proba(Xva)[:, 1], est.predict_proba(Xte)[:, 1]
        thr = best_f1_threshold(yva, p_val)                                          # threshold from VALIDATION
        cv_auc = cross_val_score(cls(**fixed, **params), Xtr, ytr, cv=cv, scoring="roc_auc")
        results[name] = {"best_params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()},
                         "validation": evaluate(yva, p_val, thr, with_curves=True),
                         "test": evaluate(yte, p_te, thr, with_curves=True),
                         "cv_roc_auc_train_mean": round(float(cv_auc.mean()), 4), "cv_roc_auc_train_std": round(float(cv_auc.std()), 4)}
        fitted[name] = est
        log.info("%-20s val PR-AUC %.3f | test ROC-AUC %.3f", name, results[name]["validation"]["pr_auc"], results[name]["test"]["roc_auc"])
        if tracker:
            tracker(name, params, results[name])

    eligible = [n for n, r in results.items() if r["validation"]["precision"] >= MIN_PRECISION and r["validation"]["recall"] >= MIN_RECALL]
    pool = eligible or list(results)
    selected = max(pool, key=lambda n: results[n]["validation"]["pr_auc"])        # VALIDATION decides, never test

    pipeline = Pipeline([("prep", prep), ("model", fitted[selected])])
    bundle = {"pipeline": pipeline, "model_name": selected, "threshold": results[selected]["validation"]["threshold"],
              "features": FEATURE_COLUMNS, "horizon_days": horizon, "top_pct": HIGH_VALUE_TOP_PCT}
    metrics = {
        "selected_model": selected, "selection_rule": f"highest validation PR-AUC among models with validation precision>={MIN_PRECISION} and recall>={MIN_RECALL}"
                                                     + ("" if eligible else " (no model met the business floor; fell back to all models)"),
        "windows": w.as_dict(), "label_rule": f"top {int(HIGH_VALUE_TOP_PCT * 100)}% of customers by net revenue in the next {horizon} days",
        "sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "positive_rate": {"train": round(float(ytr.mean()), 4), "val": round(float(yva.mean()), 4), "test": round(float(yte.mean()), 4)},
        "revenue_threshold_gbp": {"train": round(train.attrs["threshold_gbp"], 2), "val": round(val.attrs["threshold_gbp"], 2), "test": round(test.attrs["threshold_gbp"], 2)},
        "models": results,
    }
    return {"bundle": bundle, "metrics": metrics, "data": {"train": train, "val": val, "test": test},
            "fitted": fitted, "prep": prep}


def score_customers(bundle: dict, feats: pd.DataFrame) -> pd.Series:
    """Probability that each customer is high-value over the next horizon."""
    p = bundle["pipeline"].predict_proba(feats[bundle["features"]])[:, 1]
    return pd.Series(p, index=feats.index, name="p_high_value")


def save_metrics(metrics: dict, path) -> None:
    path.write_text(json.dumps(metrics, indent=2, default=str))
