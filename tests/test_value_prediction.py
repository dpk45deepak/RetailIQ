import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score

from src.features.customer_features import FEATURE_COLUMNS, build_customer_features
from src.models.rbf_network import RBFNetworkClassifier
from src.models.value_prediction import (MIN_PRECISION, MIN_RECALL, best_f1_threshold, build_split, evaluate, make_windows, signed_log1p, train_and_select)


def test_windows_are_consecutive_and_cover_end(clean_synth):
    w = make_windows(clean_synth, 90)
    h = pd.Timedelta(days=90)
    assert w.val - w.train == h and w.test - w.val == h and w.end - w.test == h


def test_label_comes_from_the_future_only(clean_synth):
    w = make_windows(clean_synth, 90)
    split = build_split(clean_synth, w.val, 90)
    # 1. positive rate ~ top 20%
    assert 0.1 <= split["high_value"].mean() <= 0.35
    # 2. features are identical when ALL post-cutoff data is deleted -> label cannot have leaked into features
    past = clean_synth[clean_synth["InvoiceDate"] < w.val]
    pd.testing.assert_frame_equal(build_customer_features(clean_synth, w.val)[FEATURE_COLUMNS],
                                  build_customer_features(past, w.val)[FEATURE_COLUMNS])
    # 3. label is not a function of the features that were given to the model
    assert "future_revenue" not in FEATURE_COLUMNS and "high_value" not in FEATURE_COLUMNS


def test_signed_log1p_handles_negatives():
    x = np.array([-100.0, 0.0, 100.0])
    y = signed_log1p(x)
    assert y[1] == 0 and y[0] == -y[2] and np.isfinite(y).all()


def test_rbf_network_learns_nonlinear_boundary():
    X, y = make_classification(n_samples=600, n_features=6, n_informative=4, random_state=0)
    Xtr, Xte, ytr, yte = X[:400], X[400:], y[:400], y[400:]
    m = RBFNetworkClassifier(n_centers=20, gamma=0.05, C=1.0, random_state=0).fit(Xtr, ytr)
    p = m.predict_proba(Xte)
    assert p.shape == (200, 2) and np.allclose(p.sum(axis=1), 1)
    assert roc_auc_score(yte, p[:, 1]) > 0.8
    assert m.centers_.shape == (20, 6)
    again = RBFNetworkClassifier(n_centers=20, gamma=0.05, C=1.0, random_state=0).fit(Xtr, ytr)
    np.testing.assert_allclose(again.predict_proba(Xte), p)                    # reproducible


def test_activations_are_between_zero_and_one():
    X = np.random.default_rng(0).normal(size=(50, 3))
    m = RBFNetworkClassifier(n_centers=5, gamma=0.5).fit(X, (X[:, 0] > 0).astype(int))
    a = m._activations(X)
    assert a.min() >= 0 and a.max() <= 1


def test_threshold_and_metrics():
    y = np.array([0, 0, 0, 1, 1, 1]); p = np.array([.1, .2, .6, .4, .7, .9])
    thr = best_f1_threshold(y, p)
    r = evaluate(y, p, thr)
    assert {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "confusion_matrix"} <= set(r)
    assert np.array(r["confusion_matrix"]).sum() == 6


def test_experiment_respects_the_protocol(pipeline_root):
    import json
    m = json.loads((pipeline_root / "models" / "value_metrics.json").read_text())
    assert set(m["models"]) == {"Logistic Regression", "Random Forest", "XGBoost", "RBF Network", "MLP Classifier"}
    # model selection must be explainable from VALIDATION numbers alone
    eligible = [n for n, r in m["models"].items() if r["validation"]["precision"] >= MIN_PRECISION and r["validation"]["recall"] >= MIN_RECALL]
    pool = eligible or list(m["models"])
    assert m["selected_model"] == max(pool, key=lambda n: m["models"][n]["validation"]["pr_auc"])
    # threshold of every model equals the one tuned on validation (test uses the same one)
    for r in m["models"].values():
        assert r["validation"]["threshold"] == r["test"]["threshold"]
    w = m["windows"]
    assert w["train"] < w["val"] < w["test"] < w["end"]
