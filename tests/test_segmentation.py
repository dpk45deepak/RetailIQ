import numpy as np
import pandas as pd

from src.features.customer_features import build_customer_features
from src.features.rfm import default_snapshot
from src.models.customer_segmentation import (ARCHETYPES, assign_segments, choose_k, evaluate_k, fit_segmentation, prepare_matrix, segment_profile)


def _feats(clean):
    return build_customer_features(clean, default_snapshot(clean))


def test_prepare_matrix_is_finite_even_with_negative_spend(clean_synth):
    f = _feats(clean_synth); f.iloc[0, f.columns.get_loc("monetary")] = -50
    x = prepare_matrix(f)
    assert np.isfinite(x).all() and x.shape == (len(f), 3)


def test_choose_k_prefers_best_silhouette_within_bounds():
    m = pd.DataFrame({"k": range(2, 9), "silhouette": [.9, .30, .31, .45, .44, .2, .1]})
    assert choose_k(m) in (5, 6)             # k=2 is excluded; 5 (best) or 6 (within tolerance, larger k wins)
    assert 3 <= choose_k(m) <= 6


def test_segmentation_bundle(clean_synth):
    f = _feats(clean_synth)
    b = fit_segmentation(f)
    assert 3 <= b["k"] <= 6                                   # never forced to six
    names = list(b["cluster_to_name"].values())
    assert len(names) == len(set(names)) == b["k"]            # unique business labels
    assert set(names) <= set(ARCHETYPES)
    seg = assign_segments(b, f)
    assert len(seg) == len(f) and seg["segment"].notna().all() and list(seg.columns) == ["cluster", "segment", "pc1", "pc2"]


def test_segmentation_is_reproducible(clean_synth):
    f = _feats(clean_synth)
    a, b = fit_segmentation(f), fit_segmentation(f)
    assert (assign_segments(a, f)["cluster"] == assign_segments(b, f)["cluster"]).all()


def test_labels_make_business_sense(clean_synth):
    f = _feats(clean_synth)
    b = fit_segmentation(f)
    prof = segment_profile(f, assign_segments(b, f)["segment"])
    assert prof["customers"].sum() == len(f)
    assert prof["revenue_share"].sum() == np.float64(1.0) or abs(prof["revenue_share"].sum() - 1) < 1e-9
    if {"Champions", "Hibernating Customers"} <= set(prof.index):
        assert prof.loc["Champions", "recency_median"] < prof.loc["Hibernating Customers", "recency_median"]
        assert prof.loc["Champions", "frequency_median"] > prof.loc["Hibernating Customers", "frequency_median"]


def test_evaluate_k_shapes(clean_synth):
    m = evaluate_k(np.random.default_rng(0).normal(size=(200, 3)), range(2, 5))
    assert list(m["k"]) == [2, 3, 4] and (m["inertia"].diff().dropna() < 0).all()
