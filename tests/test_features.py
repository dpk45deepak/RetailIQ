import numpy as np
import pandas as pd
import pytest

from src.data.cleaning import clean_transactions
from src.features.customer_features import FEATURE_COLUMNS, build_customer_features, validate_features
from src.features.rfm import add_rfm_scores, compute_rfm, default_snapshot
from tests.conftest import make_raw

# Customer 1: three orders (Jan 5, Feb 5, Mar 5) + one return in March. Customer 2: one order.
RAW = [
    ("1001", "A", "ITEM A", 2, "2011-01-05 10:00", 5.0, 1, "UK"), ("1001", "X", "ITEM X", 1, "2011-01-05 10:00", 10.0, 1, "UK"),
    ("1002", "A", "ITEM A", 4, "2011-02-05 10:00", 5.0, 1, "UK"),
    ("1003", "C", "ITEM C", 1, "2011-03-05 10:00", 100.0, 1, "UK"),
    ("C1004", "C", "ITEM C", -1, "2011-03-06 10:00", 100.0, 1, "UK"),
    ("1005", "A", "ITEM A", 1, "2011-03-20 10:00", 5.0, 2, "UK"),
]
SNAP = pd.Timestamp("2011-04-01")


@pytest.fixture
def clean():
    return clean_transactions(make_raw(RAW))[0]


def test_rfm_values(clean):
    rfm = compute_rfm(clean, SNAP)
    c1 = rfm.loc[1]
    assert c1["recency_days"] == (SNAP - pd.Timestamp("2011-03-05 10:00")).days
    assert c1["frequency"] == 3                                   # return invoice is not an order
    assert c1["monetary"] == pytest.approx(2 * 5 + 10 + 4 * 5 + 100 - 100)   # net of the return
    assert rfm.loc[2, "frequency"] == 1


def test_feature_values(clean):
    f = build_customer_features(clean, SNAP)
    c1 = f.loc[1]
    assert c1["total_items"] == 2 + 1 + 4 + 1
    assert c1["product_diversity"] == 3
    assert c1["active_months"] == 3
    assert c1["avg_basket_size"] == pytest.approx((2 + 1 + 1) / 3)          # distinct products per order
    assert c1["avg_days_between_purchases"] == pytest.approx(((pd.Timestamp("2011-03-05 10:00") - pd.Timestamp("2011-01-05 10:00")).days) / 2)
    assert np.isnan(f.loc[2, "avg_days_between_purchases"])               # single order -> undefined
    assert c1["spend_30d"] == pytest.approx(0.0)                            # +100 sale, -100 return inside 30 days
    assert c1["spend_60d"] == pytest.approx(20.0)                           # Feb 5 order (20) + March sale and its return (net 0)
    assert c1["spend_90d"] == pytest.approx(40.0)                           # everything
    assert set(FEATURE_COLUMNS) <= set(f.columns)
    assert validate_features(f) == []


def test_no_future_leakage(clean):
    """Features at a cutoff must be identical whether or not later transactions exist."""
    cutoff = pd.Timestamp("2011-03-01")
    full = build_customer_features(clean, cutoff)
    past_only = build_customer_features(clean[clean["InvoiceDate"] < cutoff], cutoff)
    pd.testing.assert_frame_equal(full, past_only)
    assert 2 not in full.index                                    # customer 2 first buys after the cutoff


def test_snapshot_is_exclusive(clean):
    f = build_customer_features(clean, pd.Timestamp("2011-03-05 10:00"))      # an order exactly at the cutoff is NOT visible
    assert f.loc[1, "frequency"] == 2


def test_default_snapshot_is_day_after_last_transaction(clean):
    assert default_snapshot(clean) == pd.Timestamp("2011-03-21")


def test_validate_features_catches_problems(clean):
    f = build_customer_features(clean, SNAP)
    bad = f.copy(); bad.loc[1, "recency_days"] = -5
    assert any("negative recency" in i for i in validate_features(bad))
    bad = f.copy(); bad.loc[2, "frequency"] = 0
    assert any("zero orders" in i for i in validate_features(bad))


def test_rfm_scores_range(clean_synth):
    rfm = add_rfm_scores(compute_rfm(clean_synth, default_snapshot(clean_synth)))
    for c in ("R_score", "F_score", "M_score"):
        assert rfm[c].between(1, 5).all()
    assert rfm["RFM_score"].str.len().eq(3).all()


def test_index_dtype_is_stable_after_parquet_round_trip(clean, tmp_path):
    """Regression: features built from a parquet-loaded table must have the same int64 index as from memory."""
    clean.to_parquet(tmp_path / "c.parquet", index=False)
    reloaded = pd.read_parquet(tmp_path / "c.parquet")
    a, b = build_customer_features(clean, SNAP), build_customer_features(reloaded, SNAP)
    assert a.index.dtype == b.index.dtype == np.dtype("int64")
    pd.testing.assert_frame_equal(a, b)
