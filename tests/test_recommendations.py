import numpy as np
import pandas as pd
import pytest

from src.recommendations.common import build_reco_data, minmax
from src.recommendations.evaluation import evaluate_recommenders, ndcg_at_k, precision_at_k, recall_at_k
from src.recommendations.hybrid import STRATEGIES, WEIGHTS, HybridRecommender


@pytest.fixture(scope="module")
def rec(clean_synth):
    return HybridRecommender.from_transactions(clean_synth)


@pytest.fixture(scope="module")
def a_customer(rec):
    return int(rec.data.customer_ids[0])


def test_metric_formulas():
    assert precision_at_k(2, 5) == 0.4 and recall_at_k(2, 4) == 0.5 and recall_at_k(0, 0) == 0.0
    assert ndcg_at_k(np.array([1, 0, 0, 0, 0]), 1) == pytest.approx(1.0)          # perfect: hit at rank 1
    assert ndcg_at_k(np.array([0, 1, 0, 0, 0]), 1) == pytest.approx(1 / np.log2(3))
    assert ndcg_at_k(np.zeros(5), 3) == 0.0


def test_weights_follow_the_specification():
    assert WEIGHTS == {"collaborative": 0.60, "content": 0.30, "popularity": 0.10}
    with pytest.raises(ValueError):
        HybridRecommender({"collaborative": .5, "content": .5, "popularity": .5})


def test_minmax_normalisation():
    s = minmax(np.array([5., 10., 20., 100.]), np.array([True, True, True, False]))
    assert s[:3].tolist() == [0.0, pytest.approx(1 / 3), 1.0] and s[3] == 0.0
    assert minmax(np.ones(4)).sum() == 0                                              # constant -> zeros, no NaN


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_never_recommends_already_bought(rec, a_customer, strategy):
    out = rec.recommend(a_customer, n=10, strategy=strategy)
    bought = {rec.data.items.at[i, "StockCode"] for i in rec.data.history[a_customer]}
    assert len(out) == 10 and not set(out["stock_code"]) & bought
    assert out["score"].is_monotonic_decreasing


def test_component_scores_are_normalised(rec, a_customer):
    out = rec.recommend(a_customer, n=10)
    for c in ("score", "collaborative", "content", "popularity"):
        assert out[c].between(0, 1).all()
    expected = 0.6 * out["collaborative"] + 0.3 * out["content"] + 0.1 * out["popularity"]
    np.testing.assert_allclose(out["score"], expected, atol=1e-9)


def test_cold_start_falls_back_to_popularity(rec):
    out = rec.recommend(customer_id=99999999, n=5)
    assert out["fallback"].all() and len(out) == 5
    top_pop = rec.data.items.sort_values("n_orders", ascending=False)["StockCode"].head(5)
    assert set(out["stock_code"]) == set(top_pop)


def test_basket_for_anonymous_visitors(rec):
    code = rec.data.items["StockCode"].iloc[0]
    out = rec.recommend(customer_id=None, basket=[code], n=5)
    assert not out["fallback"].any() and code not in set(out["stock_code"])


def test_unknown_strategy_rejected(rec, a_customer):
    with pytest.raises(ValueError):
        rec.recommend(a_customer, strategy="magic")


def test_reasons_are_human_readable(rec, a_customer):
    assert rec.recommend(a_customer, n=3)["reason"].str.len().gt(10).all()


def test_evaluation_is_time_aware(clean_synth):
    split = clean_synth["InvoiceDate"].max().normalize() - pd.Timedelta(days=60)
    res = evaluate_recommenders(clean_synth, split, max_customers=60)
    assert res["n_eval_customers"] > 0
    for name in ("popularity", "collaborative", "content", "hybrid"):
        m = res["models"][name]
        assert 0 <= m["precision@5"] <= 1 and 0 <= m["ndcg@5"] <= 1 and 0 <= m["coverage"] <= 1
    # the training catalogue must be built from the past only
    past_items = clean_synth[(clean_synth["InvoiceDate"] < split) & (clean_synth["RecordType"] == "sale")]["StockCode"].nunique()
    assert res["n_train_items"] == past_items
