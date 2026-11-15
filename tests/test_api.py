import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(pipeline_root):
    mp = pytest.MonkeyPatch()
    mp.setenv("RETAILIQ_ROOT", str(pipeline_root))
    from api import deps
    from api.main import app
    deps.reset_store()
    with TestClient(app) as c:
        yield c
    deps.reset_store()
    mp.undo()


@pytest.fixture(scope="module")
def cid(client):
    from api import deps
    return int(deps.get_store().scores.index[0])


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["artifacts_loaded"]
    assert "X-Request-ID" in r.headers


def test_customer_profile_has_no_private_data(client, cid):
    r = client.get(f"/customers/{cid}")
    assert r.status_code == 200
    body = r.json()
    assert body["customer_id"] == cid and 0 <= body["p_high_value"] <= 1
    assert set(body) == {"customer_id", "segment", "country", "rfm", "avg_order_value", "active_months", "tenure_days", "p_high_value"}


def test_customer_segment(client, cid):
    r = client.get(f"/customers/{cid}/segment")
    assert r.status_code == 200 and r.json()["recommended_action"]


def test_unknown_customer_is_404_and_bad_id_is_422(client):
    assert client.get("/customers/999999").status_code == 404
    assert client.get("/customers/0").status_code == 422
    assert client.get("/customers/abc").status_code == 422


def test_predict_value(client, cid):
    r = client.post(f"/customers/{cid}/predict-value")
    assert r.status_code == 200 and 0 <= r.json()["probability_high_value"] <= 1 and r.json()["baseline_probability"] is None
    r2 = client.post(f"/customers/{cid}/predict-value", json={"overrides": {"recency_days": 1}})
    assert r2.status_code == 200 and r2.json()["baseline_probability"] is not None
    assert client.post(f"/customers/{cid}/predict-value", json={"overrides": {"not_a_feature": 1}}).status_code == 422


def test_recommendations(client, cid):
    r = client.post("/recommendations", json={"customer_id": cid, "top_n": 4})
    body = r.json()
    assert r.status_code == 200 and len(body["items"]) == 4 and not body["fallback"]
    cold = client.post("/recommendations", json={"customer_id": 999998, "top_n": 3}).json()
    assert cold["fallback"] and cold["strategy_used"] == "popularity"
    assert client.post("/recommendations", json={"customer_id": cid, "top_n": 0}).status_code == 422
    assert client.post("/recommendations", json={"customer_id": cid, "strategy": "magic"}).status_code == 422


def test_segment_analytics(client):
    r = client.get("/analytics/segments")
    body = r.json()
    assert r.status_code == 200 and body["n_segments"] == len(body["segments"]) >= 3
    assert sum(s["customers"] for s in body["segments"]) > 0
    assert abs(sum(s["revenue_share"] for s in body["segments"]) - 1) < 0.01


def test_api_key_is_enforced_when_configured(client, monkeypatch, cid):
    monkeypatch.setenv("RETAILIQ_API_KEY", "s3cret")
    assert client.get(f"/customers/{cid}").status_code == 401
    assert client.get(f"/customers/{cid}", headers={"X-API-Key": "s3cret"}).status_code == 200
    assert client.get("/health").status_code == 200                      # health stays public


def test_openapi_docs_available(client):
    assert client.get("/openapi.json").status_code == 200 and client.get("/docs").status_code == 200
