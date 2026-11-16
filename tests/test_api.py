import pytest
from fastapi.testclient import TestClient

from api.main import app

GOOD = {
    "revolving_utilization": 0.45, "age": 41, "past_due_30_59": 0, "debt_ratio": 0.35,
    "monthly_income": 5400, "open_credit_lines": 8, "times_90_days_late": 0,
    "real_estate_loans": 1, "past_due_60_89": 0, "dependents": 2,
}
RISKY = {**GOOD, "revolving_utilization": 1.3, "past_due_30_59": 4,
         "times_90_days_late": 3, "past_due_60_89": 2, "age": 24}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:      # runs startup -> loads the champion model
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_format(client):
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["default_probability"] <= 1
    assert body["risk_level"] in {"low", "medium", "high"}
    assert body["decision"] in {"approve", "reject"}


def test_risky_applicant_scores_higher(client):
    p_good = client.post("/predict", json=GOOD).json()["default_probability"]
    p_risky = client.post("/predict", json=RISKY).json()["default_probability"]
    assert p_risky > p_good


def test_missing_income_and_dependents_are_accepted(client):
    r = client.post("/predict", json={**GOOD, "monthly_income": None, "dependents": None})
    assert r.status_code == 200


def test_minor_is_rejected(client):
    assert client.post("/predict", json={**GOOD, "age": 17}).status_code == 422


def test_negative_value_is_rejected(client):
    assert client.post("/predict", json={**GOOD, "debt_ratio": -1}).status_code == 422


def test_unknown_field_is_rejected(client):
    assert client.post("/predict", json={**GOOD, "salary": 1}).status_code == 422


def test_explain(client):
    r = client.post("/explain?top_k=3", json=RISKY)
    assert r.status_code == 200
    contrib = r.json()["top_contributions"]
    assert len(contrib) == 3
    assert abs(contrib[0]["shap_value"]) >= abs(contrib[-1]["shap_value"])