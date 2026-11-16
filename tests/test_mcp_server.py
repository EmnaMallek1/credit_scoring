import asyncio
import sqlite3

import httpx
import numpy as np
import pandas as pd
import pytest
from mcp import Client
from mcp.server.mcpserver.exceptions import ToolError

import build_applicant_db
from mcp_server import server


def make_csv_frame(n=60):
    """A tiny frame shaped like cs-training.csv."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "Unnamed: 0": np.arange(1, n + 1),
        "SeriousDlqin2yrs": [1] * 12 + [0] * (n - 12),
        "RevolvingUtilizationOfUnsecuredLines": rng.random(n),
        "age": rng.integers(25, 80, n),
        "NumberOfTime30-59DaysPastDueNotWorse": rng.integers(0, 3, n),
        "DebtRatio": rng.random(n),
        "MonthlyIncome": rng.integers(2000, 9000, n).astype(float),
        "NumberOfOpenCreditLinesAndLoans": rng.integers(1, 12, n),
        "NumberOfTimes90DaysLate": rng.integers(0, 2, n),
        "NumberRealEstateLoansOrLines": rng.integers(0, 3, n),
        "NumberOfTime60-89DaysPastDueNotWorse": rng.integers(0, 2, n),
        "NumberOfDependents": rng.integers(0, 4, n).astype(float),
    })
    df.loc[0, "MonthlyIncome"] = np.nan        # applicant 1: unknown income
    df.loc[1, "NumberOfDependents"] = np.nan   # applicant 2: unknown dependents
    df.loc[2, "age"] = 0                       # applicant 3: invalid age, must be dropped
    return df


@pytest.fixture
def db(tmp_path, monkeypatch):
    folder = tmp_path / "folder with spaces"            # Windows paths often contain spaces
    folder.mkdir()
    path = folder / "applicants.db"
    build_applicant_db.write_db(make_csv_frame(), path)
    monkeypatch.setenv("APPLICANTS_DB", str(path))
    return path


@pytest.fixture
def fake_api(monkeypatch):
    calls = []

    def fake_post(path, payload, params=None):
        calls.append({"path": path, "payload": payload, "params": params})
        base = {"default_probability": 0.2, "risk_level": "high", "decision": "reject",
                "threshold": 0.165, "model_version": "1"}
        if path == "/explain":
            base["top_contributions"] = [
                {"feature": "RevolvingUtilizationOfUnsecuredLines", "value": 1.2,
                 "shap_value": 0.9, "effect": "increases_risk"},
                {"feature": "total_past_due", "value": 3.0, "shap_value": 0.4, "effect": "increases_risk"},
            ]
        return base

    monkeypatch.setattr(server, "_api_post", fake_post)
    return calls


# ------------------------------------------------------------------ database
def test_db_drops_invalid_age_and_never_stores_the_outcome(db):
    conn = sqlite3.connect(db)
    ids = {r[0] for r in conn.execute("SELECT applicant_id FROM applicants")}
    assert 3 not in ids and len(ids) == 59
    all_columns = {c[1] for t in ("applicants", "credit_history")
                   for c in conn.execute(f"PRAGMA table_info({t})")}
    assert "SeriousDlqin2yrs" not in all_columns


def test_db_keeps_unknown_values_as_null(db):
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT monthly_income FROM applicants WHERE applicant_id=1").fetchone()[0] is None
    assert conn.execute("SELECT dependents FROM applicants WHERE applicant_id=2").fetchone()[0] is None


def test_db_is_read_only_for_the_tools(db):
    with pytest.raises(sqlite3.OperationalError):
        conn = server._connect()
        try:
            conn.execute("DELETE FROM applicants")
        finally:
            conn.close()


# ------------------------------------------------------------------ tools
def test_get_applicant(db):
    result = server.get_applicant(5)
    assert set(result) == {"applicant_id", "age", "monthly_income", "dependents"}
    assert result["applicant_id"] == 5


def test_unknown_income_is_returned_as_none_not_zero(db):
    assert server.get_applicant(1)["monthly_income"] is None


def test_get_credit_history(db):
    result = server.get_credit_history(5)
    assert set(result) == {"applicant_id", *server.HISTORY_FIELDS}


def test_unknown_applicant_raises_a_clear_error(db):
    with pytest.raises(ToolError, match="not found"):
        server.get_applicant(999999)


def test_predict_sends_the_ten_model_fields(db, fake_api):
    result = server.predict_credit_risk(5)
    sent = fake_api[0]
    assert sent["path"] == "/predict"
    assert set(sent["payload"]) == {*server.PROFILE_FIELDS, *server.HISTORY_FIELDS}
    assert len(sent["payload"]) == 10
    assert result["decision"] == "reject" and result["applicant_id"] == 5


def test_explain_passes_top_k_and_uses_one_vocabulary(db, fake_api):
    result = server.explain_prediction(5, top_k=2)
    assert fake_api[0]["params"] == {"top_k": 2}
    names = [c["feature"] for c in result["top_contributions"]]
    assert names == ["revolving_utilization", "total_past_due"]


def test_api_down_gives_an_actionable_message(db, monkeypatch):
    def refuse(*args, **kwargs):
        raise httpx.ConnectError("refused")
    monkeypatch.setattr(httpx, "post", refuse)
    with pytest.raises(ToolError, match="Is the Docker container running"):
        server.predict_credit_risk(5)


# ------------------------------------------------------------------ through the MCP protocol
def test_four_tools_are_exposed_with_descriptions():
    async def run():
        async with Client(server.mcp) as client:
            return (await client.list_tools()).tools
    tools = asyncio.run(run())
    assert {t.name for t in tools} == {"get_applicant", "get_credit_history",
                                       "predict_credit_risk", "explain_prediction"}
    assert all(t.description for t in tools)


def test_errors_reach_the_llm_as_tool_errors_not_crashes(db):
    async def run():
        async with Client(server.mcp) as client:
            return await client.call_tool("get_applicant", {"applicant_id": 999999})
    result = asyncio.run(run())
    assert result.is_error
    assert "not found" in result.content[0].text