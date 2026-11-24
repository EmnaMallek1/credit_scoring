"""MCP server for the credit-scoring project.

Exposes four READ-ONLY tools. The LLM never sees raw model files or the database:
it can only call these tools, and every tool takes an applicant_id (never raw
numbers), so the LLM cannot mistype a figure.

  get_applicant          -> profile            (SQLite)
  get_credit_history     -> credit behaviour   (SQLite)
  predict_credit_risk    -> POST /predict      (FastAPI, e.g. the Docker container)
  explain_prediction     -> POST /explain      (FastAPI)

IMPORTANT: with the stdio transport, stdout is the protocol channel.
Never use print() here - log to stderr only.
"""
import logging
import os
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from typing import Any

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

#logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
level=os.getenv("MCP_LOG_LEVEL", "INFO"),
log = logging.getLogger("credit-mcp")

mcp = MCPServer("credit-scoring")

PROFILE_FIELDS = ["age", "monthly_income", "dependents"]
HISTORY_FIELDS = ["revolving_utilization", "debt_ratio", "open_credit_lines", "real_estate_loans",
                  "past_due_30_59", "past_due_60_89", "times_90_days_late"]

# The model's raw column names -> the names the other tools use, so the LLM sees one vocabulary.
# (Engineered features such as total_past_due keep their own name.)
FEATURE_NAMES = {
    "RevolvingUtilizationOfUnsecuredLines": "revolving_utilization",
    "age": "age",
    "NumberOfTime30-59DaysPastDueNotWorse": "past_due_30_59",
    "DebtRatio": "debt_ratio",
    "MonthlyIncome": "monthly_income",
    "NumberOfOpenCreditLinesAndLoans": "open_credit_lines",
    "NumberOfTimes90DaysLate": "times_90_days_late",
    "NumberRealEstateLoansOrLines": "real_estate_loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "past_due_60_89",
    "NumberOfDependents": "dependents",
}


# --------------------------------------------------------------------------- helpers
# Errors we anticipate (unknown applicant, API down...) are raised as ToolError: the model
# receives the message and can react to it. Any other exception is treated as a crash and
# the model only sees "Error executing tool <name>".
def _db_path() -> Path:
    return Path(os.getenv("APPLICANTS_DB", "applicants.db"))


def _api_url() -> str:
    return os.getenv("CREDIT_API_URL", "http://127.0.0.1:8000").rstrip("/")


def _connect() -> sqlite3.Connection:
    """Open the database READ-ONLY: no tool can modify applicant data."""
    path = _db_path().resolve()
    if not path.exists():
        raise ToolError(f"Applicant database not found at {path}. Run build_applicant_db.py first.")
    conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _fetch(table: str, applicant_id: int, columns: list[str]) -> dict[str, Any]:
    # table/columns are internal constants; applicant_id is passed as a bound parameter
    with closing(_connect()) as conn:
        row = conn.execute(
            f"SELECT {', '.join(columns)} FROM {table} WHERE applicant_id = ?", (applicant_id,)
        ).fetchone()
    if row is None:
        raise ToolError(f"Applicant {applicant_id} was not found in the database.")
    return dict(row)


def _api_post(path: str, payload: dict, params: dict | None = None) -> dict:
    url = f"{_api_url()}{path}"
    try:
        response = httpx.post(url, json=payload, params=params, timeout=15)
    except httpx.HTTPError as exc:
        raise ToolError(
            f"The credit scoring API is unreachable at {_api_url()} ({type(exc).__name__}). "
            "Is the Docker container running?"
        ) from exc
    if response.status_code == 422:
        raise ToolError(f"The API rejected this applicant's data: {response.json().get('detail')}")
    if response.status_code >= 400:
        raise ToolError(f"The API returned an error {response.status_code}: {response.text[:200]}")
    return response.json()


def _scoring_payload(applicant_id: int) -> dict[str, Any]:
    """Profile + history = the 10 fields the model needs."""
    return {**_fetch("applicants", applicant_id, PROFILE_FIELDS),
            **_fetch("credit_history", applicant_id, HISTORY_FIELDS)}

def _readable(result: dict) -> dict:
    """Add percentages computed in code, so the LLM never converts numbers itself."""
    return {**result,
            "default_probability_pct": f"{result['default_probability'] * 100:.1f}%",
            "threshold_pct": f"{result['threshold'] * 100:.1f}%"}
# --------------------------------------------------------------------------- tools
@mcp.tool()
def get_applicant(applicant_id: int) -> dict:
    """Get the personal profile of a loan applicant: age, monthly income and number of dependents.

    Use this first to describe who the applicant is. A null monthly_income or dependents
    means the information is unknown (it is not zero).
    """
    log.info("get_applicant(%s)", applicant_id)
    return {"applicant_id": applicant_id, **_fetch("applicants", applicant_id, PROFILE_FIELDS)}


@mcp.tool()
def get_credit_history(applicant_id: int) -> dict:
    """Get the credit behaviour of a loan applicant.

    Returns: revolving_utilization (balance / credit limit, 1.0 = fully used), debt_ratio
    (monthly debt payments / income), open_credit_lines, real_estate_loans, and the number
    of times the applicant was 30-59 days, 60-89 days and 90+ days late on payments.
    """
    log.info("get_credit_history(%s)", applicant_id)
    return {"applicant_id": applicant_id, **_fetch("credit_history", applicant_id, HISTORY_FIELDS)}


@mcp.tool()
def predict_credit_risk(applicant_id: int) -> dict:
    """Run the credit scoring model on an applicant.

    Returns default_probability (calibrated chance of serious delinquency within 2 years),
    risk_level (low / medium / high), decision (approve / reject), the decision threshold
    and the model version. The decision comes from the model and its threshold: report it
    as is, do not override it.
    """
    log.info("predict_credit_risk(%s)", applicant_id)
    result = _api_post("/predict", _scoring_payload(applicant_id))
    return {"applicant_id": applicant_id, **_readable(result)}

@mcp.tool()
def explain_prediction(applicant_id: int, top_k: int = 5) -> dict:
    """Explain WHY the model scored an applicant the way it did (SHAP).

    Returns the prediction plus the top_k features (1-20, default 5) that moved the risk most.
    Each contribution has the feature name, the applicant's value, a shap_value and an effect:
    'increases_risk' or 'decreases_risk'. shap_value is in log-odds, NOT percentage points:
    describe which factors weigh most, never say a factor 'adds X%' to the probability.
    """
    log.info("explain_prediction(%s, top_k=%s)", applicant_id, top_k)
    result = _api_post("/explain", _scoring_payload(applicant_id), params={"top_k": top_k})
    for item in result.get("top_contributions", []):
        item["feature"] = FEATURE_NAMES.get(item["feature"], item["feature"])
    return {"applicant_id": applicant_id, **_readable(result)}


if __name__ == "__main__":
    mcp.run()          # stdio transport by default