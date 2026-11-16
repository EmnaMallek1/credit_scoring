"""Build applicants.db (SQLite) from the Kaggle CSV.

Two tables, like two systems in a real bank:
  applicants      -> personal profile            (used by get_applicant)
  credit_history  -> credit behaviour            (used by get_credit_history)

The actual outcome (SeriousDlqin2yrs) is deliberately NOT stored: at decision
time nobody knows it, and the agent must never see it.

Run from the project root:   python build_applicant_db.py
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

CSV_PATH = Path("cs-training.csv")
DB_PATH = Path("applicants.db")

SCHEMA = """
CREATE TABLE applicants (
    applicant_id   INTEGER PRIMARY KEY,
    age            INTEGER NOT NULL,
    monthly_income REAL,
    dependents     INTEGER,
    split          TEXT NOT NULL            -- 'train' or 'test' (same split as the notebook)
);
CREATE TABLE credit_history (
    applicant_id          INTEGER PRIMARY KEY,
    revolving_utilization REAL    NOT NULL,
    debt_ratio            REAL    NOT NULL,
    open_credit_lines     INTEGER NOT NULL,
    real_estate_loans     INTEGER NOT NULL,
    past_due_30_59        INTEGER NOT NULL,
    past_due_60_89        INTEGER NOT NULL,
    times_90_days_late    INTEGER NOT NULL
);
"""

RENAME = {
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


def _val(v, cast):
    """NaN -> None (SQL NULL), otherwise cast to a plain Python type."""
    return None if pd.isna(v) else cast(v)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Add applicant_id and split, rename columns, drop applicants the API would refuse."""
    df = df.copy()
    ids = df["Unnamed: 0"] if "Unnamed: 0" in df.columns else pd.Series(df.index + 1, index=df.index)

    # Same split as the notebook (test_size=0.2, stratified, random_state=42)
    features = df.drop(columns=[c for c in ["Unnamed: 0", "SeriousDlqin2yrs"] if c in df.columns])
    target = df["SeriousDlqin2yrs"]
    _, x_test, _, _ = train_test_split(features, target, test_size=0.2, stratify=target, random_state=42)

    out = features.rename(columns=RENAME)
    out.insert(0, "applicant_id", ids.astype(int))
    out["split"] = np.where(out.index.isin(x_test.index), "test", "train")
    return out[out["age"] >= 18].reset_index(drop=True)      # the API requires age >= 18


def write_db(df: pd.DataFrame, db_path: Path) -> dict:
    db_path = Path(db_path)
    if db_path.exists():
        db_path.unlink()
    data = prepare(df)

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.executemany(
            "INSERT INTO applicants VALUES (?,?,?,?,?)",
            [(int(r.applicant_id), int(r.age), _val(r.monthly_income, float), _val(r.dependents, int), r.split)
             for r in data.itertuples()],
        )
        conn.executemany(
            "INSERT INTO credit_history VALUES (?,?,?,?,?,?,?,?)",
            [(int(r.applicant_id), float(r.revolving_utilization), float(r.debt_ratio),
              int(r.open_credit_lines), int(r.real_estate_loans), int(r.past_due_30_59),
              int(r.past_due_60_89), int(r.times_90_days_late))
             for r in data.itertuples()],
        )
        conn.commit()
    finally:
        conn.close()
    return {"applicants": len(data), "test_split": int((data["split"] == "test").sum())}


if __name__ == "__main__":
    if not CSV_PATH.exists():
        sys.exit(f"{CSV_PATH} not found - run this script from the project root.")
    stats = write_db(pd.read_csv(CSV_PATH), DB_PATH)
    print(f"Created {DB_PATH.resolve()}  ->  {stats['applicants']:,} applicants "
          f"({stats['test_split']:,} in the test split)")