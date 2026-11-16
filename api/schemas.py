#schemas.py : checks the data is valid (age ≥ 18, no negatives, etc.) and converts it into the table format THE model was trained on
#It also turns "unknown income" into a proper missing value, which CreditCleaner already knows how to handle.

from typing import Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field

# API field name -> column name used when the model was trained
COLUMN_MAP = {
    "revolving_utilization": "RevolvingUtilizationOfUnsecuredLines",
    "age": "age",
    "past_due_30_59": "NumberOfTime30-59DaysPastDueNotWorse",
    "debt_ratio": "DebtRatio",
    "monthly_income": "MonthlyIncome",
    "open_credit_lines": "NumberOfOpenCreditLinesAndLoans",
    "times_90_days_late": "NumberOfTimes90DaysLate",
    "real_estate_loans": "NumberRealEstateLoansOrLines",
    "past_due_60_89": "NumberOfTime60-89DaysPastDueNotWorse",
    "dependents": "NumberOfDependents",
}
MODEL_COLUMNS = list(COLUMN_MAP.values())


class Applicant(BaseModel):
    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "example": {
                "revolving_utilization": 0.45,
                "age": 41,
                "past_due_30_59": 0,
                "debt_ratio": 0.35,
                "monthly_income": 5400,
                "open_credit_lines": 8,
                "times_90_days_late": 0,
                "real_estate_loans": 1,
                "past_due_60_89": 0,
                "dependents": 2,
            }
        },
    }

    revolving_utilization: float = Field(..., ge=0, description="Balance / credit limit on unsecured lines")
    age: int = Field(..., ge=18, le=120)
    past_due_30_59: int = Field(..., ge=0, le=98, description="Times 30-59 days past due (last 2 years)")
    debt_ratio: float = Field(..., ge=0, description="Monthly debt payments / monthly income")
    monthly_income: Optional[float] = Field(None, ge=0, description="Leave null if unknown")
    open_credit_lines: int = Field(..., ge=0, le=100)
    times_90_days_late: int = Field(..., ge=0, le=98)
    real_estate_loans: int = Field(..., ge=0, le=100)
    past_due_60_89: int = Field(..., ge=0, le=98)
    dependents: Optional[int] = Field(None, ge=0, le=20, description="Leave null if unknown")

    def to_frame(self) -> pd.DataFrame:
        row = {COLUMN_MAP[k]: v for k, v in self.model_dump().items()}
        # float64 turns None into NaN, which the pipeline knows how to handle
        return pd.DataFrame([row], columns=MODEL_COLUMNS).astype("float64")


class PredictionResponse(BaseModel):
    default_probability: float
    risk_level: Literal["low", "medium", "high"]
    decision: Literal["approve", "reject"]
    threshold: float
    model_version: str


class Contribution(BaseModel):
    feature: str
    value: Optional[float]
    shap_value: float
    effect: Literal["increases_risk", "decreases_risk"]


class ExplanationResponse(PredictionResponse):
    top_contributions: list[Contribution]
    note: str = (
        "shap_value is the contribution to the model's raw risk score (log-odds, "
        "before calibration). Positive values push the applicant toward default."
    )