import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

REVOLVING = "RevolvingUtilizationOfUnsecuredLines"
DEBT_RATIO = "DebtRatio"
INCOME = "MonthlyIncome"
DEPENDENTS = "NumberOfDependents"
PAST_DUE = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]
ERROR_CODES = [96, 98]


class CreditCleaner(BaseEstimator, TransformerMixin):
    """Cleaning + feature engineering. Caps and medians are learned in fit() (train only)."""

    def __init__(self, cap_quantile=0.99, add_features=True):
        self.cap_quantile = cap_quantile
        self.add_features = add_features

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.cap_revolving_ = X[REVOLVING].quantile(self.cap_quantile)
        self.cap_debt_ratio_ = X[DEBT_RATIO].quantile(self.cap_quantile)
        past_due = X[PAST_DUE].where(~X[PAST_DUE].isin(ERROR_CODES))
        self.past_due_medians_ = past_due.median()
        self.income_median_ = X[INCOME].median()
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        # 96/98 error codes: one flag, then NaN, then train median
        X["had_error_code"] = X[PAST_DUE].isin(ERROR_CODES).any(axis=1).astype(int)
        for c in PAST_DUE:
            X[c] = X[c].where(~X[c].isin(ERROR_CODES)).fillna(self.past_due_medians_[c])

        # outliers
        X[REVOLVING] = X[REVOLVING].clip(upper=self.cap_revolving_)
        X[DEBT_RATIO] = X[DEBT_RATIO].clip(upper=self.cap_debt_ratio_)

        # missing values: keep the "was missing" information
        X["income_missing"] = X[INCOME].isna().astype(int)
        X["dependents_missing"] = X[DEPENDENTS].isna().astype(int)
        X[DEPENDENTS] = X[DEPENDENTS].fillna(0)
        income = X[INCOME].fillna(self.income_median_)

        if self.add_features:
            X["total_past_due"] = X[PAST_DUE].sum(axis=1)
            X["log_income_per_person"] = np.log1p(income / (X[DEPENDENTS] + 1))

        X[INCOME] = np.log1p(income)
        return X