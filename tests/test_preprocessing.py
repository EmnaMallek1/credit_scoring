import numpy as np
import pandas as pd
import pytest

from preprocessing import CreditCleaner

PAST_DUE = ["NumberOfTime30-59DaysPastDueNotWorse",
            "NumberOfTime60-89DaysPastDueNotWorse",
            "NumberOfTimes90DaysLate"]


def make_df():
    return pd.DataFrame({
        "RevolvingUtilizationOfUnsecuredLines": [0.2, 0.5, 0.9, 5000.0],
        "age": [30, 45, 60, 25],
        "NumberOfTime30-59DaysPastDueNotWorse": [0, 1, 2, 98],
        "DebtRatio": [0.3, 0.4, 0.5, 9000.0],
        "MonthlyIncome": [3000.0, np.nan, 5000.0, 4000.0],
        "NumberOfOpenCreditLinesAndLoans": [5, 6, 7, 8],
        "NumberOfTimes90DaysLate": [0, 0, 1, 98],
        "NumberRealEstateLoansOrLines": [0, 1, 1, 2],
        "NumberOfTime60-89DaysPastDueNotWorse": [0, 0, 1, 98],
        "NumberOfDependents": [0.0, 1.0, np.nan, 2.0],
    })


def test_output_has_no_missing_values():
    out = CreditCleaner().fit(make_df()).transform(make_df())
    assert out.isna().sum().sum() == 0


def test_error_codes_are_flagged_and_replaced():
    df = make_df()
    out = CreditCleaner().fit(df).transform(df)
    assert out["had_error_code"].tolist() == [0, 0, 0, 1]
    assert (out[PAST_DUE] < 96).all().all()


def test_missing_indicators():
    df = make_df()
    out = CreditCleaner().fit(df).transform(df)
    assert out["income_missing"].tolist() == [0, 1, 0, 0]
    assert out["dependents_missing"].tolist() == [0, 0, 1, 0]


def test_outliers_are_capped_with_train_quantile():
    df = make_df()
    cleaner = CreditCleaner().fit(df)
    out = cleaner.transform(df)
    assert out["RevolvingUtilizationOfUnsecuredLines"].max() <= cleaner.cap_revolving_
    assert out["DebtRatio"].max() <= cleaner.cap_debt_ratio_


def test_transform_uses_train_statistics_not_new_data():
    train = make_df()
    cleaner = CreditCleaner().fit(train)
    new = train.iloc[[1]].copy()          # income missing
    out = cleaner.transform(new)
    assert out["MonthlyIncome"].iloc[0] == pytest.approx(np.log1p(cleaner.income_median_))


def test_missing_column_raises():
    df = make_df().drop(columns=["age", "DebtRatio"])
    with pytest.raises(KeyError):
        CreditCleaner().fit(df)