import numpy as np
import pytest

from model_track.woe import WoeCalculator
from model_track.woe import WoeByPeriod



def test_missing_date_column(sample_df_woe_by_period):
    with pytest.raises(KeyError):
        WoeByPeriod.compute(
            df=sample_df_woe_by_period,
            target_col="target",
            feature_col="feature",
            date_col="missing",
        )


def test_periods_present(sample_df_woe_by_period):
    result = WoeByPeriod.compute(
        df=sample_df_woe_by_period,
        target_col="target",
        feature_col="feature",
        date_col="date",
    )

    assert set(result["period"]) == {"2024-01", "2024-02"}


def test_rates_sum_to_one_per_period(sample_df_woe_by_period):
    result = WoeByPeriod.compute(
        df=sample_df_woe_by_period,
        target_col="target",
        feature_col="feature",
        date_col="date",
    )

    grouped = result.groupby("period")

    for _, g in grouped:
        assert np.isclose(g["event_rate"].sum(), 1.0)
        assert np.isclose(g["non_event_rate"].sum(), 1.0)
        assert np.isclose(g["exposure"].sum(), 1.0)


def test_woe_matches_base_calculator(sample_df_woe_by_period):
    result = WoeByPeriod.compute(
        df=sample_df_woe_by_period,
        target_col="target",
        feature_col="feature",
        date_col="date",
    )

    for period, g in sample_df_woe_by_period.groupby("date"):
        base = WoeCalculator.compute_table(
            df=g,
            target_col="target",
            feature_col="feature",
        )

        merged = result[result["period"] == period].merge(
            base,
            on="feature",
            suffixes=("_period", "_base"),
        )

        assert np.allclose(
            merged["woe_period"], merged["woe_base"]
        )


def test_all_null_dates(sample_df_woe_by_period):
    df_null_dates = sample_df_woe_by_period.copy()
    df_null_dates["date"] = np.nan

    with pytest.raises(ValueError):
        WoeByPeriod.compute(
            df=df_null_dates,
            target_col="target",
            feature_col="feature",
            date_col="date",
        )
