import numpy as np
import pandas as pd
import pytest
from model_track.woe import WoeCalculator


# ------------------------------------------------------------------
# Validation tests
# ------------------------------------------------------------------

def test_missing_target_column(sample_df_woe_calculator):
    with pytest.raises(KeyError):
        WoeCalculator.compute_table(
            df=sample_df_woe_calculator,
            target_col="missing",
            feature_col="feature",
        )


def test_missing_feature_column(sample_df_woe_calculator):
    with pytest.raises(KeyError):
        WoeCalculator.compute_table(
            df=sample_df_woe_calculator,
            target_col="target",
            feature_col="missing",
        )


def test_non_binary_target_raises():
    df = pd.DataFrame(
        {
            "feature": ["A", "B", "C"],
            "target": [0, 1, 2],
        }
    )

    with pytest.raises(ValueError):
        WoeCalculator.compute_table(
            df=df,
            target_col="target",
            feature_col="feature",
        )


def test_invalid_event_value_raises(sample_df_woe_calculator):
    with pytest.raises(ValueError):
        WoeCalculator.compute_table(
            df=sample_df_woe_calculator,
            target_col="target",
            feature_col="feature",
            event_value=2,
        )


# ------------------------------------------------------------------
# Correctness tests (excluding totals row)
# ------------------------------------------------------------------

def _without_totals(table: pd.DataFrame) -> pd.DataFrame:
    return table[table["feature"] != "__TOTAL__"]


def test_rates_sum_to_one(sample_df_woe_calculator):
    table = WoeCalculator.compute_table(
        df=sample_df_woe_calculator,
        target_col="target",
        feature_col="feature",
    )

    table = _without_totals(table)

    assert np.isclose(table["event_rate"].sum(), 1.0)
    assert np.isclose(table["non_event_rate"].sum(), 1.0)


def test_exposure_sums_to_one(sample_df_woe_calculator):
    table = WoeCalculator.compute_table(
        df=sample_df_woe_calculator,
        target_col="target",
        feature_col="feature",
    )

    table = _without_totals(table)

    assert np.isclose(table["exposure"].sum(), 1.0)


def test_woe_no_nan_or_inf(sample_df_woe_calculator):
    table = WoeCalculator.compute_table(
        df=sample_df_woe_calculator,
        target_col="target",
        feature_col="feature",
    )

    table = _without_totals(table)

    assert np.isinf(table["woe"]).any()
    assert np.isinf(table["iv"]).any()
    assert np.isinf(table["iv_total"]).any()


def test_expected_categories_with_totals(sample_df_woe_calculator):
    table = WoeCalculator.compute_table(
        df=sample_df_woe_calculator,
        target_col="target",
        feature_col="feature",
    )

    assert set(table["feature"]) == {"A", "B", "C", "__TOTAL__"}

def test_woe_sem_infs(sample_df_woe_calculator):
    table = WoeCalculator.compute_table(
        df=sample_df_woe_calculator,
        target_col="target_balanced",
        feature_col="feature_balanced",
    )

    assert not np.isinf(table["woe"]).any()
    assert not np.isinf(table["iv"]).any()
    assert not np.isinf(table["iv_total"]).any()
    
# ------------------------------------------------------------------
# IV tests
# ------------------------------------------------------------------

def test_iv_column_exists(sample_df_woe_calculator):
    table = WoeCalculator.compute_table(
        df=sample_df_woe_calculator,
        target_col="target",
        feature_col="feature",
    )

    assert "iv" in table.columns
    assert "iv_total" in table.columns


def test_iv_total_equals_sum_of_iv(sample_df_woe_calculator):
    table = WoeCalculator.compute_table(
        df=sample_df_woe_calculator,
        target_col="target",
        feature_col="feature",
    )

    data = _without_totals(table)

    iv_sum = data["iv"].sum()
    iv_total = table.loc[
        table["feature"] == "__TOTAL__", "iv"
    ].iloc[0]

    assert np.isclose(iv_sum, iv_total)


def test_iv_total_repeated_per_row(sample_df_woe_calculator):
    table = WoeCalculator.compute_table(
        df=sample_df_woe_calculator,
        target_col="target",
        feature_col="feature",
    )

    iv_total_values = table["iv_total"].unique()
    assert len(iv_total_values) == 1


# ------------------------------------------------------------------
# Mapping tests
# ------------------------------------------------------------------

def test_compute_mapping_matches_table(sample_df_woe_calculator):
    table = WoeCalculator.compute_table(
        df=sample_df_woe_calculator,
        target_col="target",
        feature_col="feature",
        add_totals=False,
    )

    mapping = WoeCalculator.compute_mapping(
        df=sample_df_woe_calculator,
        target_col="target",
        feature_col="feature",
    )

    for _, row in table.iterrows():
        assert np.isclose(mapping[row["feature"]], row["woe"])
