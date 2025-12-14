import pandas as pd
import matplotlib.pyplot as plt
import pytest

from model_track.stability import WoeStability


def test_missing_date_column_raises(sample_df_woe_stability):
    with pytest.raises(KeyError):
        WoeStability(
            df=sample_df_woe_stability,
            date_col="missing",
        )


def test_global_table_returns_dataframe(sample_df_woe_stability):
    ws = WoeStability(
        df=sample_df_woe_stability,
        date_col="period",
    )

    table = ws.global_table(
        feature_col="feature_cat",
        target_col="target",
    )

    assert isinstance(table, pd.DataFrame)
    assert "woe" in table.columns
    assert "feature_cat" in table.columns


def test_generate_view_creates_figure(sample_df_woe_stability):
    ws = WoeStability(
        df=sample_df_woe_stability,
        date_col="period",
    )

    fig = ws.generate_view(
        feature_col="feature_cat",
        target_col="target",
        ax=None,
    )

    assert fig is not None
    assert hasattr(fig, "axes")


def test_generate_view_with_ax_returns_none(sample_df_woe_stability):
    ws = WoeStability(
        df=sample_df_woe_stability,
        date_col="period",
    )

    fig, ax = plt.subplots()

    result = ws.generate_view(
        feature_col="feature_cat",
        target_col="target",
        ax=ax,
    )

    assert result is None


def test_generate_view_adds_lines_to_axis(sample_df_woe_stability):
    ws = WoeStability(
        df=sample_df_woe_stability,
        date_col="period",
    )

    fig, ax = plt.subplots()

    ws.generate_view(
        feature_col="feature_cat",
        target_col="target",
        ax=ax,
    )

    assert len(ax.lines) > 0