import pandas as pd
import pytest
from model_track.binning import BinApplier


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age": [10, 20, 30, 40, 50, None]
    })


def test_bin_applier_single_bin(sample_df):
    applier = BinApplier(sample_df)
    result = applier.apply("age", [30])

    assert result.dtype == object
    assert "N/A" in result.values
    assert sorted([v for v in result.unique() if v != "N/A"]) == ["<= 30", "> 30"]


def test_bin_applier_multiple_bins(sample_df):
    applier = BinApplier(sample_df)
    result = applier.apply("age", [20, 40])

    assert result.dtype == object
    assert "N/A" in result.values
    assert sorted([v for v in result.unique() if v != "N/A"]) == ["(20, 40]", "<= 20", "> 40"]


def test_bin_applier_column_missing(sample_df):
    applier = BinApplier(sample_df)
    with pytest.raises(ValueError):
        applier.apply("unknown", [10])


def test_bin_applier_invalid_bins():
    df = pd.DataFrame({"x": [1, 2, 3]})
    applier = BinApplier(df)

    with pytest.raises(ValueError):
        applier.apply("x", [5, 3])

    with pytest.raises(ValueError):
        applier.apply("x", [2, 2])

    with pytest.raises(ValueError):
        applier.apply("x", [])


def test_bin_applier_non_dataframe_input():
    with pytest.raises(ValueError):
        BinApplier("not a dataframe")


def test_bin_applier_non_list_bins(sample_df):
    applier = BinApplier(sample_df)
    with pytest.raises(ValueError):
        applier.apply("age", "not a list")
        

def test_bin_applier_with_none_and_nan():
    df = pd.DataFrame({
        "value": [1, 2, None, 4, float('nan'), 6]
    })
    applier = BinApplier(df)
    result = applier.apply("value", [2, 4])

    assert result.dtype == object
    assert "N/A" in result.values
    assert sorted([v for v in result.unique() if v != "N/A"]) == ["(2, 4]", "<= 2", "> 4"]

