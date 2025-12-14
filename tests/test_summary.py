import pandas as pd
from model_track.stats import get_summary

def test_get_summary_basic():
    df = pd.DataFrame({
        "age": [10, 20, 30, None],
        "city": ["SP", "RJ", "SP", None],
        "period": pd.to_datetime(["2020-01-01", "2020-06-15", None, "2020-12-31"])
    })

    summary = get_summary(df)

    # Deve ser um DataFrame
    assert isinstance(summary, pd.DataFrame)

    # Deve ter uma linha por coluna
    assert len(summary) == 3

    # Deve conter as colunas esperadas
    expected_cols = [
        "column_name", "dtype", "n_na", "pct_na", "top_class",
        "top_class_pct", "n_distinct", "distinct_values",
        "min", "max"
    ]
    assert all(col in summary.columns for col in expected_cols)

    # Testa uma linha específica
    age_summary = summary[summary["column_name"] == "age"].iloc[0]
    assert age_summary["n_na"] == 1
    assert age_summary["n_distinct"] == 4  # 10, 20, 30, None
    assert age_summary["min"] == 10
    assert age_summary["max"] == 30

    city_summary = summary[summary["column_name"] == "city"].iloc[0]
    assert city_summary["n_na"] == 1
    assert city_summary["n_distinct"] == 3  # SP, RJ

def test_get_summary_More_than_10_distincts():
    df = pd.DataFrame({
        "numbers": list(range(15)) + [None]
    })

    summary = get_summary(df)

    numbers_summary = summary[summary["column_name"] == "numbers"].iloc[0]
    assert numbers_summary["n_distinct"] == 16  # 0-14 + None
    assert numbers_summary["distinct_values"] == "..."
    

def test_get_summary_all_na():
    df = pd.DataFrame({
        "all_na": [None, None, None]
    })

    summary = get_summary(df)

    all_na_summary = summary[summary["column_name"] == "all_na"].iloc[0]
    assert all_na_summary["n_na"] == 3
    assert all_na_summary["pct_na"] == 100.0
    assert all_na_summary["n_distinct"] == 1
    assert pd.isna(all_na_summary["min"])
    assert pd.isna(all_na_summary["max"])
