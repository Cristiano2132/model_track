import pandas as pd
import numpy as np
from model_track.binning import QuantileBinner


def test_quantile_binner_basic():
    df = pd.DataFrame({"x": [1,2,3,4,5,6,7,8,9]})
    qb = QuantileBinner(n_bins=3)
    qb.fit(df, "x")
    bins = [float(round(b, 1)) for b in qb.bins_]
    # Expected: 33% and 66% quantiles
    assert bins== [3.7, 6.3]


def test_quantile_binner_low_unique():
    df = pd.DataFrame({"x": [10, 10, 10, 10]})
    qb = QuantileBinner(n_bins=3)
    qb.fit(df, "x")
    assert qb.bins_ == []


def test_quantile_binner_ignores_target():
    df = pd.DataFrame({
        "x": np.arange(1, 101),
        "y": np.random.randint(0, 2, 100)
    })

    qb = QuantileBinner(n_bins=4)
    qb.fit(df, "x", target="y")

    # Quartis → 25%, 50%, 75%
    assert qb.bins_ == [25.75, 50.5, 75.25]


def test_quantile_binner_few_distinct():
    # If too few distinct cut points → fallback sem bins
    df = pd.DataFrame({"x": [1, 2, 2, 2, 3, 3, 4, 4, 5]})
    qb = QuantileBinner(n_bins=0)
    qb.fit(df, "x")
    assert qb.bins_ == []