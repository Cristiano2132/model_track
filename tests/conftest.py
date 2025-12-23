import pytest
import pandas as pd


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age": [10, 20, 30, 40, 50, None, 60, 70],
        "income": [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000],
    })


@pytest.fixture
def simple_bins():
    return {"age": [0, 18, 30, 50, float("inf")]}


@pytest.fixture
def valid_bins():
    return [0, 18, 40, 60, float("inf")]



@pytest.fixture
def sample_df_woe_calculator():
    return pd.DataFrame(
        {
            # Feature original (continua tendo WOE infinito em C)
            "feature": [
                "A", "A",
                "B", "B", "B",
                "C", "C",
            ],
            "target": [
                1, 0,      # A → ok
                1, 0, 0,   # B → ok
                0, 0,      # C → só non-event → WOE = -inf
            ],

            # Nova feature SEM WOE infinito
            "feature_balanced": [
                "X", "X",
                "Y", "Y", "Y",
                "Z", "Z",
            ],
            "target_balanced": [
                1, 0,      # X → ok
                1, 0, 0,   # Y → ok
                1, 0,      # Z → ok
            ],
        }
    )


@pytest.fixture
def sample_df_woe_by_period():
    return pd.DataFrame(
        {
            "date": ["2024-01", "2024-01", "2024-02", "2024-02", "2024-02"],
            "feature": ["A", "B", "A", "B", "B"],
            "target": [1, 0, 0, 1, 0],
        }
    )



@pytest.fixture
def sample_df_woe_stability():
    return pd.DataFrame(
        {
            "period": pd.to_datetime(
                ["2024-01", "2024-01", "2024-02", "2024-02", "2024-03", "2024-03"]
            ),
            "feature_cat": ["A", "B", "A", "B", "A", "B"],
            "target": [1, 0, 0, 1, 1, 0],
        }
    )


@pytest.fixture
def global_woe_numeric_category_mapper():
    return pd.DataFrame(
        {
            "income": ["10", "20", "30", "40", "50", "60", "__TOTAL__"],
            "woe": [-0.8, -0.6, -0.1, 0.1, 0.5, 0.8, None],
        }
    )


@pytest.fixture
def global_woe_intervals_category_mapper():
    return pd.DataFrame(
        {
            "age": ["<=1", "(1,3]", "(3,5]", "(5,7]", "__TOTAL__"],
            "woe": [-0.9, -0.4, 0.1, 0.6, None],
        }
    )

