# model_track/stats/correlation/strategies.py

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict
from .base import CorrelationStrategy

# Utilitário

def is_categorical(s: pd.Series) -> bool:
    return (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_categorical_dtype(s)
    )

# 🔹 Cat × Cat — Cramér’s V

class CategoricalCategoricalStrategy(CorrelationStrategy):

    def supports(self, x: pd.Series, y: pd.Series) -> bool:
        return is_categorical(x) and is_categorical(y)

    def compute(self, x: pd.Series, y: pd.Series) -> Dict:
        table = pd.crosstab(x, y)
        chi2 = stats.chi2_contingency(table)[0]
        n = table.sum().sum()
        r, k = table.shape

        value = np.sqrt(chi2 / (n * (min(r, k) - 1)))

        return {
            "method": "cramers_v",
            "correlation": value,
            "p_value": None,
        }


# 🔹 Cont × Cont — Spearman

class ContinuousContinuousStrategy(CorrelationStrategy):

    def supports(self, x: pd.Series, y: pd.Series) -> bool:
        return not is_categorical(x) and not is_categorical(y)

    def compute(self, x: pd.Series, y: pd.Series) -> Dict:
        corr, p = stats.spearmanr(x, y)

        return {
            "method": "spearman",
            "correlation": corr,
            "p_value": p,
        }


# 🔹 Cont × Cat — Correlation Ratio (η²)

class ContinuousCategoricalStrategy(CorrelationStrategy):

    def supports(self, x: pd.Series, y: pd.Series) -> bool:
        return is_categorical(x) != is_categorical(y)

    def compute(self, x: pd.Series, y: pd.Series) -> Dict:
        if is_categorical(x):
            categories, values = x, y
        else:
            categories, values = y, x

        categories = categories.astype("category")
        means = values.groupby(categories).mean()
        counts = values.groupby(categories).count()

        overall_mean = values.mean()
        numerator = np.sum(counts * (means - overall_mean) ** 2)
        denominator = np.sum((values - overall_mean) ** 2)

        eta = np.sqrt(numerator / denominator) if denominator != 0 else 0.0

        return {
            "method": "eta_squared",
            "correlation": eta,
            "p_value": None,
        }

