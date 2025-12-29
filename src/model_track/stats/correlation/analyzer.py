#  model_track/stats/correlation/analyzer.py

import pandas as pd
from typing import Dict, List
from .strategies import (
    CategoricalCategoricalStrategy,
    ContinuousContinuousStrategy,
    ContinuousCategoricalStrategy,
)
from .base import CorrelationStrategy


class CorrelationAnalyzer:

    def __init__(self, dropna: bool = True):
        self.dropna = dropna
        self.strategies = [
            CategoricalCategoricalStrategy(),
            ContinuousContinuousStrategy(),
            ContinuousCategoricalStrategy(),
        ]

    def _select_strategy(
        self,
        x: pd.Series,
        y: pd.Series,
    ) -> CorrelationStrategy:

        x_is_numeric = pd.api.types.is_numeric_dtype(x)
        y_is_numeric = pd.api.types.is_numeric_dtype(y)

        # 1️⃣ numérico x numérico → Pearson / Spearman
        if x_is_numeric and y_is_numeric:
            return ContinuousContinuousStrategy()

        # 2️⃣ categórico x categórico → Cramér's V
        if not x_is_numeric and not y_is_numeric:
            return CategoricalCategoricalStrategy()

        # 3️⃣ misto → correlation ratio / ANOVA
        return ContinuousCategoricalStrategy()

    def analyze(self, x: pd.Series, y: pd.Series) -> Dict:
        if self.dropna:
            df = pd.concat([x, y], axis=1).dropna()
            x, y = df.iloc[:, 0], df.iloc[:, 1]

        strategy = self._select_strategy(x, y)
        result = strategy.compute(x, y)
        result["x_type"] = "categorical" if x.dtype == "object" else "continuous"
        result["y_type"] = "categorical" if y.dtype == "object" else "continuous"
        return result