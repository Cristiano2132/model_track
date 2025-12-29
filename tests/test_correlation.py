
import pandas as pd
from model_track.stats.correlation.analyzer import (
    CorrelationAnalyzer,
    ContinuousContinuousStrategy, 
    CategoricalCategoricalStrategy, 
    ContinuousCategoricalStrategy
)

def test_supports_strategies():
    cat_series = pd.Series(["A", "B", "A", "C"])
    cont_series = pd.Series([1.0, 2.5, 3.3, 4.1])

    cat_cat_strategy = CategoricalCategoricalStrategy()
    cont_cont_strategy = ContinuousContinuousStrategy()
    cont_cat_strategy = ContinuousCategoricalStrategy()

    assert cat_cat_strategy.supports(cat_series, cat_series) is True
    assert cat_cat_strategy.supports(cat_series, cont_series) is False
    assert cat_cat_strategy.supports(cont_series,cat_series) is False
    assert cat_cat_strategy.supports(cont_series, cont_series) is False

    assert cont_cont_strategy.supports(cont_series, cont_series) is True
    assert cont_cont_strategy.supports(cat_series, cont_series) is False
    assert cont_cont_strategy.supports(cont_series, cat_series) is False
    assert cont_cont_strategy.supports(cat_series, cat_series) is False

    assert cont_cat_strategy.supports(cat_series, cont_series) is True
    assert cont_cat_strategy.supports(cont_series, cat_series) is True
    assert cont_cat_strategy.supports(cat_series, cat_series) is False
    assert cont_cat_strategy.supports(cont_series, cont_series) is False

# Cat × Cat

def test_categorical_categorical():
    x = pd.Series(["A", "A", "B", "B"])
    y = pd.Series(["X", "X", "Y", "Y"])

    analyzer = CorrelationAnalyzer()
    result = analyzer.analyze(x, y)

    assert result["method"] == "cramers_v"
    assert 0 <= result["correlation"] <= 1


# Cont × Cont

def test_continuous_continuous():
    x = pd.Series([1, 2, 3, 4, 5])
    y = pd.Series([5, 4, 3, 2, 1])

    analyzer = CorrelationAnalyzer()
    result = analyzer.analyze(x, y)

    assert result["method"] == "spearman"
    assert result["correlation"] < 0


# Cont × Cat

def test_continuous_categorical():
    x = pd.Series(["A", "A", "B", "B"])
    y = pd.Series([10, 12, 30, 32])

    analyzer = CorrelationAnalyzer()
    result = analyzer.analyze(x, y)

    assert result["method"] == "eta_squared"
    assert 0 <= result["correlation"] <= 1

