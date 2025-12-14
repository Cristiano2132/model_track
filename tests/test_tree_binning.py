import pandas as pd
from model_track.binning.tree_binner import TreeBinner


def test_tree_binner_finds_splits():
    df = pd.DataFrame({
        "x": [1, 2, 3, 10, 11, 12],
        "y": [0, 0, 0, 1, 1, 1]
    })

    binner = TreeBinner(max_depth=2, min_samples_leaf=1)
    binner.fit(df, feature="x", target="y")

    splits = binner.bins_

    assert isinstance(splits, list)
    assert len(splits) > 0
    assert all(isinstance(x, (int, float)) for x in splits)


def test_tree_binner_bins_are_sorted_and_unique():
    df = pd.DataFrame({
        "x": [1, 2, 3, 10, 11, 12],
        "y": [0, 0, 0, 1, 1, 1]
    })

    binner = TreeBinner(max_depth=3, min_samples_leaf=1)
    binner.fit(df, feature="x", target="y")

    splits = binner.bins_

    assert splits == sorted(splits)
    assert len(splits) == len(set(splits))
    assert min(splits) > min(df["x"])
    assert max(splits) < max(df["x"])


def test_tree_binner_returns_empty_list_when_no_splits():
    df = pd.DataFrame({"x": [1, 1, 1], "y": [0, 1, 0]})

    binner = TreeBinner(max_depth=2, min_samples_leaf=1)
    binner.fit(df, feature="x", target="y")

    assert binner.bins_ == []


def test_tree_binner_constant_feature():
    df = pd.DataFrame({"x": [5, 5, 5], "y": [0, 1, 0]})

    binner = TreeBinner(max_depth=3, min_samples_leaf=1)
    binner.fit(df, feature="x", target="y")

    assert binner.bins_ == []


def test_tree_binner_handles_nans():
    df = pd.DataFrame({
        "x": [1, 2, None, 3, None, 10],
        "y": [0, 0, 1, 1, 1, 1]
    })

    binner = TreeBinner(max_depth=2, min_samples_leaf=1)
    binner.fit(df, feature="x", target="y")

    assert isinstance(binner.bins_, list)


def test_tree_binner_respects_max_depth():
    df = pd.DataFrame({
        "x": [1, 2, 3, 10, 11, 12],
        "y": [0, 0, 0, 1, 1, 1]
    })

    binner = TreeBinner(max_depth=1, min_samples_leaf=1)
    binner.fit(df, feature="x", target="y")

    assert len(binner.bins_) <= 1


def test_tree_binner_min_samples_leaf_too_large():
    df = pd.DataFrame({
        "x": [1, 2, 3, 4],
        "y": [0, 1, 0, 1]
    })

    binner = TreeBinner(max_depth=3, min_samples_leaf=10)
    binner.fit(df, feature="x", target="y")

    assert binner.bins_ == []