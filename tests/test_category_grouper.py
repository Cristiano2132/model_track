from model_track.stability.category_mapper import CategoryMapper



# ------------------------------------------------------------------
# Core behavior
# ------------------------------------------------------------------

def test_create_map_numeric_ordered(global_woe_numeric_category_mapper):
    mapper = CategoryMapper()

    mapper.create_map(
        df=global_woe_numeric_category_mapper,
        feature_name="income",
        category_col="income",
        groups=[[0, 1], [4, 5]],
        ordered=True,
    )

    mapping = mapper.get("income")

    assert mapping["10"] == "<=20"
    assert mapping["20"] == "<=20"
    assert mapping["50"] == ">=50"
    assert mapping["60"] == ">=50"
    assert mapping["30"] == "30"
    assert mapping["40"] == "40"


def test_create_map_middle_group_numeric(global_woe_numeric_category_mapper):
    mapper = CategoryMapper()

    mapper.create_map(
        df=global_woe_numeric_category_mapper,
        feature_name="income",
        category_col="income",
        groups=[[2, 3]],
        ordered=True,
    )

    mapping = mapper.get("income")

    assert mapping["30"] == "[30, 40]"
    assert mapping["40"] == "[30, 40]"


def test_create_map_unordered_fallback(global_woe_numeric_category_mapper):
    mapper = CategoryMapper()

    mapper.create_map(
        df=global_woe_numeric_category_mapper,
        feature_name="income",
        category_col="income",
        groups=[[1, 3]],
        ordered=False,
    )

    mapping = mapper.get("income")

    assert mapping["20"] == "20 | 40"
    assert mapping["40"] == "20 | 40"


# ------------------------------------------------------------------
# Interval logic
# ------------------------------------------------------------------

def test_merge_intervals_continuous(global_woe_intervals_category_mapper):
    mapper = CategoryMapper()

    mapper.create_map(
        df=global_woe_intervals_category_mapper,
        feature_name="age",
        category_col="age",
        groups=[[0, 1]],
        ordered=True,
    )

    mapping = mapper.get("age")

    assert mapping["<=1"] == "<=3"
    assert mapping["(1,3]"] == "<=3"


def test_merge_intervals_non_continuous(global_woe_intervals_category_mapper):
    mapper = CategoryMapper()

    mapper.create_map(
        df=global_woe_intervals_category_mapper,
        feature_name="age",
        category_col="age",
        groups=[[0, 2]],
        ordered=True,
    )

    mapping = mapper.get("age")

    assert mapping["<=1"] == "<=1 | (3,5]"
    assert mapping["(3,5]"] == "<=1 | (3,5]"


# ------------------------------------------------------------------
# Multiple features & state
# ------------------------------------------------------------------

def test_multiple_features_same_instance(global_woe_numeric_category_mapper, global_woe_intervals_category_mapper):
    mapper = CategoryMapper()

    mapper.create_map(
        df=global_woe_numeric_category_mapper,
        feature_name="income",
        category_col="income",
        groups=[[0, 1]],
    )

    mapper.create_map(
        df=global_woe_intervals_category_mapper,
        feature_name="age",
        category_col="age",
        groups=[[1, 2]],
    )

    all_maps = mapper.get()

    assert "income" in all_maps
    assert "age" in all_maps


# ------------------------------------------------------------------
# get / set API
# ------------------------------------------------------------------

def test_get_all_and_single_feature(global_woe_numeric_category_mapper):
    mapper = CategoryMapper()

    mapper.create_map(
        df=global_woe_numeric_category_mapper,
        feature_name="income",
        category_col="income",
        groups=[[0, 1]],
    )

    assert isinstance(mapper.get(), dict)
    assert isinstance(mapper.get("income"), dict)


def test_set_override_mapping():
    mapper = CategoryMapper()

    manual_map = {
        "A": "GROUP_1",
        "B": "GROUP_1",
    }

    mapper.set("feature_x", manual_map)

    assert mapper.get("feature_x") == manual_map


def test_merge_intervals_starting_with_minus_infinity():
    mapper = CategoryMapper()

    categories = ["<=1", "(1,3]"]

    label = mapper._merge_interval_labels(categories)

    assert label == "<=3"


def test_merge_intervals_middle_range():
    mapper = CategoryMapper()

    categories = ["(10,20]", "(20,40]"]

    label = mapper._merge_interval_labels(categories)

    assert label == "(10,40]"


def test_merge_single_interval():
    mapper = CategoryMapper()

    categories = ["(5,7]"]

    label = mapper._merge_interval_labels(categories)

    assert label == "(5,7]"


def test_merge_intervals_with_whitespace():
    mapper = CategoryMapper()

    categories = [" <=1 ", " (1, 3] "]

    label = mapper._merge_interval_labels(categories)

    assert label == "<=3"


def test_merge_intervals_unordered_input():
    mapper = CategoryMapper()

    categories = ["(1,3]", "<=1"]

    label = mapper._merge_interval_labels(categories)

    assert label == "<=3"

def test_merge_intervals_ending_with_plus_infinity():
    mapper = CategoryMapper()

    categories = ["(3,5]", ">=5"]

    label = mapper._merge_interval_labels(categories)

    assert label == ">=3"

def test_check_continuous_intervals_true():
    mapper = CategoryMapper()

    categories = ["<=1", "(1,3]", "(3,5]", "(5,7]"]
    group_cats = ["<=1", "(1,3]", "(3,5]"]

    result = mapper._check_continuous_intervals(
        categories=categories,
        group_cats=group_cats,
    )

    assert result is True

def test_check_continuous_intervals_false_gap():
    mapper = CategoryMapper()

    categories = ["<=1", "(1,3]", "(3,5]", "(5,7]"]
    group_cats = ["<=1", "(3,5]"]

    result = mapper._check_continuous_intervals(
        categories=categories,
        group_cats=group_cats,
    )

    assert result is False