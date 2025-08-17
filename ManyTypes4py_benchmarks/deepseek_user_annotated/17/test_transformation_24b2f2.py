from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import math
import pandas as pd
import pytest
from numpy import nan, round, sqrt, floor, log as ln
from numpy.testing import assert_almost_equal
from pandas.testing import assert_frame_equal

from fklearn.training.transformation import (
    selector,
    capper,
    floorer,
    prediction_ranger,
    count_categorizer,
    label_categorizer,
    quantile_biner,
    truncate_categorical,
    rank_categorical,
    onehot_categorizer,
    target_categorizer,
    standard_scaler,
    ecdfer,
    discrete_ecdfer,
    custom_transformer,
    value_mapper,
    null_injector,
    missing_warner,
)


def test_selector() -> None:
    input_df = pd.DataFrame(
        {"feat1": [1, 2, 3], "feat2": [100, 200, 300], "target": [0, 1, 0]}
    )

    expected = pd.DataFrame({"feat1": [1, 2, 3], "target": [0, 1, 0]})

    expected2 = pd.DataFrame({"feat1": [1, 2, 3]})

    pred_fn, data, log = selector(input_df, ["feat1", "target"], ["feat1"])

    assert expected.equals(data)
    assert expected2.equals(pred_fn(input_df))


def test_capper() -> None:
    input_df = pd.DataFrame({"feat1": [10, 13, 50], "feat2": [50, 75, None]})

    input_df2 = pd.DataFrame({"feat1": [7, 15], "feat2": [200, None]})

    expected1 = pd.DataFrame({"feat1": [9, 9, 9], "feat2": [50, 75, None]})

    expected2 = pd.DataFrame({"feat1": [7, 9], "feat2": [75, None]})

    pred_fn1, data1, log = capper(input_df, ["feat1", "feat2"], {"feat1": 9})
    pred_fn2, data2, log = capper(
        input_df, ["feat1", "feat2"], {"feat1": 9}, suffix="_suffix"
    )
    pred_fn3, data3, log = capper(
        input_df, ["feat1", "feat2"], {"feat1": 9}, prefix="prefix_"
    )
    pred_fn4, data4, log = capper(
        input_df,
        ["feat1", "feat2"],
        {"feat1": 9},
        columns_mapping={"feat1": "feat1_raw", "feat2": "feat2_raw"},
    )

    assert expected1.equals(data1)
    assert expected2.equals(pred_fn1(input_df2))

    assert pd.concat(
        [expected1, input_df.copy().add_suffix("_suffix")], axis=1
    ).equals(data2)
    assert pd.concat(
        [expected2, input_df2.copy().add_suffix("_suffix")], axis=1
    ).equals(pred_fn2(input_df2))

    assert pd.concat(
        [expected1, input_df.copy().add_prefix("prefix_")], axis=1
    ).equals(data3)
    assert pd.concat(
        [expected2, input_df2.copy().add_prefix("prefix_")], axis=1
    ).equals(pred_fn3(input_df2))

    assert pd.concat(
        [expected1, input_df.copy().add_suffix("_raw")], axis=1
    ).equals(data4)
    assert pd.concat(
        [expected2, input_df2.copy().add_suffix("_raw")], axis=1
    ).equals(pred_fn4(input_df2))


def test_floorer() -> None:
    input_df = pd.DataFrame({"feat1": [10, 13, 10], "feat2": [50, 75, None]})

    input_df2 = pd.DataFrame({"feat1": [7, 15], "feat2": [15, None]})

    expected1 = pd.DataFrame({"feat1": [11, 13, 11], "feat2": [50, 75, None]})

    expected2 = pd.DataFrame({"feat1": [11, 15], "feat2": [50, None]})

    pred_fn1, data1, log = floorer(input_df, ["feat1", "feat2"], {"feat1": 11})
    pred_fn2, data2, log = floorer(
        input_df, ["feat1", "feat2"], {"feat1": 11}, suffix="_suffix"
    )
    pred_fn3, data3, log = floorer(
        input_df, ["feat1", "feat2"], {"feat1": 11}, prefix="prefix_"
    )
    pred_fn4, data4, log = floorer(
        input_df,
        ["feat1", "feat2"],
        {"feat1": 11},
        columns_mapping={"feat1": "feat1_raw", "feat2": "feat2_raw"},
    )

    assert expected1.equals(data1)
    assert expected2.equals(pred_fn1(input_df2))

    assert pd.concat(
        [expected1, input_df.copy().add_suffix("_suffix")], axis=1
    ).equals(data2)
    assert pd.concat(
        [expected2, input_df2.copy().add_suffix("_suffix")], axis=1
    ).equals(pred_fn2(input_df2))

    assert pd.concat(
        [expected1, input_df.copy().add_prefix("prefix_")], axis=1
    ).equals(data3)
    assert pd.concat(
        [expected2, input_df2.copy().add_prefix("prefix_")], axis=1
    ).equals(pred_fn3(input_df2))

    assert pd.concat(
        [expected1, input_df.copy().add_suffix("_raw")], axis=1
    ).equals(data4)
    assert pd.concat(
        [expected2, input_df2.copy().add_suffix("_raw")], axis=1
    ).equals(pred_fn4(input_df2))


def test_prediction_ranger() -> None:
    input_df = pd.DataFrame(
        {"feat1": [10, 13, 10, 15], "prediction": [100, 200, 300, None]}
    )

    pred_fn, data, log = prediction_ranger(input_df, 150, 250)

    expected = pd.DataFrame(
        {"feat1": [10, 13, 10, 15], "prediction": [150, 200, 250, None]}
    )

    assert expected.equals(data)


def test_value_mapper() -> None:
    input_df = pd.DataFrame(
        {
            "feat1": [10, 10, 13, 15],
            "feat2": [100, 200, 300, None],
            "feat3": ["a", "b", "c", "b"],
        }
    )

    value_maps = {
        "feat1": {10: 1, 13: 2},
        "feat2": {100: [1, 2, 3]},
        "feat3": {"a": "b", "b": nan},
    }

    pred_fn, data_ignore, log = value_mapper(input_df, value_maps)
    pred_fn2, data_ignore2, log2 = value_mapper(input_df, value_maps, suffix="_suffix")
    pred_fn3, data_ignore3, log3 = value_mapper(input_df, value_maps, prefix="prefix_")
    pred_fn4, data_ignore4, log4 = value_mapper(input_df, value_maps,
                                                columns_mapping={"feat1": "feat1_raw",
                                                                 "feat2": "feat2_raw",
                                                                 "feat3": "feat3_raw"})
    pred_fn, data_not_ignore, log = value_mapper(
        input_df, value_maps, ignore_unseen=False
    )

    expected_ignore = pd.DataFrame(
        {
            "feat1": [1, 1, 2, 15],
            "feat2": [[1, 2, 3], 200, 300, None],
            "feat3": ["b", nan, "c", nan],
        }
    )

    expected_not_ignore = pd.DataFrame(
        {
            "feat1": [1, 1, 2, nan],
            "feat2": [[1, 2, 3], nan, nan, nan],
            "feat3": ["b", nan, nan, nan],
        }
    )

    assert expected_ignore.equals(data_ignore)
    assert expected_not_ignore.equals(data_not_ignore)

    assert pd.concat(
        [expected_ignore, input_df.copy().add_suffix("_suffix")], axis=1
    ).equals(data_ignore2)

    assert pd.concat(
        [expected_ignore, input_df.copy().add_prefix("prefix_")], axis=1
    ).equals(data_ignore3)

    assert pd.concat(
        [expected_ignore, input_df.copy().add_suffix("_raw")], axis=1
    ).equals(data_ignore4)


def test_truncate_categorical() -> None:
    input_df_train = pd.DataFrame(
        {
            "col": ["a", "a", "a", "b", "b", "b", "b", "c", "d", "f", nan],
            "y": [1.0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    input_df_test = pd.DataFrame(
        {
            "col": ["a", "a", "b", "c", "d", "f", "e", nan],
            "y": [1.0, 0, 1, 1, 1, 0, 1, 1],
        }
    )

    expected_output_train = pd.DataFrame(
        {
            "col": [
                "a",
                "a",
                "a",
                "b",
                "b",
                "b",
                "b",
                -9999,
                -9999,
                -9999,
                nan,
            ],
            "y": [1.0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    expected_output_test = pd.DataFrame(
        {
            "col": ["a", "a", "b", -9999, -9999, -9999, -9999, nan],
            "y": [1.0, 0, 1, 1, 1, 0, 1, 1],
        }
    )

    truncate_learner1 = truncate_categorical(
        columns_to_truncate=["col"], percentile=0.1
    )
    truncate_learner2 = truncate_categorical(
        columns_to_truncate=["col"], percentile=0.1, suffix="_suffix"
    )
    truncate_learner3 = truncate_categorical(
        columns_to_truncate=["col"], percentile=0.1, prefix="prefix_"
    )
    truncate_learner4 = truncate_categorical(
        columns_to_truncate=["col"],
        percentile=0.1,
        columns_mapping={"col": "col_raw"},
    )

    pred_fn1, data1, log = truncate_learner1(input_df_train)
    pred_fn2, data2, log = truncate_learner2(input_df_train)
    pred_fn3, data3, log = truncate_learner3(input_df_train)
    pred_fn4, data4, log = truncate_learner4(input_df_train)

    assert expected_output_train.equals(data1)
    assert expected_output_test.equals(pred_fn1(input_df_test))

    assert pd.concat(
        [
            expected_output_train,
            input_df_train[["col"]].copy().add_suffix("_suffix"),
        ],
        axis=1,
    ).equals(data2)
    assert pd.concat(
        [
            expected_output_test,
            input_df_test[["col"]].copy().add_suffix("_suffix"),
        ],
        axis=1,
    ).equals(pred_fn2(input_df_test))

    assert pd.concat(
        [
            expected_output_train,
            input_df_train[["col"]].copy().add_prefix("prefix_"),
        ],
        axis=1,
    ).equals(data3)
    assert pd.concat(
        [
            expected_output_test,
            input_df_test[["col"]].copy().add_prefix("prefix_"),
        ],
        axis=1,
    ).equals(pred_fn3(input_df_test))

    assert pd.concat(
        [
            expected_output_train,
            input_df_train[["col"]].copy().add_suffix("_raw"),
        ],
        axis=1,
    ).equals(data4)
    assert pd.concat(
        [
            expected_output_test,
            input_df_test[["col"]].copy().add_suffix("_raw"),
        ],
        axis=1,
    ).equals(pred_fn4(input_df_test))


def test_rank_categorical() -> None:
    input_df_train = pd.DataFrame(
        {"col": ["a", "b", "b", "c", "c", "d", "d", "d", nan, nan, nan]}
    )

    input_df_test = pd.DataFrame({"col": ["a", "b", "c", "d", "d", nan, nan]})

    expected_output_train = pd.DataFrame(
        {"col": [4, 2, 2, 3, 3, 1, 1, 1, nan, nan, nan]}
    )

    expected_output_test = pd.DataFrame({"col": [4, 2, 3, 1, 1, nan, nan]})

    pred_fn1, data1, log = rank_categorical(input_df_train, ["col"])
    pred_fn2, data2, log = rank_categorical(
        input_df_train, ["col"], suffix="_suffix"
    )
    pred_fn3, data3, log = rank_categorical(
        input_df_train, ["col"], prefix="prefix_"
    )
    pred_fn4, data4, log = rank_categorical(
        input_df_train, ["col"], columns_mapping={"col": "col_raw"}
    )

    assert expected_output_train.equals(data1)
    assert expected_output_test.equals(pred_fn1(input_df_test))

    assert pd.concat(
        [
            expected_output_train,
            input_df_train[["col"]].copy().add_suffix("_suffix"),
        ],
        axis=1,
    ).equals(data2)
    assert pd.concat(
        [
            expected_output_test,
            input_df_test[["col"]].copy().add_suffix("_suffix"),
        ],
        axis=1,
    ).equals(pred_fn2(input_df_test))

    assert pd.concat(
        [
            expected_output_train,
            input_df_train[["col"]].copy().add_prefix("prefix_"),
        ],
        axis=1,
    ).equals(data3)
    assert pd.concat(
        [
            expected_output_test,
            input_df_test[["col"]].copy().add_prefix("prefix_"),
        ],
        axis=1,
    ).equals(pred_fn3(input_df_test))

    assert pd.concat(
        [
            expected_output_train,
            input_df_train[["col"]].copy().add_suffix("_raw"),
        ],
        axis=1,
    ).equals(data4)
    assert pd.concat(
        [
            expected_output_test,
            input_df_test[["col"]].copy().add_suffix("_raw"),
        ],
        axis=1,
    ).equals(pred_fn4(input_df_test))


def test_count_categorizer() -> None:
    input_df_train = pd.DataFrame(
        {
            "feat1_num": [1, 0.5, nan, 100],
            "feat2_cat": ["a", "a", "a", "b"],
            "feat3_cat": ["c", "c", "c", nan],
        }
    )

    expected_output_train = pd.DataFrame(
        {
            "feat1_num": [1, 0.5, nan, 100],
            "feat2_cat": [3, 3, 3, 1],
            "feat3_cat": [3, 3, 3, nan],
        }
    )

    input_df_test = pd.DataFrame(
        {
            "feat1_num": [2, 20, 200, 2000],
            "feat2_cat": ["a", "b", "b", "d"],
            "feat3_cat": [nan, nan, "c", "c"],
        }
    )

    expected_output_test = pd.DataFrame(
        {
            "feat1_num": [2, 20, 200, 2000],
            "feat2_cat": [3, 1, 1, 1],  # replace unseen vars with constant (1)
            "feat3_cat": [nan, nan, 3, 3],
        }
    )

    categorizer_learner1 = count_categorizer(
        columns_to_categorize=["feat2_cat", "feat3_cat"], replace_unseen=1
    )
    categorizer_learner2 = count_categorizer(
        columns_to_categorize=["feat2_cat", "feat3_cat"],
        replace_unseen=1,
        suffix="_suffix",
    )
    categorizer_learner3 = count_categorizer(
        columns_to_categorize=["feat2_cat", "feat3_cat"],
        replace_unseen=1,
        prefix="prefix_",
    )
    categorizer_learner4 = count_categorizer(
        columns_to_categorize=["feat2_cat", "feat3_cat"],
        replace_unseen=1,
        columns_mapping={
            "feat2_cat": "feat2_cat_raw",
            "feat3_cat": "feat3_cat_raw",
        },
    )

    pred_fn1, data1, log = categorizer_