from __future__ import annotations

from datetime import date, datetime, timedelta
from itertools import product
import re
from typing import Any

import numpy as np
import pytest

from pandas._config import using_string_dtype
from pandas.compat.numpy import np_version_gte1p25

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    Grouper,
    Index,
    MultiIndex,
    Series,
    concat,
    date_range,
)
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.core.reshape import reshape as reshape_lib
from pandas.core.reshape.pivot import pivot_table


class TestPivotTable:

    @pytest.fixture
    def data(self) -> DataFrame: ...

    def test_pivot_table(self, observed: bool, data: DataFrame) -> None: ...

    def test_pivot_table_categorical_observed_equal(self, observed: bool) -> None: ...

    def test_pivot_table_nocols(self) -> None: ...

    def test_pivot_table_dropna(self) -> None: ...

    def test_pivot_table_categorical(self) -> None: ...

    def test_pivot_table_dropna_categoricals(self, dropna: bool) -> None: ...

    def test_pivot_with_non_observable_dropna(self, dropna: bool) -> None: ...

    def test_pivot_with_non_observable_dropna_multi_cat(self, dropna: bool) -> None: ...

    @pytest.mark.parametrize("left_right", [([0] * 4, [1] * 4), (range(3), range(1, 4))])
    def test_pivot_with_interval_index(self, left_right: tuple[Any, Any], dropna: bool, closed: str) -> None: ...

    def test_pivot_with_interval_index_margins(self) -> None: ...

    def test_pass_array(self, data: DataFrame) -> None: ...

    def test_pass_function(self, data: DataFrame) -> None: ...

    def test_pivot_table_multiple(self, data: DataFrame) -> None: ...

    def test_pivot_dtypes(self) -> None: ...

    @pytest.mark.parametrize(
        "columns,values",
        [
            ("bool1", ["float1", "float2"]),
            ("bool1", ["float1", "float2", "bool1"]),
            ("bool2", ["float1", "float2", "bool1"]),
        ],
    )
    def test_pivot_preserve_dtypes(self, columns: str, values: list[str]) -> None: ...

    def test_pivot_no_values(self) -> None: ...

    def test_pivot_multi_values(self, data: DataFrame) -> None: ...

    def test_pivot_multi_functions(self, data: DataFrame) -> None: ...

    @pytest.mark.parametrize("method", [True, False])
    def test_pivot_index_with_nan(self, method: bool) -> None: ...

    @pytest.mark.parametrize("method", [True, False])
    def test_pivot_index_with_nan_dates(self, method: bool) -> None: ...

    @pytest.mark.parametrize("method", [True, False])
    def test_pivot_with_tz(self, method: bool, unit: str) -> None: ...

    def test_pivot_tz_in_values(self) -> None: ...

    @pytest.mark.parametrize("method", [True, False])
    def test_pivot_periods(self, method: bool) -> None: ...

    def test_pivot_periods_with_margins(self) -> None: ...

    @pytest.mark.parametrize("box", [list, np.array, Series, Index])
    @pytest.mark.parametrize("method", [True, False])
    def test_pivot_with_list_like_values(self, box: type, method: bool) -> None: ...

    @pytest.mark.parametrize(
        "values",
        [
            ["bar", "baz"],
            np.array(["bar", "baz"]),
            Series(["bar", "baz"]),
            Index(["bar", "baz"]),
        ],
    )
    @pytest.mark.parametrize("method", [True, False])
    def test_pivot_with_list_like_values_nans(self, values: Any, method: bool) -> None: ...

    def test_pivot_columns_none_raise_error(self) -> None: ...

    @pytest.mark.xfail(
        reason="MultiIndexed unstack with tuple names fails with KeyError GH#19966"
    )
    @pytest.mark.parametrize("method", [True, False])
    def test_pivot_with_multiindex(self, method: bool) -> None: ...

    @pytest.mark.parametrize("method", [True, False])
    def test_pivot_with_tuple_of_values(self, method: bool) -> None: ...

    def _check_output(
        self,
        result: DataFrame,
        values_col: str,
        data: DataFrame,
        index: list[str] | None = ...,
        columns: list[str] | None = ...,
        margins_col: str = ...,
    ) -> None: ...

    def test_margins(self, data: DataFrame) -> None: ...

    def test_no_col(self, data: DataFrame, using_infer_string: bool) -> None: ...

    @pytest.mark.parametrize(
        "columns, aggfunc, values, expected_columns",
        [
            (
                "A",
                "mean",
                [[5.5, 5.5, 2.2, 2.2], [8.0, 8.0, 4.4, 4.4]],
                Index(["bar", "All", "foo", "All"], name="A"),
            ),
            (
                ["A", "B"],
                "sum",
                [[9, 13, 22, 5, 6, 11], [14, 18, 32, 11, 11, 22]],
                MultiIndex.from_tuples(
                    [
                        ("bar", "one"),
                        ("bar", "two"),
                        ("bar", "All"),
                        ("foo", "one"),
                        ("foo", "two"),
                        ("foo", "All"),
                    ],
                    names=["A", "B"],
                ),
            ),
        ],
    )
    def test_margin_with_only_columns_defined(
        self,
        columns: Any,
        aggfunc: str,
        values: list[list[Any]],
        expected_columns: Index | MultiIndex,
        using_infer_string: bool,
    ) -> None: ...

    def test_margins_dtype(self, data: DataFrame) -> None: ...

    def test_margins_dtype_len(self, data: DataFrame) -> None: ...

    @pytest.mark.parametrize(
        "cols", [(1, 2), ("a", "b"), (1, "b"), ("a", 1)]
    )
    def test_pivot_table_multiindex_only(self, cols: tuple[Any, Any]) -> None: ...

    def test_pivot_table_retains_tz(self) -> None: ...

    def test_pivot_integer_columns(self) -> None: ...

    def test_pivot_no_level_overlap(self) -> None: ...

    def test_pivot_columns_lexsorted(self) -> None: ...

    def test_pivot_complex_aggfunc(self, data: DataFrame) -> None: ...

    def test_margins_no_values_no_cols(self, data: DataFrame) -> None: ...

    def test_margins_no_values_two_rows(self, data: DataFrame) -> None: ...

    def test_margins_no_values_one_row_one_col(self, data: DataFrame) -> None: ...

    def test_margins_no_values_two_row_two_cols(self, data: DataFrame) -> None: ...

    @pytest.mark.parametrize(
        "margin_name", ["foo", "one", 666, None, ["a", "b"]]
    )
    def test_pivot_table_with_margins_set_margin_name(
        self, margin_name: Any, data: DataFrame
    ) -> None: ...

    def test_pivot_timegrouper(self) -> None: ...

    def test_pivot_timegrouper_double(self) -> None: ...

    def test_pivot_datetime_tz(self) -> None: ...

    def test_pivot_dtaccessor(self) -> None: ...

    def test_daily(self) -> None: ...

    def test_monthly(self) -> None: ...

    def test_pivot_table_with_iterator_values(self, data: DataFrame) -> None: ...

    def test_pivot_table_margins_name_with_aggfunc_list(self) -> None: ...

    def test_categorical_margins(self, observed: bool) -> None: ...

    def test_categorical_margins_category(self, observed: bool) -> None: ...

    def test_margins_casted_to_float(self) -> None: ...

    def test_pivot_with_categorical(self, observed: bool, ordered: bool) -> None: ...

    def test_categorical_aggfunc(self, observed: bool) -> None: ...

    def test_categorical_pivot_index_ordering(self, observed: bool) -> None: ...

    def test_pivot_table_not_series(self) -> None: ...

    def test_pivot_margins_name_unicode(self) -> None: ...

    def test_pivot_string_as_func(self) -> None: ...

    @pytest.mark.parametrize(
        "kwargs", [{"a": 2}, {"a": 2, "b": 3}, {"b": 3, "a": 2}]
    )
    def test_pivot_table_kwargs(self, kwargs: dict[str, int]) -> None: ...

    @pytest.mark.parametrize(
        "kwargs",
        [{}, {"b": 10}, {"a": 3}, {"a": 3, "b": 10}, {"b": 10, "a": 3}],
    )
    def test_pivot_table_kwargs_margin(
        self, data: DataFrame, kwargs: dict[str, int]
    ) -> None: ...

    @pytest.mark.parametrize(
        "f, f_numpy",
        [
            ("sum", np.sum),
            ("mean", np.mean),
            ("min", np.min),
            (["sum", "mean"], [np.sum, np.mean]),
            (["sum", "min"], [np.sum, np.min]),
            (["max", "mean"], [np.max, np.mean]),
        ],
    )
    def test_pivot_string_func_vs_func(
        self, f: Any, f_numpy: Any, data: DataFrame
    ) -> None: ...

    @pytest.mark.slow
    def test_pivot_number_of_levels_larger_than_int32(
        self, performance_warning: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None: ...

    def test_pivot_table_aggfunc_dropna(self, dropna: bool) -> None: ...

    def test_pivot_table_aggfunc_scalar_dropna(self, dropna: bool) -> None: ...

    @pytest.mark.parametrize("margins", [True, False])
    def test_pivot_table_empty_aggfunc(self, margins: bool) -> None: ...

    def test_pivot_table_no_column_raises(self) -> None: ...

    def test_pivot_table_multiindex_columns_doctest_case(self) -> None: ...

    def test_pivot_table_sort_false(self) -> None: ...

    def test_pivot_table_nullable_margins(self) -> None: ...

    def test_pivot_table_sort_false_with_multiple_values(self) -> None: ...

    def test_pivot_table_with_margins_and_numeric_columns(self) -> None: ...

    @pytest.mark.parametrize(
        "dtype,expected_dtype",
        [("Int64", "Float64"), ("int64", "float64")],
    )
    def test_pivot_ea_dtype_dropna(
        self, dropna: bool, dtype: str, expected_dtype: str
    ) -> None: ...

    def test_pivot_table_datetime_warning(self) -> None: ...

    def test_pivot_table_with_mixed_nested_tuples(self) -> None: ...

    def test_pivot_table_aggfunc_nunique_with_different_values(self) -> None: ...


class TestPivot:

    def test_pivot(self) -> None: ...

    def test_pivot_duplicates(self) -> None: ...

    def test_pivot_empty(self) -> None: ...

    def test_pivot_integer_bug(self, any_string_dtype: str) -> None: ...

    def test_pivot_index_none(self) -> None: ...

    def test_pivot_index_list_values_none_immutable_args(self) -> None: ...

    def test_pivot_columns_not_given(self) -> None: ...

    @pytest.mark.xfail(
        using_string_dtype(), reason="TODO(infer_string) None is cast to NaN"
    )
    def test_pivot_columns_is_none(self) -> None: ...

    def test_pivot_index_is_none(self, using_infer_string: bool) -> None: ...

    def test_pivot_values_is_none(self) -> None: ...

    def test_pivot_not_changing_index_name(self) -> None: ...

    def test_pivot_table_empty_dataframe_correct_index(self) -> None: ...

    def test_pivot_table_handles_explicit_datetime_types(self) -> None: ...

    def test_pivot_table_with_margins_and_numeric_column_names(self) -> None: ...

    @pytest.mark.parametrize("m", [1, 10])
    def test_unstack_copy(self, m: int) -> None: ...

    def test_pivot_empty_with_datetime(self) -> None: ...

    def test_pivot_margins_with_none_index(self) -> None: ...