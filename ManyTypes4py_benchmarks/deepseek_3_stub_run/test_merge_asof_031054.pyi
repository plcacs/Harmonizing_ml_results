import datetime
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd
import pandas._testing as tm
import pandas.util._test_decorators as td
import pytest
from pandas import Index, Timedelta, merge_asof, option_context, to_datetime
from pandas.core.reshape.merge import MergeError


class TestAsOfMerge:
    def prep_data(self, df: pd.DataFrame, dedupe: bool = ...) -> pd.DataFrame: ...

    @pytest.fixture
    def trades(self) -> pd.DataFrame: ...

    @pytest.fixture
    def quotes(self) -> pd.DataFrame: ...

    @pytest.fixture
    def asof(self) -> pd.DataFrame: ...

    @pytest.fixture
    def tolerance(self) -> pd.DataFrame: ...

    def test_examples1(self) -> None: ...

    def test_examples2(self, unit: str) -> None: ...

    def test_examples3(self) -> None: ...

    def test_examples4(self) -> None: ...

    def test_basic(
        self, trades: pd.DataFrame, asof: pd.DataFrame, quotes: pd.DataFrame
    ) -> None: ...

    def test_basic_categorical(
        self, trades: pd.DataFrame, asof: pd.DataFrame, quotes: pd.DataFrame
    ) -> None: ...

    def test_basic_left_index(
        self, trades: pd.DataFrame, asof: pd.DataFrame, quotes: pd.DataFrame
    ) -> None: ...

    def test_basic_right_index(
        self, trades: pd.DataFrame, asof: pd.DataFrame, quotes: pd.DataFrame
    ) -> None: ...

    def test_basic_left_index_right_index(
        self, trades: pd.DataFrame, asof: pd.DataFrame, quotes: pd.DataFrame
    ) -> None: ...

    def test_multi_index_left(
        self, trades: pd.DataFrame, quotes: pd.DataFrame
    ) -> None: ...

    def test_multi_index_right(
        self, trades: pd.DataFrame, quotes: pd.DataFrame
    ) -> None: ...

    def test_on_and_index_left_on(
        self, trades: pd.DataFrame, quotes: pd.DataFrame
    ) -> None: ...

    def test_on_and_index_right_on(
        self, trades: pd.DataFrame, quotes: pd.DataFrame
    ) -> None: ...

    def test_basic_left_by_right_by(
        self, trades: pd.DataFrame, asof: pd.DataFrame, quotes: pd.DataFrame
    ) -> None: ...

    def test_missing_right_by(
        self, trades: pd.DataFrame, asof: pd.DataFrame, quotes: pd.DataFrame
    ) -> None: ...

    def test_multiby(self) -> None: ...

    @pytest.mark.parametrize("dtype", ["object", "string"])
    def test_multiby_heterogeneous_types(self, dtype: str) -> None: ...

    def test_mismatched_index_dtype(self) -> None: ...

    def test_multiby_indexed(self) -> None: ...

    def test_basic2(self, datapath: Any) -> None: ...

    def test_basic_no_by(
        self, trades: pd.DataFrame, asof: pd.DataFrame, quotes: pd.DataFrame
    ) -> None: ...

    def test_valid_join_keys(
        self, trades: pd.DataFrame, quotes: pd.DataFrame
    ) -> None: ...

    def test_with_duplicates(
        self, datapath: Any, trades: pd.DataFrame, quotes: pd.DataFrame, asof: pd.DataFrame
    ) -> None: ...

    def test_with_duplicates_no_on(self) -> None: ...

    def test_valid_allow_exact_matches(
        self, trades: pd.DataFrame, quotes: pd.DataFrame
    ) -> None: ...

    def test_valid_tolerance(
        self, trades: pd.DataFrame, quotes: pd.DataFrame
    ) -> None: ...

    def test_non_sorted(
        self, trades: pd.DataFrame, quotes: pd.DataFrame
    ) -> None: ...

    @pytest.mark.parametrize(
        "tolerance_ts",
        [Timedelta("1day"), datetime.timedelta(days=1)],
        ids=["Timedelta", "datetime.timedelta"],
    )
    def test_tolerance(
        self,
        tolerance_ts: Union[Timedelta, datetime.timedelta],
        trades: pd.DataFrame,
        quotes: pd.DataFrame,
        tolerance: pd.DataFrame,
    ) -> None: ...

    def test_tolerance_forward(self) -> None: ...

    def test_tolerance_nearest(self) -> None: ...

    def test_tolerance_tz(self, unit: str) -> None: ...

    def test_tolerance_float(self) -> None: ...

    def test_index_tolerance(
        self,
        trades: pd.DataFrame,
        quotes: pd.DataFrame,
        tolerance: pd.DataFrame,
    ) -> None: ...

    def test_allow_exact_matches(
        self, trades: pd.DataFrame, quotes: pd.DataFrame
    ) -> None: ...

    def test_allow_exact_matches_forward(self) -> None: ...

    def test_allow_exact_matches_nearest(self) -> None: ...

    def test_allow_exact_matches_and_tolerance(
        self, trades: pd.DataFrame, quotes: pd.DataFrame
    ) -> None: ...

    def test_allow_exact_matches_and_tolerance2(self) -> None: ...

    def test_allow_exact_matches_and_tolerance3(self) -> None: ...

    def test_allow_exact_matches_and_tolerance_forward(self) -> None: ...

    def test_allow_exact_matches_and_tolerance_nearest(self) -> None: ...

    def test_forward_by(self) -> None: ...

    def test_nearest_by(self) -> None: ...

    def test_by_int(self) -> None: ...

    def test_on_float(self) -> None: ...

    def test_on_specialized_type(self, any_real_numpy_dtype: Any) -> None: ...

    def test_on_specialized_type_by_int(self, any_real_numpy_dtype: Any) -> None: ...

    def test_on_float_by_int(self) -> None: ...

    def test_merge_datatype_error_raises(self) -> None: ...

    def test_merge_datatype_categorical_error_raises(self) -> None: ...

    def test_merge_groupby_multiple_column_with_categorical_column(self) -> None: ...

    @pytest.mark.parametrize("func", [lambda x: x, to_datetime], ids=["numeric", "datetime"])
    @pytest.mark.parametrize("side", ["left", "right"])
    def test_merge_on_nans(self, func: Any, side: str) -> None: ...

    def test_by_nullable(self, any_numeric_ea_dtype: Any, using_infer_string: bool) -> None: ...

    def test_merge_by_col_tz_aware(self) -> None: ...

    def test_by_mixed_tz_aware(self, using_infer_string: bool) -> None: ...

    @pytest.mark.parametrize("dtype", ["float64", "int16", "m8[ns]", "M8[us]"])
    def test_by_dtype(self, dtype: str) -> None: ...

    def test_timedelta_tolerance_nearest(self, unit: str) -> None: ...

    def test_int_type_tolerance(self, any_int_dtype: Any) -> None: ...

    def test_merge_index_column_tz(self) -> None: ...

    def test_left_index_right_index_tolerance(self, unit: str) -> None: ...


@pytest.mark.parametrize("infer_string", [False, pytest.param(True, marks=td.skip_if_no("pyarrow"))])
@pytest.mark.parametrize("kwargs", [{"on": "x"}, {"left_index": True, "right_index": True}])
@pytest.mark.parametrize("data", [["2019-06-01 00:09:12", "2019-06-01 00:10:29"], [1.0, "2019-06-01 00:10:29"]])
def test_merge_asof_non_numerical_dtype(
    kwargs: dict[str, Any], data: list[Any], infer_string: bool
) -> None: ...


def test_merge_asof_non_numerical_dtype_object() -> None: ...


@pytest.mark.parametrize(
    "kwargs",
    [
        {"right_index": True, "left_index": True},
        {"left_on": "left_time", "right_index": True},
        {"left_index": True, "right_on": "right"},
    ],
)
def test_merge_asof_index_behavior(kwargs: dict[str, Any]) -> None: ...


def test_merge_asof_numeric_column_in_index() -> None: ...


def test_merge_asof_numeric_column_in_multiindex() -> None: ...


def test_merge_asof_numeri_column_in_index_object_dtype() -> None: ...


def test_merge_asof_array_as_on(unit: str) -> None: ...


def test_merge_asof_raise_for_duplicate_columns() -> None: ...


@pytest.mark.parametrize(
    "dtype",
    [
        "Int64",
        pytest.param("int64[pyarrow]", marks=td.skip_if_no("pyarrow")),
        pytest.param("timestamp[s][pyarrow]", marks=td.skip_if_no("pyarrow")),
    ],
)
def test_merge_asof_extension_dtype(dtype: str) -> None: ...


@td.skip_if_no("pyarrow")
def test_merge_asof_pyarrow_td_tolerance() -> None: ...


def test_merge_asof_read_only_ndarray() -> None: ...


def test_merge_asof_multiby_with_categorical() -> None: ...