from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Protocol, Sequence, Type, TypeVar, Union, overload
from importlib import reload
import string
import sys

import numpy as np
import pytest

from pandas._libs.tslibs import iNaT
import pandas.util._test_decorators as td
from pandas import (
    NA,
    Categorical,
    CategoricalDtype,
    DatetimeTZDtype,
    Index,
    Interval,
    NaT,
    Series,
    Timedelta,
    Timestamp,
    cut,
    date_range,
    to_datetime,
)
import pandas._testing as tm
import pandas as pd


def rand_str(nchars: int) -> str: ...


class TestAstypeAPI:
    def test_astype_unitless_dt64_raises(self) -> None: ...
    def test_arg_for_errors_in_astype(self) -> None: ...
    @pytest.mark.parametrize("dtype_class", [dict, Series])
    def test_astype_dict_like(self, dtype_class: type) -> None: ...


class TestAstype:
    @pytest.mark.parametrize("tz", [None, "UTC", "US/Pacific"])
    def test_astype_object_to_dt64_non_nano(self, tz: Optional[str]) -> None: ...
    def test_astype_mixed_object_to_dt64tz(self) -> None: ...
    @pytest.mark.parametrize("dtype", np.typecodes["All"])
    def test_astype_empty_constructor_equality(self, dtype: str) -> None: ...
    @pytest.mark.parametrize("dtype", [str, np.str_])
    @pytest.mark.parametrize(
        "data",
        [
            [string.digits * 10, rand_str(63), rand_str(64), rand_str(1000)],
            [string.digits * 10, rand_str(63), rand_str(64), np.nan, 1.0],
        ],
    )
    def test_astype_str_map(
        self,
        dtype: type,
        data: List[Union[str, float]],
        using_infer_string: bool,
    ) -> None: ...
    def test_astype_float_to_period(self) -> None: ...
    def test_astype_no_pandas_dtype(self) -> None: ...
    @pytest.mark.parametrize("dtype", [np.datetime64, np.timedelta64])
    def test_astype_generic_timestamp_no_frequency(
        self, dtype: type, request: Any
    ) -> None: ...
    def test_astype_dt64_to_str(self) -> None: ...
    def test_astype_dt64tz_to_str(self) -> None: ...
    def test_astype_datetime(self, unit: str) -> None: ...
    def test_astype_datetime64tz(self) -> None: ...
    def test_astype_str_cast_dt64(self) -> None: ...
    def test_astype_str_cast_td64(self) -> None: ...
    def test_dt64_series_astype_object(self) -> None: ...
    def test_td64_series_astype_object(self) -> None: ...
    @pytest.mark.parametrize(
        "data, dtype",
        [
            (["x", "y", "z"], "string[python]"),
            pytest.param(
                ["x", "y", "z"],
                "string[pyarrow]",
                marks=td.skip_if_no("pyarrow"),
            ),
            (["x", "y", "z"], "category"),
            (3 * [Timestamp("2020-01-01", tz="UTC")], None),
            (3 * [Interval(0, 1)], None),
        ],
    )
    @pytest.mark.parametrize("errors", ["raise", "ignore"])
    def test_astype_ignores_errors_for_extension_dtypes(
        self,
        data: List[Any],
        dtype: Optional[str],
        errors: str,
    ) -> None: ...
    def test_astype_from_float_to_str(self, any_float_dtype: str) -> None: ...
    @pytest.mark.parametrize(
        "value, string_value",
        [(None, "None"), (np.nan, "nan"), (NA, "<NA>")],
    )
    def test_astype_to_str_preserves_na(
        self,
        value: Optional[Any],
        string_value: str,
        using_infer_string: bool,
    ) -> None: ...
    @pytest.mark.parametrize("dtype", ["float32", "float64", "int64", "int32"])
    def test_astype(self, dtype: str) -> None: ...
    @pytest.mark.parametrize("value", [np.nan, np.inf])
    def test_astype_cast_nan_inf_int(
        self, any_int_numpy_dtype: str, value: float
    ) -> None: ...
    def test_astype_cast_object_int_fail(self, any_int_numpy_dtype: str) -> None: ...
    def test_astype_float_to_uint_negatives_raise(
        self,
        float_numpy_dtype: str,
        any_unsigned_int_numpy_dtype: str,
    ) -> None: ...
    def test_astype_cast_object_int(self) -> None: ...
    def test_astype_unicode(self, using_infer_string: bool) -> None: ...
    def test_astype_bytes(self) -> None: ...
    def test_astype_nan_to_bool(self) -> None: ...
    def test_astype_ea_to_datetimetzdtype(self, any_numeric_ea_dtype: str) -> None: ...
    def test_astype_retain_attrs(self, any_numpy_dtype: str) -> None: ...


class TestAstypeString:
    @pytest.mark.parametrize(
        "data, dtype",
        [
            ([True, NA], "boolean"),
            (["A", NA], "category"),
            (["2020-10-10", "2020-10-10"], "datetime64[ns]"),
            (["2020-10-10", "2020-10-10", NaT], "datetime64[ns]"),
            (
                ["2012-01-01 00:00:00-05:00", NaT],
                "datetime64[ns, US/Eastern]",
            ),
            ([1, None], "UInt16"),
            (["1/1/2021", "2/1/2021"], "period[M]"),
            (["1/1/2021", "2/1/2021", NaT], "period[M]"),
            (["1 Day", "59 Days", NaT], "timedelta64[ns]"),
        ],
    )
    def test_astype_string_to_extension_dtype_roundtrip(
        self,
        data: List[Any],
        dtype: str,
        request: Any,
        nullable_string_dtype: str,
    ) -> None: ...


class TestAstypeCategorical:
    def test_astype_categorical_to_other(self) -> None: ...
    def test_astype_categorical_invalid_conversions(self) -> None: ...
    def test_astype_categoricaldtype(self) -> None: ...
    @pytest.mark.parametrize("name", [None, "foo"])
    @pytest.mark.parametrize("dtype_ordered", [True, False])
    @pytest.mark.parametrize("series_ordered", [True, False])
    def test_astype_categorical_to_categorical(
        self,
        name: Optional[str],
        dtype_ordered: bool,
        series_ordered: bool,
    ) -> None: ...
    def test_astype_bool_missing_to_categorical(self) -> None: ...
    def test_astype_categories_raises(self) -> None: ...
    @pytest.mark.parametrize("items", [["a", "b", "c", "a"], [1, 2, 3, 1]])
    def test_astype_from_categorical(self, items: List[Any]) -> None: ...
    def test_astype_from_categorical_with_keywords(self) -> None: ...
    def test_astype_timedelta64_with_np_nan(self) -> None: ...
    @td.skip_if_no("pyarrow")
    def test_astype_int_na_string(self) -> None: ...