from __future__ import annotations
from datetime import datetime, timedelta, timezone
import numpy as np
import pytest
from pandas import (
    Categorical,
    DataFrame,
    DatetimeIndex,
    NaT,
    Period,
    Series,
    Timedelta,
    Timestamp,
    date_range,
    isna,
    timedelta_range,
)
import pandas._testing as tm
from pandas.core.arrays import period_array
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Self,
    Tuple,
    Union,
)

class TestSeriesFillNA:
    def test_fillna_nat(self: Self) -> None:
        ...

    def test_fillna(self: Self) -> None:
        ...

    def test_fillna_nonscalar(self: Self) -> None:
        ...

    def test_fillna_aligns(self: Self) -> None:
        ...

    def test_fillna_limit(self: Self) -> None:
        ...

    def test_fillna_dont_cast_strings(self: Self) -> None:
        ...

    def test_fillna_consistency(self: Self) -> None:
        ...

    def test_timedelta_fillna(
        self: Self,
        frame_or_series: Any,
        unit: str = ...,
    ) -> None:
        ...

    def test_datetime64_fillna(self: Self) -> None:
        ...

    @pytest.mark.parametrize("scalar", [False, True])
    @pytest.mark.parametrize("tz", [None, "UTC"])
    def test_datetime64_fillna_mismatched_reso_no_rounding(
        self: Self,
        tz: Optional[str],
        scalar: bool,
    ) -> None:
        ...

    @pytest.mark.parametrize("scalar", [False, True])
    def test_timedelta64_fillna_mismatched_reso_no_rounding(
        self: Self,
        scalar: bool,
    ) -> None:
        ...

    def test_datetime64_fillna_backfill(self: Self) -> None:
        ...

    @pytest.mark.parametrize("tz", ["US/Eastern", "Asia/Tokyo"])
    def test_datetime64_tz_fillna(
        self: Self,
        tz: str,
        unit: str = ...,
    ) -> None:
        ...

    def test_fillna_dt64_timestamp(self: Self, frame_or_series: Any) -> None:
        ...

    def test_fillna_dt64_non_nao(self: Self) -> None:
        ...

    def test_fillna_numeric_inplace(self: Self) -> None:
        ...

    @pytest.mark.parametrize(
        "fill_value, expected_output",
        [
            ("a", ["a", "a", "b", "a", "a"]),
            ({1: "a", 3: "b", 4: "b"}, ["a", "a", "b", "b", "b"]),
            ({1: "a"}, ["a", "a", "b", np.nan, np.nan]),
            ({1: "a", 3: "b"}, ["a", "a", "b", "b", np.nan]),
            (Series("a"), ["a", np.nan, "b", np.nan, np.nan]),
            (Series("a", index=[1]), ["a", "a", "b", np.nan, np.nan]),
            (Series({1: "a", 3: "b"}), ["a", "a", "b", "b", np.nan]),
            (Series(["a", "b"], index=[3, 4]), ["a", np.nan, "b", "a", "b"]),
        ],
    )
    def test_fillna_categorical(
        self: Self,
        fill_value: Union[str, Dict[int, str], Series],
        expected_output: List[Union[str, float]],
    ) -> None:
        ...

    @pytest.mark.parametrize(
        "fill_value, expected_output",
        [
            (["a", "b", "c", "d", "e"], ["a", "b", "b", "d", "e"]),
            (["b", "d", "a", "d", "a"], ["a", "d", "b", "d", "a"]),
            (
                Categorical(
                    ["b", "d", "a", "d", "a"],
                    categories=["b", "c", "d", "e", "a"],
                ),
                ["a", "d", "b", "d", "a"],
            ),
        ],
    )
    def test_fillna_categorical_with_new_categories(
        self: Self,
        fill_value: Union[List[str], Categorical],
        expected_output: List[str],
    ) -> None:
        ...

    def test_fillna_categorical_raises(self: Self) -> None:
        ...

    @pytest.mark.parametrize("dtype", [float, "float32", "float64"])
    @pytest.mark.parametrize("scalar", [True, False])
    def test_fillna_float_casting(
        self: Self,
        dtype: Union[type, str],
        any_real_numpy_dtype: Any,
        scalar: bool,
    ) -> None:
        ...

    def test_fillna_f32_upcast_with_dict(self: Self) -> None:
        ...

    def test_fillna_listlike_invalid(self: Self) -> None:
        ...

    def test_fillna_method_and_limit_invalid(self: Self) -> None:
        ...

    def test_fillna_datetime64_with_timezone_tzinfo(self: Self) -> None:
        ...

    @pytest.mark.parametrize(
        "input, input_fillna, expected_data, expected_categories",
        [
            (["A", "B", None, "A"], "B", ["A", "B", "B", "A"], ["A", "B"]),
            (["A", "B", np.nan, "A"], "B", ["A", "B", "B", "A"], ["A", "B"]),
        ],
    )
    def test_fillna_categorical_accept_same_type(
        self: Self,
        input: List[Union[str, float]],
        input_fillna: str,
        expected_data: List[str],
        expected_categories: List[str],
    ) -> None:
        ...

class TestFillnaPad:
    def test_fillna_bug(self: Self) -> None:
        ...

    def test_ffill_mixed_dtypes_without_missing_data(self: Self) -> None:
        ...

    def test_pad_nan(self: Self) -> None:
        ...

    def test_series_fillna_limit(self: Self) -> None:
        ...

    def test_series_pad_backfill_limit(self: Self) -> None:
        ...

    def test_fillna_int(self: Self) -> None:
        ...

    def test_datetime64tz_fillna_round_issue(self: Self) -> None:
        ...

    def test_fillna_parr(self: Self) -> None:
        ...

@pytest.mark.parametrize(
    "data, expected_data, method, kwargs",
    [
        (
            [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [np.nan, np.nan, 3.0, 3.0, 3.0, 3.0, 7.0, np.nan, np.nan],
            "ffill",
            {"limit_area": "inside"},
        ),
        (
            [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [np.nan, np.nan, 3.0, 3.0, np.nan, np.nan, 7.0, np.nan, np.nan],
            "ffill",
            {"limit_area": "inside", "limit": 1},
        ),
        (
            [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [np.nan, np.nan, 3.0, np.nan, np.nan, np.nan, 7.0, 7.0, 7.0],
            "ffill",
            {"limit_area": "outside"},
        ),
        (
            [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [np.nan, np.nan, 3.0, np.nan, np.nan, np.nan, 7.0, 7.0, np.nan],
            "ffill",
            {"limit_area": "outside", "limit": 1},
        ),
        (
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            "ffill",
            {"limit_area": "outside", "limit": 1},
        ),
        (range(5), range(5), "ffill", {"limit_area": "outside", "limit": 1}),
        (
            [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [np.nan, np.nan, 3.0, 7.0, 7.0, 7.0, 7.0, np.nan, np.nan],
            "bfill",
            {"limit_area": "inside"},
        ),
        (
            [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [np.nan, np.nan, 3.0, np.nan, np.nan, 7.0, 7.0, np.nan, np.nan],
            "bfill",
            {"limit_area": "inside", "limit": 1},
        ),
        (
            [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [3.0, 3.0, 3.0, np.nan, np.nan, np.nan, 7.0, np.nan, np.nan],
            "bfill",
            {"limit_area": "outside"},
        ),
        (
            [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [np.nan, 3.0, 3.0, np.nan, np.nan, np.nan, 7.0, np.nan, np.nan],
            "bfill",
            {"limit_area": "outside", "limit": 1},
        ),
    ],
)
def test_ffill_bfill_limit_area(
    data: List[Union[float, int]],
    expected_data: List[Union[float, int]],
    method: str,
    kwargs: Dict[str, Union[str, int]],
) -> None:
    ...