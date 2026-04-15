import decimal
from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import numpy as np
import pandas as pd
from pandas import ArrowDtype, DataFrame, Index, Series
from pandas._testing import assert_almost_equal

_T = TypeVar("_T")
_ArrayLike = Union[
    Sequence[Union[str, int, float, bool, None]],
    Series,
    Index,
    np.ndarray,
    pd.Timedelta,
    pd.Timestamp,
    str,
    int,
    float,
    bool,
]
_Downcast = Optional[Literal["integer", "signed", "unsigned", "float"]]
_Errors = Optional[Literal["raise", "coerce", "ignore"]]
_DtypeBackend = Literal["numpy_nullable", "pyarrow"]

@pytest.fixture
def errors(request: Any) -> Optional[Literal["raise", "coerce"]]: ...

@pytest.fixture
def signed(request: Any) -> bool: ...

@pytest.fixture
def transform(request: Any) -> Callable[[Union[int, float]], Union[int, float, str]]: ...

@pytest.fixture
def large_val(request: Any) -> int: ...

@pytest.fixture
def multiple_elts(request: Any) -> bool: ...

@pytest.fixture
def transform_assert_equal(
    request: Any,
) -> Tuple[
    Callable[[Sequence[_T]], Union[Index, Series, np.ndarray]],
    Callable[[Any, Any], None],
]: ...

@pytest.mark.parametrize("input_kwargs,result_kwargs")
def test_empty(
    input_kwargs: dict[str, Any], result_kwargs: dict[str, Any]
) -> None: ...

@pytest.mark.parametrize("infer_string", [False, pytest.param(True, marks=td.skip_if_no('pyarrow'))])
@pytest.mark.parametrize("last_val", ["7", 7])
def test_series(last_val: Union[str, int], infer_string: bool) -> None: ...

@pytest.mark.parametrize("data", [[1, 3, 4, 5], [1.0, 3.0, 4.0, 5.0], [True, False, True, True]])
def test_series_numeric(data: Union[list[int], list[float], list[bool]]) -> None: ...

@pytest.mark.parametrize("data,msg")
def test_error(data: list[Union[int, float, str]], msg: str) -> None: ...

def test_ignore_error() -> None: ...

@pytest.mark.parametrize("errors,exp")
def test_bool_handling(
    errors: Literal["raise", "coerce"], exp: Union[str, list[Union[float, None]]]
) -> None: ...

def test_list() -> None: ...

@pytest.mark.parametrize("data,arr_kwargs")
def test_list_numeric(
    data: Union[list[int], list[float], list[bool]],
    arr_kwargs: dict[str, Any],
) -> None: ...

@pytest.mark.parametrize("kwargs", [{"dtype": "O"}, {}])
def test_numeric(kwargs: dict[str, Any]) -> None: ...

@pytest.mark.parametrize("columns", ["a", ["a", "b"]])
def test_numeric_df_columns(columns: Union[str, list[str]]) -> None: ...

@pytest.mark.parametrize("data,exp_data")
def test_numeric_embedded_arr_likes(
    data: list[Any], exp_data: list[Any]
) -> None: ...

def test_all_nan() -> None: ...

def test_type_check(errors: Optional[Literal["raise", "coerce"]]) -> None: ...

@pytest.mark.parametrize("val", [1, 1.1, 20001])
def test_scalar(
    val: Union[int, float], signed: bool, transform: Callable[[Union[int, float]], Union[int, float, str]]
) -> None: ...

def test_really_large_scalar(
    large_val: int,
    signed: bool,
    transform: Callable[[int], Union[int, str]],
    errors: Optional[Literal["raise", "coerce"]],
) -> None: ...

def test_really_large_in_arr(
    large_val: int,
    signed: bool,
    transform: Callable[[int], Union[int, str]],
    multiple_elts: bool,
    errors: Optional[Literal["raise", "coerce"]],
) -> None: ...

def test_really_large_in_arr_consistent(
    large_val: int,
    signed: bool,
    multiple_elts: bool,
    errors: Optional[Literal["raise", "coerce"]],
) -> None: ...

@pytest.mark.parametrize("errors,checker")
def test_scalar_fail(
    errors: Literal["raise", "coerce"],
    checker: Union[str, Callable[[Any], bool]],
) -> None: ...

@pytest.mark.parametrize("data", [[1, 2, 3], [1.0, np.nan, 3, np.nan]])
def test_numeric_dtypes(
    data: Union[list[int], list[Union[float, None]]],
    transform_assert_equal: Tuple[
        Callable[[Sequence[_T]], Union[Index, Series, np.ndarray]],
        Callable[[Any, Any], None],
    ],
) -> None: ...

@pytest.mark.parametrize("data,exp")
def test_str(
    data: list[str],
    exp: np.ndarray,
    transform_assert_equal: Tuple[
        Callable[[Sequence[_T]], Union[Index, Series, np.ndarray]],
        Callable[[Any, Any], None],
    ],
) -> None: ...

def test_datetime_like(
    tz_naive_fixture: Any,
    transform_assert_equal: Tuple[
        Callable[[Sequence[_T]], Union[Index, Series, np.ndarray]],
        Callable[[Any, Any], None],
    ],
) -> None: ...

def test_timedelta(
    transform_assert_equal: Tuple[
        Callable[[Sequence[_T]], Union[Index, Series, np.ndarray]],
        Callable[[Any, Any], None],
    ],
) -> None: ...

@pytest.mark.parametrize("scalar", [pd.Timedelta(1, 'D'), pd.Timestamp('2017-01-01T12'), pd.Timestamp('2017-01-01T12', tz='US/Pacific')])
def test_timedelta_timestamp_scalar(scalar: Union[pd.Timedelta, pd.Timestamp]) -> None: ...

def test_period(
    request: Any,
    transform_assert_equal: Tuple[
        Callable[[Sequence[_T]], Union[Index, Series, np.ndarray]],
        Callable[[Any, Any], None],
    ],
) -> None: ...

@pytest.mark.parametrize("errors,expected")
def test_non_hashable(
    errors: Literal["raise", "coerce"],
    expected: Union[str, Series],
) -> None: ...

def test_downcast_invalid_cast() -> None: ...

def test_errors_invalid_value() -> None: ...

@pytest.mark.parametrize("data", [['1', 2, 3], [1, 2, 3], np.array(['1970-01-02', '1970-01-03', '1970-01-04'], dtype='datetime64[D]')])
@pytest.mark.parametrize("kwargs,exp_dtype")
def test_downcast_basic(
    data: Union[list[Union[str, int]], list[int], np.ndarray],
    kwargs: dict[str, Any],
    exp_dtype: np.dtype,
) -> None: ...

@pytest.mark.parametrize("signed_downcast", ["integer", "signed"])
@pytest.mark.parametrize("data", [['1', 2, 3], [1, 2, 3], np.array(['1970-01-02', '1970-01-03', '1970-01-04'], dtype='datetime64[D]')])
def test_signed_downcast(
    data: Union[list[Union[str, int]], list[int], np.ndarray],
    signed_downcast: Literal["integer", "signed"],
) -> None: ...

def test_ignore_downcast_neg_to_unsigned() -> None: ...

@pytest.mark.parametrize("downcast", ["integer", "signed", "unsigned"])
@pytest.mark.parametrize("data,expected")
def test_ignore_downcast_cannot_convert_float(
    data: list[Union[str, int, float]],
    expected: np.ndarray,
    downcast: Literal["integer", "signed", "unsigned"],
) -> None: ...

@pytest.mark.parametrize("downcast,expected_dtype")
def test_downcast_not8bit(
    downcast: Literal["integer", "signed", "unsigned"],
    expected_dtype: np.dtype,
) -> None: ...

@pytest.mark.parametrize("dtype,downcast,min_max")
def test_downcast_limits(
    dtype: str,
    downcast: Literal["integer", "unsigned"],
    min_max: list[int],
) -> None: ...

def test_downcast_float64_to_float32() -> None: ...

def test_downcast_uint64() -> None: ...

@pytest.mark.parametrize("data,exp_data")
def test_coerce_uint64_conflict(
    data: list[Union[int, str]],
    exp_data: list[Union[int, float, None]],
) -> None: ...

def test_non_coerce_uint64_conflict() -> None: ...

@pytest.mark.parametrize("dc1", ["integer", "float", "unsigned"])
@pytest.mark.parametrize("dc2", ["integer", "float", "unsigned"])
def test_downcast_empty(
    dc1: Literal["integer", "float", "unsigned"],
    dc2: Literal["integer", "float", "unsigned"],
) -> None: ...

def test_failure_to_convert_uint64_string_to_NaN() -> None: ...

@pytest.mark.parametrize("strrep")
def test_precision_float_conversion(strrep: str) -> None: ...

@pytest.mark.parametrize("values, expected")
def test_to_numeric_from_nullable_string(
    values: list[Union[str, int, float, None]],
    nullable_string_dtype: str,
    expected: Series,
) -> None: ...

def test_to_numeric_from_nullable_string_coerce(nullable_string_dtype: str) -> None: ...

@pytest.mark.parametrize("data, input_dtype, downcast, expected_dtype")
def test_downcast_nullable_numeric(
    data: list[Union[int, float, None]],
    input_dtype: str,
    downcast: Literal["integer", "signed", "unsigned", "float"],
    expected_dtype: str,
) -> None: ...

def test_downcast_nullable_mask_is_copied() -> None: ...

def test_to_numeric_scientific_notation() -> None: ...

@pytest.mark.parametrize("val", [9876543210.0, 2.0 ** 128])
def test_to_numeric_large_float_not_downcast_to_float_32(val: float) -> None: ...

@pytest.mark.parametrize("val, dtype")
def test_to_numeric_dtype_backend(
    val: Union[int, float, bool],
    dtype: str,
) -> None: ...

@pytest.mark.parametrize("val, dtype")
def test_to_numeric_dtype_backend_na(
    val: Union[int, float, bool],
    dtype: str,
) -> None: ...

@pytest.mark.parametrize("val, dtype, downcast")
def test_to_numeric_dtype_backend_downcasting(
    val: Union[int, float],
    dtype: str,
    downcast: Literal["integer", "signed", "float"],
) -> None: ...

@pytest.mark.parametrize("smaller, dtype_backend")
def test_to_numeric_dtype_backend_downcasting_uint(
    smaller: str,
    dtype_backend: _DtypeBackend,
) -> None: ...

@pytest.mark.parametrize("dtype")
def test_to_numeric_dtype_backend_already_nullable(dtype: str) -> None: ...

def test_to_numeric_dtype_backend_error(dtype_backend: _DtypeBackend) -> None: ...

def test_invalid_dtype_backend() -> None: ...

def test_coerce_pyarrow_backend() -> None: ...