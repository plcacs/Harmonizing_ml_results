from decimal import Decimal
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    Iterable,
    Sequence,
    Type,
    TYPE_CHECKING,
)
import numpy as np
import pandas as pd
from pandas import (
    DataFrame,
    Index,
    Series,
    ArrowDtype,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float32,
    Float64,
    boolean,
    Int64Dtype,
    UInt64Dtype,
    Float64Dtype,
    BooleanDtype,
)
import pytest
from _pytest.fixtures import SubRequest

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
else:
    ArrayLike = Any


@pytest.fixture
def errors(request: SubRequest) -> Optional[str]:
    ...


@pytest.fixture
def signed(request: SubRequest) -> bool:
    ...


@pytest.fixture
def transform(request: SubRequest) -> Callable[[Any], Any]:
    ...


@pytest.fixture
def large_val(request: SubRequest) -> int:
    ...


@pytest.fixture
def multiple_elts(request: SubRequest) -> bool:
    ...


@pytest.fixture
def transform_assert_equal(request: SubRequest) -> Tuple[Callable[[Any], Any], Callable[[Any, Any], None]]:
    ...


def test_empty(input_kwargs: Dict[str, Any], result_kwargs: Dict[str, Any]) -> None:
    ...


def test_series(last_val: Union[str, int], infer_string: bool) -> None:
    ...


def test_series_numeric(data: List[Union[int, float, bool]]) -> None:
    ...


def test_error(data: List[Union[str, int, float]], msg: str) -> None:
    ...


def test_ignore_error() -> None:
    ...


def test_bool_handling(errors: str, exp: Union[str, List[Union[float, np.nan]]]) -> None:
    ...


def test_list() -> None:
    ...


def test_list_numeric(data: List[Union[int, float, bool]], arr_kwargs: Dict[str, Any]) -> None:
    ...


def test_numeric(kwargs: Dict[str, Any]) -> None:
    ...


def test_numeric_df_columns(columns: Union[str, List[str]]) -> None:
    ...


def test_numeric_embedded_arr_likes(data: List[Any], exp_data: List[Any]) -> None:
    ...


def test_all_nan() -> None:
    ...


def test_type_check(errors: Optional[str]) -> None:
    ...


def test_scalar(val: Union[int, float], signed: bool, transform: Callable[[Any], Any]) -> None:
    ...


def test_really_large_scalar(large_val: int, signed: bool, transform: Callable[[Any], Any], errors: Optional[str]) -> None:
    ...


def test_really_large_in_arr(large_val: int, signed: bool, transform: Callable[[Any], Any], multiple_elts: bool, errors: Optional[str]) -> None:
    ...


def test_really_large_in_arr_consistent(large_val: int, signed: bool, multiple_elts: bool, errors: Optional[str]) -> None:
    ...


def test_scalar_fail(errors: str, checker: Union[str, Callable[[Any], bool]]) -> None:
    ...


def test_numeric_dtypes(data: List[Any], transform_assert_equal: Tuple[Callable[[Any], Any], Callable[[Any, Any], None]]) -> None:
    ...


def test_str(data: List[str], exp: np.ndarray, transform_assert_equal: Tuple[Callable[[Any], Any], Callable[[Any, Any], None]]) -> None:
    ...


def test_datetime_like(tz_naive_fixture: str, transform_assert_equal: Tuple[Callable[[Any], Any], Callable[[Any, Any], None]]) -> None:
    ...


def test_timedelta(transform_assert_equal: Tuple[Callable[[Any], Any], Callable[[Any, Any], None]]) -> None:
    ...


def test_timedelta_timestamp_scalar(scalar: Union[pd.Timestamp, pd.Timedelta]) -> None:
    ...


def test_period(request: SubRequest, transform_assert_equal: Tuple[Callable[[Any], Any], Callable[[Any, Any], None]]) -> None:
    ...


def test_non_hashable(errors: str, expected: Union[str, Series]) -> None:
    ...


def test_downcast_invalid_cast() -> None:
    ...


def test_errors_invalid_value() -> None:
    ...


def test_downcast_basic(data: List[Any], kwargs: Dict[str, Any], exp_dtype: Type) -> None:
    ...


def test_signed_downcast(data: List[Any], signed_downcast: str) -> None:
    ...


def test_ignore_downcast_neg_to_unsigned(data: List[str]) -> None:
    ...


def test_ignore_downcast_cannot_convert_float(data: List[Any], expected: np.ndarray, downcast: str) -> None:
    ...


def test_downcast_not8bit(downcast: str, expected_dtype: Type) -> None:
    ...


def test_downcast_limits(dtype: str, downcast: str, min_max: List[int]) -> None:
    ...


def test_downcast_float64_to_float32() -> None:
    ...


def test_downcast_uint64() -> None:
    ...


def test_coerce_uint64_conflict(data: List[Any], exp_data: List[Any]) -> None:
    ...


def test_non_coerce_uint64_conflict() -> None:
    ...


def test_downcast_empty(dc1: str, dc2: str) -> None:
    ...


def test_failure_to_convert_uint64_string_to_NaN() -> None:
    ...


def test_precision_float_conversion(strrep: str) -> None:
    ...


def test_to_numeric_from_nullable_string(values: List[Any], nullable_string_dtype: str, expected: Series) -> None:
    ...


def test_to_numeric_from_nullable_string_coerce(values: List[Any], nullable_string_dtype: str) -> None:
    ...


def test_downcast_nullable_numeric(data: List[Any], input_dtype: str, downcast: str, expected_dtype: str) -> None:
    ...


def test_downcast_nullable_mask_is_copied() -> None:
    ...


def test_to_numeric_scientific_notation() -> None:
    ...


def test_to_numeric_large_float_not_downcast_to_float_32(val: float) -> None:
    ...


def test_to_numeric_dtype_backend(val: Union[int, float, bool], dtype: str) -> None:
    ...


def test_to_numeric_dtype_backend_na(val: Union[int, float, bool], dtype: str) -> None:
    ...


def test_to_numeric_dtype_backend_downcasting(val: Union[int, float], dtype: str, downcast: str) -> None:
    ...


def test_to_numeric_dtype_backend_downcasting_uint(smaller: str, dtype_backend: str) -> None:
    ...


def test_to_numeric_dtype_backend_already_nullable(dtype: str) -> None:
    ...


def test_to_numeric_dtype_backend_error(dtype_backend: str) -> None:
    ...


def test_invalid_dtype_backend() -> None:
    ...


def test_coerce_pyarrow_backend() -> None:
    ...