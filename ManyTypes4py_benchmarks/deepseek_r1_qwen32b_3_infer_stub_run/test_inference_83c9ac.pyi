"""
Stub file for test_inference_83c9ac module
"""

from collections import namedtuple
from collections.abc import Iterator
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from fractions import Fraction
from io import StringIO
from typing import (
    Any,
    AnyStr,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    ParamFixture,
    Set,
    Tuple,
    Union,
)
import numpy as np
import pytest
from pandas._libs import lib, missing as libmissing, ops as libops
from pandas.compat.numpy import np_version_gt2
from pandas.core.dtypes import inference
from pandas.core.dtypes.cast import find_result_type
from pandas.core.dtypes.common import (
    ensure_int32,
    is_bool,
    is_complex,
    is_datetime64_any_dtype,
    is_datetime64_dtype,
    is_datetime64_ns_dtype,
    is_datetime64tz_dtype,
    is_float,
    is_integer,
    is_number,
    is_scalar,
    is_scipy_sparse,
    is_timedelta64_dtype,
    is_timedelta64_ns_dtype,
)
import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    DateOffset,
    DatetimeIndex,
    Index,
    Interval,
    Period,
    Series,
    Timedelta,
    TimedeltaIndex,
    Timestamp,
)
import pandas._testing as tm
from pandas.core.arrays import (
    BooleanArray,
    FloatingArray,
    IntegerArray,
)

ll_params: List[Tuple[List[Any], bool, str]] = ...

objs: Tuple[Any, ...] = ...
expected: Tuple[bool, ...] = ...
ids: Tuple[str, ...] = ...

@pytest.fixture
def coerce() -> bool:
    ...

@pytest.fixture
def maybe_list_like() -> Tuple[Any, bool]:
    ...

def test_is_list_like(maybe_list_like: Tuple[Any, bool]) -> None:
    ...

def test_is_list_like_disallow_sets(maybe_list_like: Tuple[Any, bool]) -> None:
    ...

def test_is_list_like_recursion() -> None:
    ...

def test_is_list_like_iter_is_none() -> None:
    ...

def test_is_list_like_generic() -> None:
    ...

def test_is_sequence() -> None:
    ...

def test_is_array_like() -> None:
    ...

@pytest.mark.parametrize('inner', [List[Any], Tuple[Any, ...], Dict[str, Any], Set[Any], Series, collections.defaultdict])
@pytest.mark.parametrize('outer', [list, Series, np.array, tuple])
def test_is_nested_list_like_passes(inner: Any, outer: Any) -> None:
    ...

@pytest.mark.parametrize('obj', [str, List[Any], Tuple[Any, ...], Dict[str, Any], Set[Any], Series, DataFrame, Generator[Any, Any, Any]])
def test_is_nested_list_like_fails(obj: Any) -> None:
    ...

@pytest.mark.parametrize('ll', [Dict[str, Any], collections.defaultdict])
def test_is_dict_like_passes(ll: Any) -> None:
    ...

@pytest.mark.parametrize('ll', [str, int, List[Any], Tuple[Any, ...], Index, dict, collections.defaultdict, Series])
def test_is_dict_like_fails(ll: Any) -> None:
    ...

@pytest.mark.parametrize('has_keys', [True, False])
@pytest.mark.parametrize('has_getitem', [True, False])
@pytest.mark.parametrize('has_contains', [True, False])
def test_is_dict_like_duck_type(has_keys: bool, has_getitem: bool, has_contains: bool) -> None:
    ...

def test_is_file_like() -> None:
    ...

@pytest.mark.parametrize('ll', [test_tuple(1, 2, 3)])
def test_is_names_tuple_passes(ll: Any) -> None:
    ...

@pytest.mark.parametrize('ll', [Tuple[Any, ...], str, Series])
def test_is_names_tuple_fails(ll: Any) -> None:
    ...

def test_is_hashable() -> None:
    ...

@pytest.mark.parametrize('ll', [re.compile('ad')])
def test_is_re_passes(ll: Any) -> None:
    ...

@pytest.mark.parametrize('ll', [str, int, object])
def test_is_re_fails(ll: Any) -> None:
    ...

@pytest.mark.parametrize('ll', [str, re.compile('adsf'), str])
def test_is_recompilable_passes(ll: Any) -> None:
    ...

@pytest.mark.parametrize('ll', [int, List[Any], object])
def test_is_recompilable_fails(ll: Any) -> None:
    ...

class TestInference:
    ...

class TestTypeInference:
    ...

class TestNumberScalar:
    ...

def test_nan_to_nat_conversions() -> None:
    ...

@pytest.mark.filterwarnings('ignore::PendingDeprecationWarning')
@pytest.mark.parametrize('spmatrix', ['bsr', 'coo', 'csc', 'csr', 'dia', 'dok', 'lil'])
def test_is_scipy_sparse(spmatrix: str) -> None:
    ...

def test_ensure_int32() -> None:
    ...

@pytest.mark.parametrize('right,result', [(0, np.uint8), (-1, np.int16), (300, np.uint16), (300.0, np.uint16), (300.1, np.float64), (np.int16(300), np.int16 if np_version_gt2 else np.uint16)])
def test_find_result_type_uint_int(right: Any, result: Any) -> None:
    ...

@pytest.mark.parametrize('right,result', [(300.0, np.float64), (np.float32(300), np.float32)])
def test_find_result_type_floats(right: Any, result: Any) -> None:
    ...