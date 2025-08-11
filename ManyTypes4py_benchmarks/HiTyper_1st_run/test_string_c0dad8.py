"""
This file contains a minimal set of tests for compliance with the extension
array interface test suite, and should contain no other tests.
The test suite for the full functionality of the array is located in
`pandas/tests/arrays/`.

The tests in this file are inherited from the BaseExtensionTests, and only
minimal tweaks should be applied to get the tests passing (by overwriting a
parent method).

Additional tests should either be added to one of the BaseExtensionTests
classes (if they are relevant for the extension interface for all dtypes), or
be added to the array-specific tests in `pandas/tests/arrays/`.

"""
from __future__ import annotations
import string
from typing import cast
import numpy as np
import pytest
from pandas.compat import HAS_PYARROW
from pandas.core.dtypes.base import StorageExtensionDtype
import pandas as pd
import pandas._testing as tm
from pandas.api.types import is_string_dtype
from pandas.core.arrays import ArrowStringArray
from pandas.core.arrays.string_ import StringDtype
from pandas.tests.extension import base

def maybe_split_array(arr: Union[list, numpy.ndarray], chunked: Union[list, list[int], numpy.ndarray]) -> Union[list, numpy.ndarray, typing.Sequence[int], numpy.array]:
    if not chunked:
        return arr
    elif arr.dtype.storage != 'pyarrow':
        return arr
    pa = pytest.importorskip('pyarrow')
    arrow_array = arr._pa_array
    split = len(arrow_array) // 2
    arrow_array = pa.chunked_array([*arrow_array[:split].chunks, *arrow_array[split:].chunks])
    assert arrow_array.num_chunks == 2
    return type(arr)(arrow_array)

@pytest.fixture(params=[True, False])
def chunked(request: Any):
    return request.param

@pytest.fixture
def dtype(string_dtype_arguments: Union[bool, str, None, typing.Iterable[typing.Hashable]]) -> StringDtype:
    storage, na_value = string_dtype_arguments
    return StringDtype(storage=storage, na_value=na_value)

@pytest.fixture
def data(dtype: Union[bool, static_frame.core.util.DtypeSpecifier, typing.Type], chunked: Union[bool, numpy.dtype, None]) -> Union[typing.Sequence[str], list[str], list]:
    strings = np.random.default_rng(2).choice(list(string.ascii_letters), size=100)
    while strings[0] == strings[1]:
        strings = np.random.default_rng(2).choice(list(string.ascii_letters), size=100)
    arr = dtype.construct_array_type()._from_sequence(strings, dtype=dtype)
    return maybe_split_array(arr, chunked)

@pytest.fixture
def data_missing(dtype: Union[bool, numpy.dtype, static_frame.core.util.UFunc], chunked: Union[bool, numpy.dtype, static_frame.core.util.UFunc]) -> Union[list[str], typing.Sequence[str], list]:
    """Length 2 array with [NA, Valid]"""
    arr = dtype.construct_array_type()._from_sequence([pd.NA, 'A'], dtype=dtype)
    return maybe_split_array(arr, chunked)

@pytest.fixture
def data_for_sorting(dtype: Union[bool, typing.Callable, str], chunked: Union[bool, static_frame.core.util.DtypeSpecifier, numpy.ndarray]) -> Union[list[str], list[int], types.CompilationTarget]:
    arr = dtype.construct_array_type()._from_sequence(['B', 'C', 'A'], dtype=dtype)
    return maybe_split_array(arr, chunked)

@pytest.fixture
def data_missing_for_sorting(dtype: Union[bool, static_frame.core.util.UFunc, typing.Iterable[typing.Any]], chunked: Union[bool, static_frame.core.util.DtypeSpecifier, starfish.types.Number]) -> Union[list[str], int, types.CompilationTarget]:
    arr = dtype.construct_array_type()._from_sequence(['B', pd.NA, 'A'], dtype=dtype)
    return maybe_split_array(arr, chunked)

@pytest.fixture
def data_for_grouping(dtype: Union[bool, static_frame.core.util.DtypeSpecifier, typing.Type], chunked: Union[bool, static_frame.core.util.DtypeSpecifier, starfish.types.Number]) -> Union[list[str], typing.IO, list]:
    arr = dtype.construct_array_type()._from_sequence(['B', 'B', pd.NA, pd.NA, 'A', 'A', 'B', 'C'], dtype=dtype)
    return maybe_split_array(arr, chunked)

class TestStringArray(base.ExtensionTests):

    def test_eq_with_str(self, dtype: Union[bool, str, static_frame.core.util.DtypeSpecifier]) -> None:
        super().test_eq_with_str(dtype)
        if dtype.na_value is pd.NA:
            assert dtype == f'string[{dtype.storage}]'
        elif dtype.storage == 'pyarrow':
            with tm.assert_produces_warning(FutureWarning):
                assert dtype == 'string[pyarrow_numpy]'

    def test_is_not_string_type(self, dtype: Union[int, typing.Callable[typing.Mapping, T], static_frame.core.util.DtypeSpecifier]) -> None:
        assert is_string_dtype(dtype)

    def test_is_dtype_from_name(self, dtype: Union[bool, str], using_infer_string: bool) -> None:
        if dtype.na_value is np.nan and (not using_infer_string):
            result = type(dtype).is_dtype(dtype.name)
            assert result is False
        else:
            super().test_is_dtype_from_name(dtype)

    def test_construct_from_string_own_name(self, dtype: Union[bool, typing.Callable, static_frame.core.util.UFunc], using_infer_string: bool) -> None:
        if dtype.na_value is np.nan and (not using_infer_string):
            with pytest.raises(TypeError, match="Cannot construct a 'StringDtype'"):
                dtype.construct_from_string(dtype.name)
        else:
            super().test_construct_from_string_own_name(dtype)

    def test_view(self, data: Union[dict, dict[str, typing.Any], bytes]) -> None:
        if data.dtype.storage == 'pyarrow':
            pytest.skip(reason='2D support not implemented for ArrowStringArray')
        super().test_view(data)

    def test_from_dtype(self, data: Union[dict, bytes, list]) -> None:
        pass

    def test_transpose(self, data: Union[dict, pandas.DataFrame]) -> None:
        if data.dtype.storage == 'pyarrow':
            pytest.skip(reason='2D support not implemented for ArrowStringArray')
        super().test_transpose(data)

    def test_setitem_preserves_views(self, data: Union[dict, bytes, dict[str, typing.Any]]) -> None:
        if data.dtype.storage == 'pyarrow':
            pytest.skip(reason='2D support not implemented for ArrowStringArray')
        super().test_setitem_preserves_views(data)

    def test_dropna_array(self, data_missing: pandas.DataFrame) -> None:
        result = data_missing.dropna()
        expected = data_missing[[1]]
        tm.assert_extension_array_equal(result, expected)

    def test_fillna_no_op_returns_copy(self, data: Union[dict, T, list[list[typing.Any]]]) -> None:
        data = data[~data.isna()]
        valid = data[0]
        result = data.fillna(valid)
        assert result is not data
        tm.assert_extension_array_equal(result, data)

    def _get_expected_exception(self, op_name: Union[str, bool, typing.Callable], obj: Union[str, bool, typing.Callable], other: Union[str, bool, typing.Callable]) -> Union[TypeError, None]:
        if op_name in ['__mod__', '__rmod__', '__divmod__', '__rdivmod__', '__pow__', '__rpow__']:
            return TypeError
        elif op_name in ['__mul__', '__rmul__']:
            return TypeError
        elif op_name in ['__truediv__', '__rtruediv__', '__floordiv__', '__rfloordiv__', '__sub__', '__rsub__']:
            return TypeError
        return None

    def _supports_reduction(self, ser: Union[str, list, list[float]], op_name: Union[str, list, list[float]]) -> bool:
        return op_name in ['min', 'max', 'sum'] or (ser.dtype.na_value is np.nan and op_name in ('any', 'all'))

    def _supports_accumulation(self, ser: Union[str, typing.Hashable, pandas.core.arrays.datetimes.DatetimeArray], op_name: Union[str, typing.AnyStr, list]) -> bool:
        assert isinstance(ser.dtype, StorageExtensionDtype)
        return ser.dtype.storage == 'pyarrow' and op_name in ['cummin', 'cummax', 'cumsum']

    def _cast_pointwise_result(self, op_name: Union[str, bool, BaseException, None], obj: Union[str, list, typing.Callable[object, typing.Any], None], other: Union[str, bool, BaseException, None], pointwise_result: Union[static_frame.core.util.DtypeSpecifier, typing.Iterable[str], T]) -> Union[str, typing.IO, int]:
        dtype = cast(StringDtype, tm.get_dtype(obj))
        if op_name in ['__add__', '__radd__']:
            cast_to = dtype
        elif dtype.na_value is np.nan:
            cast_to = np.bool_
        elif dtype.storage == 'pyarrow':
            cast_to = 'boolean[pyarrow]'
        else:
            cast_to = 'boolean'
        return pointwise_result.astype(cast_to)

    def test_compare_scalar(self, data: Union[pandas.DataFrame, typing.Iterable[typing.Any]], comparison_op: Union[typing.Callable, typing.Iterable[typing.Any]]) -> None:
        ser = pd.Series(data)
        self._compare_other(ser, data, comparison_op, 'abc')

    def test_groupby_extension_apply(self, data_for_grouping: Union[bool, str, None, numpy.dtype], groupby_apply_op: Union[bool, str, None, numpy.dtype]) -> None:
        super().test_groupby_extension_apply(data_for_grouping, groupby_apply_op)

    def test_combine_add(self, data_repeated: bool, using_infer_string: Union[list, bool], request: bool) -> None:
        dtype = next(data_repeated(1)).dtype
        if using_infer_string and (dtype.na_value is pd.NA and dtype.storage == 'python'):
            mark = pytest.mark.xfail(reason='The pointwise operation result will be inferred to string[nan, pyarrow], which does not match the input dtype')
            request.applymarker(mark)
        super().test_combine_add(data_repeated)

    def test_arith_series_with_array(self, data: Union[bool, tuple, dict], all_arithmetic_operators: Union[bool, list[typing.Optional[mypy.types.Type]], dict], using_infer_string: Union[bool, dict, tuple[typing.Union[bool,str]]], request: Union[bool, dict[str, object], list[typing.Optional[mypy.types.Type]]]) -> None:
        dtype = data.dtype
        if using_infer_string and all_arithmetic_operators == '__radd__' and (dtype.na_value is pd.NA or (dtype.storage == 'python' and HAS_PYARROW)):
            mark = pytest.mark.xfail(reason='The pointwise operation result will be inferred to string[nan, pyarrow], which does not match the input dtype')
            request.applymarker(mark)
        super().test_arith_series_with_array(data, all_arithmetic_operators)

class Test2DCompat(base.Dim2CompatTests):

    @pytest.fixture(autouse=True)
    def arrow_not_supported(self, data: Union[str, list[list[str]], bytes]) -> None:
        if isinstance(data, ArrowStringArray):
            pytest.skip(reason='2D support not implemented for ArrowStringArray')

def test_searchsorted_with_na_raises(data_for_sorting: Union[numpy.ndarray, list[int]], as_series: Union[bool, pandas.DataFrame]) -> None:
    b, c, a = data_for_sorting
    arr = data_for_sorting.take([2, 0, 1])
    arr[-1] = pd.NA
    if as_series:
        arr = pd.Series(arr)
    msg = 'searchsorted requires array to be sorted, which is impossible with NAs present.'
    with pytest.raises(ValueError, match=msg):
        arr.searchsorted(b)