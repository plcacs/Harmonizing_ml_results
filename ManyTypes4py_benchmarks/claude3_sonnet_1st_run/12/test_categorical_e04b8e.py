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
import string
from typing import Any, Callable, List, Optional, Sequence, Union

import numpy as np
import pytest
from pandas._config import using_string_dtype
import pandas as pd
from pandas import Categorical
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.tests.extension import base


def make_data() -> np.ndarray:
    while True:
        values = np.random.default_rng(2).choice(list(string.ascii_letters), size=100)
        if values[0] != values[1]:
            break
    return values


@pytest.fixture
def dtype() -> CategoricalDtype:
    return CategoricalDtype()


@pytest.fixture
def data() -> Categorical:
    """Length-100 array for this type.

    * data[0] and data[1] should both be non missing
    * data[0] and data[1] should not be equal
    """
    return Categorical(make_data())


@pytest.fixture
def data_missing() -> Categorical:
    """Length 2 array with [NA, Valid]"""
    return Categorical([np.nan, 'A'])


@pytest.fixture
def data_for_sorting() -> Categorical:
    return Categorical(['A', 'B', 'C'], categories=['C', 'A', 'B'], ordered=True)


@pytest.fixture
def data_missing_for_sorting() -> Categorical:
    return Categorical(['A', None, 'B'], categories=['B', 'A'], ordered=True)


@pytest.fixture
def data_for_grouping() -> Categorical:
    return Categorical(['a', 'a', None, None, 'b', 'b', 'a', 'c'])


class TestCategorical(base.ExtensionTests):

    def test_contains(self, data: Categorical, data_missing: Categorical) -> None:
        na_value = data.dtype.na_value
        data = data[~data.isna()]
        assert data[0] in data
        assert data_missing[0] in data_missing
        assert na_value in data_missing
        assert na_value not in data
        for na_value_obj in tm.NULL_OBJECTS:
            if na_value_obj is na_value:
                continue
            assert na_value_obj not in data
            if not using_string_dtype():
                assert na_value_obj in data_missing

    def test_empty(self, dtype: CategoricalDtype) -> None:
        cls = dtype.construct_array_type()
        result = cls._empty((4,), dtype=dtype)
        assert isinstance(result, cls)
        assert result.dtype == CategoricalDtype([])

    @pytest.mark.skip(reason='Backwards compatibility')
    def test_getitem_scalar(self, data: Categorical) -> None:
        super().test_getitem_scalar(data)

    def test_combine_add(self, data_repeated: Callable) -> None:
        orig_data1, orig_data2 = data_repeated(2)
        s1 = pd.Series(orig_data1)
        s2 = pd.Series(orig_data2)
        result = s1.combine(s2, lambda x1, x2: x1 + x2)
        expected = pd.Series([a + b for a, b in zip(list(orig_data1), list(orig_data2))])
        tm.assert_series_equal(result, expected)
        val = s1.iloc[0]
        result = s1.combine(val, lambda x1, x2: x1 + x2)
        expected = pd.Series([a + val for a in list(orig_data1)])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('na_action', [None, 'ignore'])
    def test_map(self, data: Categorical, na_action: Optional[str]) -> None:
        result = data.map(lambda x: x, na_action=na_action)
        tm.assert_extension_array_equal(result, data)

    def test_arith_frame_with_scalar(self, data: Categorical, all_arithmetic_operators: str, request: Any) -> None:
        op_name = all_arithmetic_operators
        if op_name == '__rmod__':
            request.applymarker(pytest.mark.xfail(reason='rmod never called when string is first argument'))
        super().test_arith_frame_with_scalar(data, op_name)

    def test_arith_series_with_scalar(self, data: Categorical, all_arithmetic_operators: str, request: Any) -> None:
        op_name = all_arithmetic_operators
        if op_name == '__rmod__':
            request.applymarker(pytest.mark.xfail(reason='rmod never called when string is first argument'))
        super().test_arith_series_with_scalar(data, op_name)

    def _compare_other(self, ser: pd.Series, data: Categorical, op: Callable, other: Any) -> Optional[pd.Series]:
        op_name = f'__{op.__name__}__'
        if op_name not in ['__eq__', '__ne__']:
            msg = 'Unordered Categoricals can only compare equality or not'
            with pytest.raises(TypeError, match=msg):
                op(data, other)
        else:
            return super()._compare_other(ser, data, op, other)

    @pytest.mark.xfail(reason='Categorical overrides __repr__')
    @pytest.mark.parametrize('size', ['big', 'small'])
    def test_array_repr(self, data: Categorical, size: str) -> None:
        super().test_array_repr(data, size)

    @pytest.mark.xfail(reason='TBD')
    @pytest.mark.parametrize('as_index', [True, False])
    def test_groupby_extension_agg(self, as_index: bool, data_for_grouping: Categorical) -> None:
        super().test_groupby_extension_agg(as_index, data_for_grouping)


class Test2DCompat(base.NDArrayBacked2DTests):

    def test_repr_2d(self, data: Categorical) -> None:
        res = repr(data.reshape(1, -1))
        assert res.count('\nCategories') == 1
        res = repr(data.reshape(-1, 1))
        assert res.count('\nCategories') == 1
