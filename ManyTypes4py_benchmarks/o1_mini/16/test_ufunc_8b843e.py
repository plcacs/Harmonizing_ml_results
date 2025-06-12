from collections import deque
import re
import string
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.arrays import SparseArray
from typing import Callable, Tuple, Any, Union, Optional
from pandas.core.arrays import ExtensionArray
from pandas.core.indexes.base import Index


@pytest.fixture(params=[np.add, np.logaddexp])
def ufunc(request: pytest.FixtureRequest) -> Callable[..., Any]:
    return request.param


@pytest.fixture(
    params=[
        pytest.param(True, marks=pytest.mark.fails_arm_wheels),
        False
    ],
    ids=['sparse', 'dense']
)
def sparse(request: pytest.FixtureRequest) -> bool:
    return request.param


@pytest.fixture
def arrays_for_binary_ufunc() -> Tuple[np.ndarray, np.ndarray]:
    """
    A pair of random, length-100 integer-dtype arrays, that are mostly 0.
    """
    a1: np.ndarray = np.random.default_rng(2).integers(0, 10, 100, dtype='int64')
    a2: np.ndarray = np.random.default_rng(2).integers(0, 10, 100, dtype='int64')
    a1[::3] = 0
    a2[::4] = 0
    return (a1, a2)


@pytest.mark.parametrize('ufunc', [np.positive, np.floor, np.exp])
def test_unary_ufunc(ufunc: Callable[[Any], Any], sparse: bool) -> None:
    arr: np.ndarray = np.random.default_rng(2).integers(0, 10, 10, dtype='int64')
    arr[::2] = 0
    if sparse:
        arr = SparseArray(arr, dtype=pd.SparseDtype('int64', 0))
    index: list[str] = list(string.ascii_letters[:10])
    name: str = 'name'
    series: pd.Series = pd.Series(arr, index=index, name=name)
    result: pd.Series = ufunc(series)
    expected: pd.Series = pd.Series(ufunc(arr), index=index, name=name)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize('flip', [True, False], ids=['flipped', 'straight'])
def test_binary_ufunc_with_array(
    flip: bool,
    sparse: bool,
    ufunc: Callable[..., Any],
    arrays_for_binary_ufunc: Tuple[np.ndarray, np.ndarray]
) -> None:
    a1, a2 = arrays_for_binary_ufunc
    if sparse:
        a1 = SparseArray(a1, dtype=pd.SparseDtype('int64', 0))
        a2 = SparseArray(a2, dtype=pd.SparseDtype('int64', 0))
    name: str = 'name'
    series: pd.Series = pd.Series(a1, name=name)
    other: Union[np.ndarray, SparseArray] = a2
    array_args: Tuple[Union[np.ndarray, SparseArray], Union[np.ndarray, SparseArray]] = (a1, a2)
    series_args: Tuple[Union[pd.Series, Any], Union[np.ndarray, SparseArray]] = (series, other)
    if flip:
        array_args = tuple(reversed(array_args))
        series_args = tuple(reversed(series_args))
    expected: pd.Series = pd.Series(ufunc(*array_args), name=name)
    result: pd.Series = ufunc(*series_args)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize('flip', [True, False], ids=['flipped', 'straight'])
def test_binary_ufunc_with_index(
    flip: bool,
    sparse: bool,
    ufunc: Callable[..., Any],
    arrays_for_binary_ufunc: Tuple[np.ndarray, np.ndarray]
) -> None:
    a1, a2 = arrays_for_binary_ufunc
    if sparse:
        a1 = SparseArray(a1, dtype=pd.SparseDtype('int64', 0))
        a2 = SparseArray(a2, dtype=pd.SparseDtype('int64', 0))
    name: str = 'name'
    series: pd.Series = pd.Series(a1, name=name)
    other: pd.Index = pd.Index(a2, name=name).astype('int64')
    array_args: Tuple[Union[np.ndarray, SparseArray], Union[np.ndarray, SparseArray]] = (a1, a2)
    series_args: Tuple[Union[pd.Series, pd.Index], Union[pd.Series, pd.Index]] = (series, other)
    if flip:
        array_args = tuple(reversed(array_args))
        series_args = tuple(reversed(series_args))
    expected: pd.Series = pd.Series(ufunc(*array_args), name=name)
    result: pd.Series = ufunc(*series_args)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize('shuffle', [True, False], ids=['unaligned', 'aligned'])
@pytest.mark.parametrize('flip', [True, False], ids=['flipped', 'straight'])
def test_binary_ufunc_with_series(
    flip: bool,
    shuffle: bool,
    sparse: bool,
    ufunc: Callable[..., Any],
    arrays_for_binary_ufunc: Tuple[np.ndarray, np.ndarray]
) -> None:
    a1, a2 = arrays_for_binary_ufunc
    if sparse:
        a1 = SparseArray(a1, dtype=pd.SparseDtype('int64', 0))
        a2 = SparseArray(a2, dtype=pd.SparseDtype('int64', 0))
    name: str = 'name'
    series: pd.Series = pd.Series(a1, name=name)
    other: pd.Series = pd.Series(a2, name=name)
    idx: np.ndarray = np.random.default_rng(2).permutation(len(a1))
    if shuffle:
        other = other.take(idx)
        if flip:
            index: Index = other.align(series)[0].index
        else:
            index = series.align(other)[0].index
    else:
        index = series.index
    array_args: Tuple[Union[np.ndarray, SparseArray], Union[np.ndarray, SparseArray]] = (a1, a2)
    series_args: Tuple[pd.Series, pd.Series] = (series, other)
    if flip:
        array_args = tuple(reversed(array_args))
        series_args = tuple(reversed(series_args))
    expected: pd.Series = pd.Series(ufunc(*array_args), index=index, name=name)
    result: pd.Series = ufunc(*series_args)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize('flip', [True, False])
def test_binary_ufunc_scalar(
    ufunc: Callable[..., Any],
    sparse: bool,
    flip: bool,
    arrays_for_binary_ufunc: Tuple[np.ndarray, np.ndarray]
) -> None:
    arr, _ = arrays_for_binary_ufunc
    if sparse:
        arr = SparseArray(arr)
    other: int = 2
    series: pd.Series = pd.Series(arr, name='name')
    series_args: Tuple[Union[pd.Series, int], Union[pd.Series, int]] = (series, other)
    array_args: Tuple[Union[np.ndarray, SparseArray], int] = (arr, other)
    if flip:
        series_args = tuple(reversed(series_args))
        array_args = tuple(reversed(array_args))
    expected: pd.Series = pd.Series(ufunc(*array_args), name='name')
    result: pd.Series = ufunc(*series_args)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize('ufunc', [np.divmod])
@pytest.mark.parametrize('shuffle', [True, False])
@pytest.mark.filterwarnings('ignore:divide by zero:RuntimeWarning')
def test_multiple_output_binary_ufuncs(
    ufunc: Callable[..., Any],
    sparse: bool,
    shuffle: bool,
    arrays_for_binary_ufunc: Tuple[np.ndarray, np.ndarray]
) -> None:
    a1, a2 = arrays_for_binary_ufunc
    a1[a1 == 0] = 1
    a2[a2 == 0] = 1
    if sparse:
        a1 = SparseArray(a1, dtype=pd.SparseDtype('int64', 0))
        a2 = SparseArray(a2, dtype=pd.SparseDtype('int64', 0))
    s1: pd.Series = pd.Series(a1)
    s2: pd.Series = pd.Series(a2)
    if shuffle:
        s2 = s2.sample(frac=1)
    expected: Tuple[Any, Any] = ufunc(a1, a2)
    assert isinstance(expected, tuple)
    result: Tuple[Any, Any] = ufunc(s1, s2)
    assert isinstance(result, tuple)
    tm.assert_series_equal(result[0], pd.Series(expected[0]))
    tm.assert_series_equal(result[1], pd.Series(expected[1]))


def test_multiple_output_ufunc(
    sparse: bool,
    arrays_for_binary_ufunc: Tuple[np.ndarray, np.ndarray]
) -> None:
    arr, _ = arrays_for_binary_ufunc
    if sparse:
        arr = SparseArray(arr)
    series: pd.Series = pd.Series(arr, name='name')
    result: Tuple[Any, Any] = np.modf(series)
    expected: Tuple[Any, Any] = np.modf(arr)
    assert isinstance(result, tuple)
    assert isinstance(expected, tuple)
    tm.assert_series_equal(result[0], pd.Series(expected[0], name='name'))
    tm.assert_series_equal(result[1], pd.Series(expected[1], name='name'))


def test_binary_ufunc_drops_series_name(
    ufunc: Callable[..., Any],
    sparse: bool,
    arrays_for_binary_ufunc: Tuple[np.ndarray, np.ndarray]
) -> None:
    a1, a2 = arrays_for_binary_ufunc
    s1: pd.Series = pd.Series(a1, name='a')
    s2: pd.Series = pd.Series(a2, name='b')
    result: pd.Series = ufunc(s1, s2)
    assert result.name is None


def test_object_series_ok() -> None:

    class Dummy:

        def __init__(self, value: int) -> None:
            self.value: int = value

        def __add__(self, other: Any) -> 'Dummy':
            return Dummy(self.value + other.value)

    arr: np.ndarray = np.array([Dummy(0), Dummy(1)])
    ser: pd.Series = pd.Series(arr)
    tm.assert_series_equal(
        np.add(ser, ser),
        pd.Series(np.add(ser, arr))
    )
    tm.assert_series_equal(
        np.add(ser, Dummy(1)),
        pd.Series(np.add(ser, Dummy(1)))
    )


@pytest.fixture(
    params=[
        pd.array([1, 3, 2], dtype=np.int64),
        pd.array([1, 3, 2], dtype='Int64'),
        pd.array([1, 3, 2], dtype='Float32'),
        pd.array([1, 10, 2], dtype='Sparse[int]'),
        pd.to_datetime(['2000', '2010', '2001']),
        pd.to_datetime(['2000', '2010', '2001']).tz_localize('CET'),
        pd.to_datetime(['2000', '2010', '2001']).to_period(freq='D'),
        pd.to_timedelta(['1 Day', '3 Days', '2 Days']),
        pd.IntervalIndex([pd.Interval(0, 1), pd.Interval(2, 3), pd.Interval(1, 2)])
    ],
    ids=lambda x: str(x.dtype)
)
def values_for_np_reduce(request: pytest.FixtureRequest) -> Union[pd.ArrayLike, pd.IntervalIndex]:
    return request.param


class TestNumpyReductions:

    def test_multiply(
        self,
        values_for_np_reduce: Union[pd.ArrayLike, pd.IntervalIndex],
        box_with_array: Callable[[Any], Any],
        request: pytest.FixtureRequest
    ) -> None:
        box: Callable[[Any], Any] = box_with_array
        values: Union[pd.ArrayLike, pd.IntervalIndex] = values_for_np_reduce
        with tm.assert_produces_warning(None):
            obj: Any = box(values)
        if isinstance(values, SparseArray):
            mark = pytest.mark.xfail(reason="SparseArray has no 'prod'")
            request.applymarker(mark)
        if getattr(values, 'dtype', None) and values.dtype.kind in 'iuf':
            result: Any = np.multiply.reduce(obj)
            if box is pd.DataFrame:
                expected: pd.Series = obj.prod(numeric_only=False)
                tm.assert_series_equal(result, expected)
            elif box is pd.Index:
                expected: Any = obj._values.prod()
                assert result == expected
            else:
                expected: Any = obj.prod()
                assert result == expected
        else:
            msg: str = '|'.join([
                'does not support operation',
                'unsupported operand type',
                "ufunc 'multiply' cannot use operands"
            ])
            with pytest.raises(TypeError, match=msg):
                np.multiply.reduce(obj)

    def test_add(
        self,
        values_for_np_reduce: Union[pd.ArrayLike, pd.IntervalIndex],
        box_with_array: Callable[[Any], Any]
    ) -> None:
        box: Callable[[Any], Any] = box_with_array
        values: Union[pd.ArrayLike, pd.IntervalIndex] = values_for_np_reduce
        with tm.assert_produces_warning(None):
            obj: Any = box(values)
        if getattr(values, 'dtype', None) and values.dtype.kind in 'miuf':
            result: Any = np.add.reduce(obj)
            if box is pd.DataFrame:
                expected: pd.Series = obj.sum(numeric_only=False)
                tm.assert_series_equal(result, expected)
            elif box is pd.Index:
                expected: Any = obj._values.sum()
                assert result == expected
            else:
                expected: Any = obj.sum()
                assert result == expected
        else:
            msg: str = '|'.join([
                'does not support operation',
                'unsupported operand type',
                "ufunc 'add' cannot use operands"
            ])
            with pytest.raises(TypeError, match=msg):
                np.add.reduce(obj)

    def test_max(
        self,
        values_for_np_reduce: Union[pd.ArrayLike, pd.IntervalIndex],
        box_with_array: Callable[[Any], Any]
    ) -> None:
        box: Callable[[Any], Any] = box_with_array
        values: Union[pd.ArrayLike, pd.IntervalIndex] = values_for_np_reduce
        same_type: bool = True
        if box is pd.Index and getattr(values, 'dtype', None) and values.dtype.kind in ['i', 'f']:
            same_type = False
        with tm.assert_produces_warning(None):
            obj: Any = box(values)
        result: Any = np.maximum.reduce(obj)
        if box is pd.DataFrame:
            expected: pd.Series = obj.max(numeric_only=False)
            tm.assert_series_equal(result, expected)
        else:
            expected: Any = values[1]
            assert result == expected
            if same_type:
                assert type(result) == type(expected)

    def test_min(
        self,
        values_for_np_reduce: Union[pd.ArrayLike, pd.IntervalIndex],
        box_with_array: Callable[[Any], Any]
    ) -> None:
        box: Callable[[Any], Any] = box_with_array
        values: Union[pd.ArrayLike, pd.IntervalIndex] = values_for_np_reduce
        same_type: bool = True
        if box is pd.Index and getattr(values, 'dtype', None) and values.dtype.kind in ['i', 'f']:
            same_type = False
        with tm.assert_produces_warning(None):
            obj: Any = box(values)
        result: Any = np.minimum.reduce(obj)
        if box is pd.DataFrame:
            expected: pd.Series = obj.min(numeric_only=False)
            tm.assert_series_equal(result, expected)
        else:
            expected: Any = values[0]
            assert result == expected
            if same_type:
                assert type(result) == type(expected)


@pytest.mark.parametrize('type_', [list, deque, tuple])
def test_binary_ufunc_other_types(type_: Callable[[Any], Any]) -> None:
    a: pd.Series = pd.Series([1, 2, 3], name='name')
    b: Union[list, deque, tuple] = type_([3, 4, 5])
    result: pd.Series = np.add(a, b)
    expected: pd.Series = pd.Series(np.add(a.to_numpy(), b), name='name')
    tm.assert_series_equal(result, expected)


def test_object_dtype_ok() -> None:

    class Thing:

        def __init__(self, value: int) -> None:
            self.value: int = value

        def __add__(self, other: Any) -> 'Thing':
            other_value: Any = getattr(other, 'value', other)
            return Thing(self.value + other_value)

        def __eq__(self, other: Any) -> bool:
            return isinstance(other, Thing) and self.value == other.value

        def __repr__(self) -> str:
            return f'Thing({self.value})'

    s: pd.Series = pd.Series([Thing(1), Thing(2)])
    result: pd.Series = np.add(s, Thing(1))
    expected: pd.Series = pd.Series([Thing(2), Thing(3)])
    tm.assert_series_equal(result, expected)


def test_outer() -> None:
    ser: pd.Series = pd.Series([1, 2, 3])
    obj: np.ndarray = np.array([1, 2, 3])
    with pytest.raises(NotImplementedError, match=''):
        np.subtract.outer(ser, obj)


def test_np_matmul() -> None:
    df1: pd.DataFrame = pd.DataFrame(data=[[-1, 1, 10]])
    df2: pd.DataFrame = pd.DataFrame(data=[-1, 1, 10])
    expected: pd.DataFrame = pd.DataFrame(data=[102])
    result: pd.DataFrame = np.matmul(df1, df2)
    tm.assert_frame_equal(expected, result)


@pytest.mark.parametrize('box', [pd.Index, pd.Series])
def test_np_matmul_1D(box: Callable[[Any], Any]) -> None:
    result: np.int64 = np.matmul(box([1, 2]), box([2, 3]))
    assert result == 8
    assert isinstance(result, np.int64)


def test_array_ufuncs_for_many_arguments() -> None:

    def add3(x: Any, y: Any, z: Any) -> Any:
        return x + y + z

    ufunc: Callable[..., Any] = np.frompyfunc(add3, 3, 1)
    ser: pd.Series = pd.Series([1, 2])
    result: pd.Series = ufunc(ser, ser, 1)
    expected: pd.Series = pd.Series([3, 5], dtype=object)
    tm.assert_series_equal(result, expected)
    df: pd.DataFrame = pd.DataFrame([[1, 2]])
    msg: str = "Cannot apply ufunc <ufunc 'add3 (vectorized)'> to mixed DataFrame and Series inputs."
    with pytest.raises(NotImplementedError, match=re.escape(msg)):
        ufunc(ser, ser, df)


@pytest.mark.xfail(reason='see https://github.com/pandas-dev/pandas/pull/51082')
def test_np_fix() -> None:
    ser: pd.Series = pd.Series([-1.5, -0.5, 0.5, 1.5])
    result: pd.Series = np.fix(ser)
    expected: pd.Series = pd.Series([-1.0, -0.0, 0.0, 1.0])
    tm.assert_series_equal(result, expected)
