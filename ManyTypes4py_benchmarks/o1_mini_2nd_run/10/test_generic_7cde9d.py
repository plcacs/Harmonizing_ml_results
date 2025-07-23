from copy import copy, deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
from pandas import DataFrame, Index, Series, date_range
import pandas._testing as tm

BoxType = Union[Series, DataFrame]

def construct(
    box: BoxType,
    shape: Union[int, Tuple[int, ...]],
    value: Optional[Union[Any, np.ndarray, str]] = None,
    dtype: Optional[Union[str, np.dtype]] = None,
    **kwargs: Any
) -> BoxType:
    """
    construct an object for the given shape
    if value is specified use that if its a scalar
    if value is an array, repeat it as needed
    """
    if isinstance(shape, int):
        shape = tuple([shape] * box._AXIS_LEN)
    if value is not None:
        if is_scalar(value):
            if value == 'empty':
                arr = None
                dtype = np.float64
                kwargs.pop(box._info_axis_name, None)
            else:
                arr = np.empty(shape, dtype=dtype)
                arr.fill(value)
        else:
            fshape = np.prod(shape)
            arr = value.ravel()
            new_shape = fshape / arr.shape[0]
            if fshape % arr.shape[0] != 0:
                raise Exception('invalid value passed in construct')
            arr = np.repeat(arr, int(new_shape)).reshape(shape)
    else:
        arr = np.random.default_rng(2).standard_normal(shape)
    return box(arr, dtype=dtype, **kwargs)

class TestGeneric:

    @pytest.mark.parametrize(
        'func',
        [
            Callable[[str], str],
            Dict[str, str],
            Series
        ]
    )
    def test_rename(
        self,
        frame_or_series: BoxType,
        func: Union[Callable[[str], str], Dict[str, str], Series]
    ) -> None:
        idx: List[str] = list('ABCD')
        for axis in frame_or_series._AXIS_ORDERS:
            kwargs: Dict[str, Any] = {axis: idx}
            obj: BoxType = construct(frame_or_series, 4, **kwargs)
            result: BoxType = obj.rename(**{axis: func})
            expected: BoxType = obj.copy()
            setattr(expected, axis, list('abcd'))
            tm.assert_equal(result, expected)

    def test_get_numeric_data(self, frame_or_series: BoxType) -> None:
        n: int = 4
        kwargs: Dict[str, List[int]] = {
            frame_or_series._get_axis_name(i): list(range(n))
            for i in range(frame_or_series._AXIS_LEN)
        }
        o: BoxType = construct(frame_or_series, n, **kwargs)
        result: BoxType = o._get_numeric_data()
        tm.assert_equal(result, o)
        result = o._get_bool_data()
        expected: BoxType = construct(frame_or_series, n, value='empty', **kwargs)
        if isinstance(o, DataFrame):
            expected.columns = o.columns[:0]
        tm.assert_equal(result, expected)
        arr: np.ndarray = np.array([True, True, False, True])
        o = construct(frame_or_series, n, value=arr, **kwargs)
        result = o._get_numeric_data()
        tm.assert_equal(result, o)

    def test_get_bool_data_empty_preserve_index(self) -> None:
        expected: Series = Series([], dtype='bool')
        result: Series = expected._get_bool_data()
        tm.assert_series_equal(result, expected, check_index_type=True)

    def test_nonzero(self, frame_or_series: BoxType) -> None:
        obj: BoxType = construct(frame_or_series, shape=4)
        msg: str = f'The truth value of a {frame_or_series.__class__.__name__} is ambiguous'
        with pytest.raises(ValueError, match=msg):
            bool(obj == 0)
        with pytest.raises(ValueError, match=msg):
            bool(obj == 1)
        with pytest.raises(ValueError, match=msg):
            bool(obj)
        obj = construct(frame_or_series, shape=4, value=1)
        with pytest.raises(ValueError, match=msg):
            bool(obj == 0)
        with pytest.raises(ValueError, match=msg):
            bool(obj == 1)
        with pytest.raises(ValueError, match=msg):
            bool(obj)
        obj = construct(frame_or_series, shape=4, value=np.nan)
        with pytest.raises(ValueError, match=msg):
            bool(obj == 0)
        with pytest.raises(ValueError, match=msg):
            bool(obj == 1)
        with pytest.raises(ValueError, match=msg):
            bool(obj)
        obj = construct(frame_or_series, shape=0)
        with pytest.raises(ValueError, match=msg):
            bool(obj)
        obj1: BoxType = construct(frame_or_series, shape=4, value=1)
        obj2: BoxType = construct(frame_or_series, shape=4, value=1)
        with pytest.raises(ValueError, match=msg):
            if obj1:
                pass
        with pytest.raises(ValueError, match=msg):
            obj1 and obj2
        with pytest.raises(ValueError, match=msg):
            obj1 or obj2
        with pytest.raises(ValueError, match=msg):
            not obj1

    def test_frame_or_series_compound_dtypes(self, frame_or_series: BoxType) -> None:

        def f(dtype: Union[str, List[Tuple[str, str]]]) -> BoxType:
            return construct(frame_or_series, shape=3, value=1, dtype=dtype)

        msg: str = f'compound dtypes are not implemented in the {frame_or_series.__class__.__name__} constructor'
        with pytest.raises(NotImplementedError, match=msg):
            f([('A', 'datetime64[h]'), ('B', 'str'), ('C', 'int32')])
        f('int64')
        f('float64')
        f('M8[ns]')

    def test_metadata_propagation(self, frame_or_series: BoxType) -> None:
        o: BoxType = construct(frame_or_series, shape=3)
        o.name = 'foo'
        o2: BoxType = construct(frame_or_series, shape=3)
        o2.name = 'bar'
        for op in ['__add__', '__sub__', '__truediv__', '__mul__']:
            result: BoxType = getattr(o, op)(1)
            tm.assert_metadata_equivalent(o, result)
        for op in ['__add__', '__sub__', '__truediv__', '__mul__']:
            result = getattr(o, op)(o)
            tm.assert_metadata_equivalent(o, result)
        for op in ['__eq__', '__le__', '__ge__']:
            v1: BoxType = getattr(o, op)(o)
            tm.assert_metadata_equivalent(o, v1)
            tm.assert_metadata_equivalent(o, v1 & v1)
            tm.assert_metadata_equivalent(o, v1 | v1)
        result = o.combine_first(o2)
        tm.assert_metadata_equivalent(o, result)
        result = o + o2
        tm.assert_metadata_equivalent(result)
        for op in ['__eq__', '__le__', '__ge__']:
            v1 = getattr(o, op)(o)
            v2 = getattr(o, op)(o2)
            tm.assert_metadata_equivalent(v2)
            tm.assert_metadata_equivalent(v1 & v2)
            tm.assert_metadata_equivalent(v1 | v2)

    def test_size_compat(self, frame_or_series: BoxType) -> None:
        o: BoxType = construct(frame_or_series, shape=10)
        assert o.size == np.prod(o.shape)
        assert o.size == 10 ** len(o.axes)

    def test_split_compat(self, frame_or_series: BoxType) -> None:
        o: BoxType = construct(frame_or_series, shape=10)
        split_result_5: List[BoxType] = list(np.array_split(o, 5))
        assert len(split_result_5) == 5
        split_result_2: List[BoxType] = list(np.array_split(o, 2))
        assert len(split_result_2) == 2

    def test_stat_unexpected_keyword(self, frame_or_series: BoxType) -> None:
        obj: BoxType = construct(frame_or_series, 5)
        starwars: str = 'Star Wars'
        errmsg: str = 'unexpected keyword'
        with pytest.raises(TypeError, match=errmsg):
            obj.max(epic=starwars)
        with pytest.raises(TypeError, match=errmsg):
            obj.var(epic=starwars)
        with pytest.raises(TypeError, match=errmsg):
            obj.sum(epic=starwars)
        with pytest.raises(TypeError, match=errmsg):
            obj.any(epic=starwars)

    @pytest.mark.parametrize('func', ['sum', 'cumsum', 'any', 'var'])
    def test_api_compat(
        self,
        func: str,
        frame_or_series: BoxType
    ) -> None:
        obj: BoxType = construct(frame_or_series, 5)
        f: Callable[..., Any] = getattr(obj, func)
        assert f.__name__ == func
        assert f.__qualname__.endswith(func)

    def test_stat_non_defaults_args(self, frame_or_series: BoxType) -> None:
        obj: BoxType = construct(frame_or_series, 5)
        out: np.ndarray = np.array([0])
        errmsg: str = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            obj.max(out=out)
        with pytest.raises(ValueError, match=errmsg):
            obj.var(out=out)
        with pytest.raises(ValueError, match=errmsg):
            obj.sum(out=out)
        with pytest.raises(ValueError, match=errmsg):
            obj.any(out=out)

    def test_truncate_out_of_bounds(self, frame_or_series: BoxType) -> None:
        shape: List[int] = [2000] + [1] * (frame_or_series._AXIS_LEN - 1)
        small: BoxType = construct(frame_or_series, shape, dtype='int8', value=1)
        tm.assert_equal(small.truncate(), small)
        tm.assert_equal(small.truncate(before=0, after=3000.0), small)
        tm.assert_equal(small.truncate(before=-1, after=2000.0), small)
        shape = [2000000] + [1] * (frame_or_series._AXIS_LEN - 1)
        big: BoxType = construct(frame_or_series, shape, dtype='int8', value=1)
        tm.assert_equal(big.truncate(), big)
        tm.assert_equal(big.truncate(before=0, after=3000000.0), big)
        tm.assert_equal(big.truncate(before=-1, after=2000000.0), big)

    @pytest.mark.parametrize(
        'func',
        [
            copy,
            deepcopy,
            Callable[[BoxType], BoxType],
            Callable[[BoxType], BoxType]
        ]
    )
    @pytest.mark.parametrize('shape', [0, 1, 2])
    def test_copy_and_deepcopy(
        self,
        frame_or_series: BoxType,
        shape: int,
        func: Callable[[BoxType], BoxType]
    ) -> None:
        obj: BoxType = construct(frame_or_series, shape)
        obj_copy: BoxType = func(obj)
        assert obj_copy is not obj
        tm.assert_equal(obj_copy, obj)

class TestNDFrame:

    @pytest.mark.parametrize(
        'ser',
        [
            Series(range(10), dtype=np.float64),
            Series([str(i) for i in range(10)], dtype=object)
        ]
    )
    def test_squeeze_series_noop(self, ser: Series) -> None:
        tm.assert_series_equal(ser.squeeze(), ser)

    def test_squeeze_frame_noop(self) -> None:
        df: DataFrame = DataFrame(np.eye(2))
        tm.assert_frame_equal(df.squeeze(), df)

    def test_squeeze_frame_reindex(self) -> None:
        df: DataFrame = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list('ABCD'), dtype=object),
            index=date_range('2000-01-01', periods=10, freq='B')
        ).reindex(columns=['A'])
        tm.assert_series_equal(df.squeeze(), df['A'])

    def test_squeeze_0_len_dim(self) -> None:
        empty_series: Series = Series([], name='five', dtype=np.float64)
        empty_frame: DataFrame = DataFrame([empty_series])
        tm.assert_series_equal(empty_series, empty_series.squeeze())
        tm.assert_series_equal(empty_series, empty_frame.squeeze())

    def test_squeeze_axis(self) -> None:
        df: DataFrame = DataFrame(
            np.random.default_rng(2).standard_normal((1, 4)),
            columns=Index(list('ABCD'), dtype=object),
            index=date_range('2000-01-01', periods=1, freq='B')
        ).iloc[:, :1]
        assert df.shape == (1, 1)
        squeezed_axis0: Series = df.squeeze(axis=0)
        tm.assert_series_equal(squeezed_axis0, df.iloc[0])
        squeezed_index: Series = df.squeeze(axis='index')
        tm.assert_series_equal(squeezed_index, df.iloc[0])
        squeezed_axis1: Series = df.squeeze(axis=1)
        tm.assert_series_equal(squeezed_axis1, df.iloc[:, 0])
        squeezed_columns: Series = df.squeeze(axis='columns')
        tm.assert_series_equal(squeezed_columns, df.iloc[:, 0])
        assert df.squeeze() == df.iloc[0, 0]
        msg: str = 'No axis named 2 for object type DataFrame'
        with pytest.raises(ValueError, match=msg):
            df.squeeze(axis=2)
        msg = 'No axis named x for object type DataFrame'
        with pytest.raises(ValueError, match=msg):
            df.squeeze(axis='x')

    def test_squeeze_axis_len_3(self) -> None:
        df: DataFrame = DataFrame(
            np.random.default_rng(2).standard_normal((3, 4)),
            columns=Index(list('ABCD'), dtype=object),
            index=date_range('2000-01-01', periods=3, freq='B')
        )
        tm.assert_frame_equal(df.squeeze(axis=0), df)

    def test_numpy_squeeze(self) -> None:
        s: Series = Series(range(2), dtype=np.float64)
        tm.assert_series_equal(np.squeeze(s), s)
        df: DataFrame = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list('ABCD'), dtype=object),
            index=date_range('2000-01-01', periods=10, freq='B')
        ).reindex(columns=['A'])
        tm.assert_series_equal(np.squeeze(df), df['A'])

    @pytest.mark.parametrize(
        'ser',
        [
            Series(range(10), dtype=np.float64),
            Series([str(i) for i in range(10)], dtype=object)
        ]
    )
    def test_transpose_series(self, ser: Series) -> None:
        tm.assert_series_equal(ser.transpose(), ser)

    def test_transpose_frame(self) -> None:
        df: DataFrame = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list('ABCD'), dtype=object),
            index=date_range('2000-01-01', periods=10, freq='B')
        )
        tm.assert_frame_equal(df.transpose().transpose(), df)

    def test_numpy_transpose(self, frame_or_series: BoxType) -> None:
        obj: DataFrame = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list('ABCD'), dtype=object),
            index=date_range('2000-01-01', periods=10, freq='B')
        )
        obj = tm.get_obj(obj, frame_or_series)
        if frame_or_series is Series:
            tm.assert_series_equal(np.transpose(obj), obj)  # type: ignore
        tm.assert_equal(np.transpose(np.transpose(obj)), obj)  # type: ignore
        msg: str = "the 'axes' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            np.transpose(obj, axes=1)  # type: ignore

    @pytest.mark.parametrize(
        'ser',
        [
            Series(range(10), dtype=np.float64),
            Series([str(i) for i in range(10)], dtype=object)
        ]
    )
    def test_take_series(self, ser: Series) -> None:
        indices: List[int] = [1, 5, -2, 6, 3, -1]
        out: Series = ser.take(indices)
        expected: Series = Series(
            data=ser.values.take(indices),
            index=ser.index.take(indices),
            dtype=ser.dtype
        )
        tm.assert_series_equal(out, expected)

    def test_take_frame(self) -> None:
        indices: List[int] = [1, 5, -2, 6, 3, -1]
        df: DataFrame = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list('ABCD'), dtype=object),
            index=date_range('2000-01-01', periods=10, freq='B')
        )
        out: DataFrame = df.take(indices)
        expected: DataFrame = DataFrame(
            data=df.values.take(indices, axis=0),
            index=df.index.take(indices),
            columns=df.columns
        )
        tm.assert_frame_equal(out, expected)

    def test_take_invalid_kwargs(self, frame_or_series: BoxType) -> None:
        indices: List[int] = [-3, 2, 0, 1]
        obj: DataFrame = DataFrame(range(5))
        obj = tm.get_obj(obj, frame_or_series)
        msg: str = "take\\(\\) got an unexpected keyword argument 'foo'"
        with pytest.raises(TypeError, match=msg):
            obj.take(indices, foo=2)  # type: ignore
        msg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            obj.take(indices, out=indices)  # type: ignore
        msg = "the 'mode' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            obj.take(indices, mode='clip')  # type: ignore

    def test_axis_classmethods(self, frame_or_series: BoxType) -> None:
        box: BoxType = frame_or_series
        obj: BoxType = box(dtype=object)
        values: List[str] = list(box._AXIS_TO_AXIS_NUMBER.keys())
        for v in values:
            assert obj._get_axis_number(v) == box._get_axis_number(v)
            assert obj._get_axis_name(v) == box._get_axis_name(v)
            assert obj._get_block_manager_axis(v) == box._get_block_manager_axis(v)

    def test_flags_identity(self, frame_or_series: BoxType) -> None:
        obj: Series = Series([1, 2])
        if frame_or_series is DataFrame:
            obj = obj.to_frame()
        assert obj.flags is obj.flags
        obj2: BoxType = obj.copy()
        assert obj2.flags is not obj.flags
