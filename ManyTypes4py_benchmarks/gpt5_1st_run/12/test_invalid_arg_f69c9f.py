from typing import Any, Callable, Mapping, Sequence, Tuple, Union
from itertools import chain
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas import DataFrame, Series, date_range
import pandas._testing as tm


@pytest.mark.parametrize('result_type', ['foo', 1])
def test_result_type_error(result_type: Union[str, int]) -> None:
    df = DataFrame(
        np.tile(np.arange(3, dtype='int64'), 6).reshape(6, -1) + 1, columns=['A', 'B', 'C']
    )
    msg = "invalid value for result_type, must be one of {None, 'reduce', 'broadcast', 'expand'}"
    with pytest.raises(ValueError, match=msg):
        df.apply(lambda x: [1, 2, 3], axis=1, result_type=result_type)


def test_apply_invalid_axis_value() -> None:
    df = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=['a', 'a', 'c'])
    msg = 'No axis named 2 for object type DataFrame'
    with pytest.raises(ValueError, match=msg):
        df.apply(lambda x: x, 2)


def test_agg_raises() -> None:
    df = DataFrame({'A': [0, 1], 'B': [1, 2]})
    msg = 'Must provide'
    with pytest.raises(TypeError, match=msg):
        df.agg()


def test_map_with_invalid_na_action_raises() -> None:
    s = Series([1, 2, 3])
    msg = "na_action must either be 'ignore' or None"
    with pytest.raises(ValueError, match=msg):
        s.map(lambda x: x, na_action='____')


@pytest.mark.parametrize('input_na_action', ['____', True])
def test_map_arg_is_dict_with_invalid_na_action_raises(input_na_action: Union[str, bool]) -> None:
    s = Series([1, 2, 3])
    msg = f"na_action must either be 'ignore' or None, {input_na_action} was passed"
    with pytest.raises(ValueError, match=msg):
        s.map({1: 2}, na_action=input_na_action)


@pytest.mark.parametrize('method', ['apply', 'agg', 'transform'])
@pytest.mark.parametrize('func', [{'A': {'B': 'sum'}}, {'A': {'B': ['sum']}}])
def test_nested_renamer(
    frame_or_series: Callable[[Mapping[str, Sequence[int]]], Union[DataFrame, Series]],
    method: str,
    func: Mapping[str, Mapping[str, Union[str, Sequence[str]]]],
) -> None:
    obj = frame_or_series({'A': [1]})
    match = 'nested renamer is not supported'
    with pytest.raises(SpecificationError, match=match):
        getattr(obj, method)(func)


@pytest.mark.parametrize('renamer', [{'foo': ['min', 'max']}, {'foo': ['min', 'max'], 'bar': ['sum', 'mean']}])
def test_series_nested_renamer(renamer: Mapping[str, Sequence[str]]) -> None:
    s = Series(range(6), dtype='int64', name='series')
    msg = 'nested renamer is not supported'
    with pytest.raises(SpecificationError, match=msg):
        s.agg(renamer)


def test_apply_dict_depr() -> None:
    tsdf = DataFrame(
        np.random.default_rng(2).standard_normal((10, 3)),
        columns=['A', 'B', 'C'],
        index=date_range('1/1/2000', periods=10),
    )
    msg = 'nested renamer is not supported'
    with pytest.raises(SpecificationError, match=msg):
        tsdf.A.agg({'foo': ['sum', 'mean']})


@pytest.mark.parametrize('method', ['agg', 'transform'])
def test_dict_nested_renaming_depr(method: str) -> None:
    df = DataFrame({'A': range(5), 'B': 5})
    msg = 'nested renamer is not supported'
    with pytest.raises(SpecificationError, match=msg):
        getattr(df, method)({'A': {'foo': 'min'}, 'B': {'bar': 'max'}})


@pytest.mark.parametrize('method', ['apply', 'agg', 'transform'])
@pytest.mark.parametrize('func', [{'B': 'sum'}, {'B': ['sum']}])
def test_missing_column(method: str, func: Mapping[str, Union[str, Sequence[str]]]) -> None:
    obj = DataFrame({'A': [1]})
    msg = "Label\\(s\\) \\['B'\\] do not exist"
    with pytest.raises(KeyError, match=msg):
        getattr(obj, method)(func)


def test_transform_mixed_column_name_dtypes() -> None:
    df = DataFrame({'a': ['1']})
    msg = "Label\\(s\\) \\[1, 'b'\\] do not exist"
    with pytest.raises(KeyError, match=msg):
        df.transform({'a': int, 1: str, 'b': int})


@pytest.mark.parametrize('how, args', [('pct_change', ()), ('nsmallest', (1, ['a', 'b'])), ('tail', 1)])
def test_apply_str_axis_1_raises(how: str, args: Union[int, Tuple[Any, ...]]) -> None:
    df = DataFrame({'a': [1, 2], 'b': [3, 4]})
    msg = f'Operation {how} does not support axis=1'
    with pytest.raises(ValueError, match=msg):
        df.apply(how, axis=1, args=args)


def test_transform_axis_1_raises() -> None:
    msg = 'No axis named 1 for object type Series'
    with pytest.raises(ValueError, match=msg):
        Series([1]).transform('sum', axis=1)


def test_apply_modify_traceback() -> None:
    data = DataFrame(
        {
            'A': ['foo', 'foo', 'foo', 'foo', 'bar', 'bar', 'bar', 'bar', 'foo', 'foo', 'foo'],
            'B': ['one', 'one', 'one', 'two', 'one', 'one', 'one', 'two', 'two', 'two', 'one'],
            'C': ['dull', 'dull', 'shiny', 'dull', 'dull', 'shiny', 'shiny', 'dull', 'shiny', 'shiny', 'shiny'],
            'D': np.random.default_rng(2).standard_normal(11),
            'E': np.random.default_rng(2).standard_normal(11),
            'F': np.random.default_rng(2).standard_normal(11),
        }
    )
    data.loc[4, 'C'] = np.nan

    def transform(row: Series) -> Series:
        if row['C'].startswith('shin') and row['A'] == 'foo':
            row['D'] = 7
        return row

    msg = "'float' object has no attribute 'startswith'"
    with pytest.raises(AttributeError, match=msg):
        data.apply(transform, axis=1)


@pytest.mark.parametrize(
    'df, func, expected',
    tm.get_cython_table_params(
        DataFrame([['a', 'b'], ['b', 'a']]),
        [['cumprod', TypeError]],
    ),
)
def test_agg_cython_table_raises_frame(
    df: DataFrame,
    func: Union[str, Callable[..., Any]],
    expected: Any,
    axis: Union[int, str],
    using_infer_string: bool,
) -> None:
    if using_infer_string:
        expected = (expected, NotImplementedError)
    msg = "can't multiply sequence by non-int of type 'str'|cannot perform cumprod with type str|operation 'cumprod' not supported for dtype 'str'"
    warn = None if isinstance(func, str) else FutureWarning
    with pytest.raises(expected, match=msg):
        with tm.assert_produces_warning(warn, match='using DataFrame.cumprod'):
            df.agg(func, axis=axis)


@pytest.mark.parametrize(
    'series, func, expected',
    chain(
        tm.get_cython_table_params(
            Series('a b c'.split()),
            [('mean', TypeError), ('prod', TypeError), ('std', TypeError), ('var', TypeError), ('median', TypeError), ('cumprod', TypeError)],
        )
    ),
)
def test_agg_cython_table_raises_series(
    series: Series,
    func: Union[str, Callable[..., Any]],
    expected: Any,
    using_infer_string: bool,
) -> None:
    msg = "[Cc]ould not convert|can't multiply sequence by non-int of type"
    if func == 'median' or func is np.nanmedian or func is np.median:
        msg = "Cannot convert \\['a' 'b' 'c'\\] to numeric"
    if using_infer_string and func == 'cumprod':
        expected = (expected, NotImplementedError)
    msg = msg + '|does not support|has no kernel|Cannot perform|cannot perform|operation'
    warn = None if isinstance(func, str) else FutureWarning
    with pytest.raises(expected, match=msg):
        with tm.assert_produces_warning(warn, match='is currently using Series.*'):
            series.agg(func)


def test_agg_none_to_type() -> None:
    df = DataFrame({'a': [None]})
    msg = re.escape('int() argument must be a string')
    with pytest.raises(TypeError, match=msg):
        df.agg({'a': lambda x: int(x.iloc[0])})


def test_transform_none_to_type() -> None:
    df = DataFrame({'a': [None]})
    msg = 'argument must be a'
    with pytest.raises(TypeError, match=msg):
        df.transform({'a': lambda x: int(x.iloc[0])})


@pytest.mark.parametrize('func', [lambda x: np.array([1, 2]).reshape(-1, 2), lambda x: [1, 2], lambda x: Series([1, 2])])
def test_apply_broadcast_error(func: Callable[[Series], Any]) -> None:
    df = DataFrame(
        np.tile(np.arange(3, dtype='int64'), 6).reshape(6, -1) + 1, columns=['A', 'B', 'C']
    )
    msg = 'too many dims to broadcast|cannot broadcast result'
    with pytest.raises(ValueError, match=msg):
        df.apply(func, axis=1, result_type='broadcast')


def test_transform_and_agg_err_agg(axis: Union[int, str], float_frame: DataFrame) -> None:
    msg = 'cannot combine transform and aggregation operations'
    with pytest.raises(ValueError, match=msg):
        with np.errstate(all='ignore'):
            float_frame.agg(['max', 'sqrt'], axis=axis)


@pytest.mark.filterwarnings('ignore::FutureWarning')
@pytest.mark.parametrize(
    'func, msg',
    [
        (['sqrt', 'max'], 'cannot combine transform and aggregation'),
        ({'foo': np.sqrt, 'bar': 'sum'}, 'cannot perform both aggregation and transformation'),
    ],
)
def test_transform_and_agg_err_series(
    string_series: Series,
    func: Union[Sequence[str], Mapping[str, Union[Callable[..., Any], str]]],
    msg: str,
) -> None:
    with pytest.raises(ValueError, match=msg):
        with np.errstate(all='ignore'):
            string_series.agg(func)


@pytest.mark.parametrize('func', [['max', 'min'], ['max', 'sqrt']])
def test_transform_wont_agg_frame(
    axis: Union[int, str], float_frame: DataFrame, func: Sequence[str]
) -> None:
    msg = 'Function did not transform'
    with pytest.raises(ValueError, match=msg):
        float_frame.transform(func, axis=axis)


@pytest.mark.parametrize('func', [['min', 'max'], ['sqrt', 'max']])
def test_transform_wont_agg_series(string_series: Series, func: Sequence[str]) -> None:
    msg = 'Function did not transform'
    with pytest.raises(ValueError, match=msg):
        string_series.transform(func)


@pytest.mark.parametrize('op_wrapper', [lambda x: x, lambda x: [x], lambda x: {'A': x}, lambda x: {'A': [x]}])
def test_transform_reducer_raises(
    all_reductions: Any,
    frame_or_series: Callable[[Mapping[str, Sequence[int]]], Union[DataFrame, Series]],
    op_wrapper: Callable[[Any], Any],
) -> None:
    op = op_wrapper(all_reductions)
    obj = DataFrame({'A': [1, 2, 3]})
    obj = tm.get_obj(obj, frame_or_series)
    msg = 'Function did not transform'
    with pytest.raises(ValueError, match=msg):
        obj.transform(op)


def test_transform_missing_labels_raises() -> None:
    df = DataFrame({'foo': [2, 4, 6], 'bar': [1, 2, 3]}, index=['A', 'B', 'C'])
    msg = "Label\\(s\\) \\['A', 'B'\\] do not exist"
    with pytest.raises(KeyError, match=msg):
        df.transform({'A': lambda x: x + 2, 'B': lambda x: x * 2}, axis=0)
    msg = "Label\\(s\\) \\['bar', 'foo'\\] do not exist"
    with pytest.raises(KeyError, match=msg):
        df.transform({'foo': lambda x: x + 2, 'bar': lambda x: x * 2}, axis=1)