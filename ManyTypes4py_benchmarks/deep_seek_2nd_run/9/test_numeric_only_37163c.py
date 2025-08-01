import re
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import pytest
from pandas._libs import lib
import pandas as pd
from pandas import DataFrame, Index, Series, Timestamp, date_range
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
from pandas.core.groupby.generic import DataFrameGroupBy, SeriesGroupBy

class TestNumericOnly:

    @pytest.fixture
    def df(self) -> DataFrame:
        df = DataFrame(
            {
                'group': [1, 1, 2],
                'int': [1, 2, 3],
                'float': [4.0, 5.0, 6.0],
                'string': Series(['a', 'b', 'c'], dtype='str'),
                'object': Series(['a', 'b', 'c'], dtype=object),
                'category_string': Series(list('abc')).astype('category'),
                'category_int': [7, 8, 9],
                'datetime': date_range('20130101', periods=3),
                'datetimetz': date_range('20130101', periods=3, tz='US/Eastern'),
                'timedelta': pd.timedelta_range('1 s', periods=3, freq='s'),
            },
            columns=[
                'group',
                'int',
                'float',
                'string',
                'object',
                'category_string',
                'category_int',
                'datetime',
                'datetimetz',
                'timedelta',
            ],
        )
        return df

    @pytest.mark.parametrize('method', ['mean', 'median'])
    def test_averages(self, df: DataFrame, method: str) -> None:
        expected_columns_numeric = Index(['int', 'float', 'category_int'])
        gb = df.groupby('group')
        expected = DataFrame(
            {
                'category_int': [7.5, 9],
                'float': [4.5, 6.0],
                'timedelta': [pd.Timedelta('1.5s'), pd.Timedelta('3s')],
                'int': [1.5, 3],
                'datetime': [
                    Timestamp('2013-01-01 12:00:00'),
                    Timestamp('2013-01-03 00:00:00'),
                ],
                'datetimetz': [
                    Timestamp('2013-01-01 12:00:00', tz='US/Eastern'),
                    Timestamp('2013-01-03 00:00:00', tz='US/Eastern'),
                ],
            },
            index=Index([1, 2], name='group'),
            columns=['int', 'float', 'category_int'],
        )
        result = getattr(gb, method)(numeric_only=True)
        tm.assert_frame_equal(result.reindex_like(expected), expected)
        expected_columns = expected.columns
        self._check(df, method, expected_columns, expected_columns_numeric)

    @pytest.mark.parametrize('method', ['min', 'max'])
    def test_extrema(self, df: DataFrame, method: str) -> None:
        expected_columns = Index(
            [
                'int',
                'float',
                'string',
                'category_int',
                'datetime',
                'datetimetz',
                'timedelta',
            ]
        )
        expected_columns_numeric = expected_columns
        self._check(df, method, expected_columns, expected_columns_numeric)

    @pytest.mark.parametrize('method', ['first', 'last'])
    def test_first_last(self, df: DataFrame, method: str) -> None:
        expected_columns = Index(
            [
                'int',
                'float',
                'string',
                'object',
                'category_string',
                'category_int',
                'datetime',
                'datetimetz',
                'timedelta',
            ]
        )
        expected_columns_numeric = expected_columns
        self._check(df, method, expected_columns, expected_columns_numeric)

    @pytest.mark.parametrize('method', ['sum', 'cumsum'])
    def test_sum_cumsum(self, df: DataFrame, method: str) -> None:
        expected_columns_numeric = Index(['int', 'float', 'category_int'])
        expected_columns = Index(
            ['int', 'float', 'string', 'category_int', 'timedelta']
        )
        if method == 'cumsum':
            expected_columns = Index(['int', 'float', 'category_int', 'timedelta'])
        self._check(df, method, expected_columns, expected_columns_numeric)

    @pytest.mark.parametrize('method', ['prod', 'cumprod'])
    def test_prod_cumprod(self, df: DataFrame, method: str) -> None:
        expected_columns = Index(['int', 'float', 'category_int'])
        expected_columns_numeric = expected_columns
        self._check(df, method, expected_columns, expected_columns_numeric)

    @pytest.mark.parametrize('method', ['cummin', 'cummax'])
    def test_cummin_cummax(self, df: DataFrame, method: str) -> None:
        expected_columns = Index(
            [
                'int',
                'float',
                'category_int',
                'datetime',
                'datetimetz',
                'timedelta',
            ]
        )
        expected_columns_numeric = expected_columns
        self._check(df, method, expected_columns, expected_columns_numeric)

    def _check(
        self,
        df: DataFrame,
        method: str,
        expected_columns: Index,
        expected_columns_numeric: Index,
    ) -> None:
        gb = df.groupby('group')
        exception = (
            (NotImplementedError, TypeError)
            if method.startswith('cum')
            else TypeError
        )
        if method in ('min', 'max', 'cummin', 'cummax', 'cumsum', 'cumprod'):
            msg = '|'.join(
                [
                    'Categorical is not ordered',
                    f'Cannot perform {method} with non-ordered Categorical',
                    re.escape(f'agg function failed [how->{method},dtype->object]'),
                    'function is not implemented for this dtype',
                    f"dtype 'str' does not support operation '{method}'",
                ]
            )
            with pytest.raises(exception, match=msg):
                getattr(gb, method)()
        elif method in ('sum', 'mean', 'median', 'prod'):
            msg = '|'.join(
                [
                    'category type does not support sum operations',
                    re.escape(f'agg function failed [how->{method},dtype->object]'),
                    re.escape(f'agg function failed [how->{method},dtype->string]'),
                    f"dtype 'str' does not support operation '{method}'",
                ]
            )
            with pytest.raises(exception, match=msg):
                getattr(gb, method)()
        else:
            result = getattr(gb, method)()
            tm.assert_index_equal(result.columns, expected_columns_numeric)
        if method not in ('first', 'last'):
            msg = '|'.join(
                [
                    'Categorical is not ordered',
                    'category type does not support',
                    'function is not implemented for this dtype',
                    f'Cannot perform {method} with non-ordered Categorical',
                    re.escape(f'agg function failed [how->{method},dtype->object]'),
                    re.escape(f'agg function failed [how->{method},dtype->string]'),
                    f"dtype 'str' does not support operation '{method}'",
                ]
            )
            with pytest.raises(exception, match=msg):
                getattr(gb, method)(numeric_only=False)
        else:
            result = getattr(gb, method)(numeric_only=False)
            tm.assert_index_equal(result.columns, expected_columns)


@pytest.mark.parametrize(
    'kernel, has_arg',
    [
        ('all', False),
        ('any', False),
        ('bfill', False),
        ('corr', True),
        ('corrwith', True),
        ('cov', True),
        ('cummax', True),
        ('cummin', True),
        ('cumprod', True),
        ('cumsum', True),
        ('diff', False),
        ('ffill', False),
        ('first', True),
        ('idxmax', True),
        ('idxmin', True),
        ('last', True),
        ('max', True),
        ('mean', True),
        ('median', True),
        ('min', True),
        ('nth', False),
        ('nunique', False),
        ('pct_change', False),
        ('prod', True),
        ('quantile', True),
        ('sem', True),
        ('skew', True),
        ('kurt', True),
        ('std', True),
        ('sum', True),
        ('var', True),
    ],
)
@pytest.mark.parametrize('numeric_only', [True, False, lib.no_default])
@pytest.mark.parametrize('keys', [['a1'], ['a1', 'a2']])
def test_numeric_only(
    kernel: str,
    has_arg: bool,
    numeric_only: Union[bool, lib.NoDefault],
    keys: List[str],
) -> None:
    df = DataFrame({'a1': [1, 1], 'a2': [2, 2], 'a3': [5, 6], 'b': 2 * [object]})
    args = get_groupby_method_args(kernel, df)
    kwargs = (
        {} if numeric_only is lib.no_default else {'numeric_only': numeric_only}
    )
    gb = df.groupby(keys)
    method = getattr(gb, kernel)
    if has_arg and numeric_only is True:
        if kernel == 'corrwith':
            warn = FutureWarning
            msg = 'DataFrameGroupBy.corrwith is deprecated'
        else:
            warn = None
            msg = ''
        with tm.assert_produces_warning(warn, match=msg):
            result = method(*args, **kwargs)
        assert 'b' not in result.columns
    elif kernel in ('first', 'last') or (
        kernel in ('any', 'all', 'bfill', 'ffill', 'nth', 'nunique')
        and numeric_only is lib.no_default
    ):
        result = method(*args, **kwargs)
        assert 'b' in result.columns
    elif has_arg:
        assert numeric_only is not True
        exception = NotImplementedError if kernel.startswith('cum') else TypeError
        msg = '|'.join(
            [
                'not allowed for this dtype',
                "cannot be performed against 'object' dtypes",
                'must be a string or a real number',
                'unsupported operand type',
                'function is not implemented for this dtype',
                re.escape(f'agg function failed [how->{kernel},dtype->object]'),
            ]
        )
        if kernel == 'quantile':
            msg = "dtype 'object' does not support operation 'quantile'"
        elif kernel == 'idxmin':
            msg = "'<' not supported between instances of 'type' and 'type'"
        elif kernel == 'idxmax':
            msg = "'>' not supported between instances of 'type' and 'type'"
        with pytest.raises(exception, match=msg):
            if kernel == 'corrwith':
                warn = FutureWarning
                msg = 'DataFrameGroupBy.corrwith is deprecated'
            else:
                warn = None
                msg = ''
            with tm.assert_produces_warning(warn, match=msg):
                method(*args, **kwargs)
    elif not has_arg and numeric_only is not lib.no_default:
        with pytest.raises(
            TypeError, match="got an unexpected keyword argument 'numeric_only'"
        ):
            method(*args, **kwargs)
    else:
        assert kernel in ('diff', 'pct_change')
        assert numeric_only is lib.no_default
        with pytest.raises(TypeError, match='unsupported operand type'):
            method(*args, **kwargs)


@pytest.mark.parametrize('dtype', [bool, int, float, object])
def test_deprecate_numeric_only_series(
    dtype: type, groupby_func: str, request: pytest.FixtureRequest
) -> None:
    grouper = [0, 0, 1]
    ser = Series([1, 0, 0], dtype=dtype)
    gb = ser.groupby(grouper)
    if groupby_func == 'corrwith':
        assert not hasattr(gb, groupby_func)
        return
    method = getattr(gb, groupby_func)
    expected_ser = Series([1, 0, 0])
    expected_gb = expected_ser.groupby(grouper)
    expected_method = getattr(expected_gb, groupby_func)
    args = get_groupby_method_args(groupby_func, ser)
    fails_on_numeric_object = (
        'corr',
        'cov',
        'cummax',
        'cummin',
        'cumprod',
        'cumsum',
        'quantile',
    )
    obj_result = (
        'first',
        'last',
        'nth',
        'bfill',
        'ffill',
        'shift',
        'sum',
        'diff',
        'pct_change',
        'var',
        'mean',
        'median',
        'min',
        'max',
        'prod',
        'skew',
        'kurt',
    )
    if groupby_func in fails_on_numeric_object and dtype is object:
        if groupby_func == 'quantile':
            msg = "dtype 'object' does not support operation 'quantile'"
        else:
            msg = 'is not supported for object dtype'
        with pytest.raises(TypeError, match=msg):
            method(*args)
    elif dtype is object:
        result = method(*args)
        expected = expected_method(*args)
        if groupby_func in obj_result:
            expected = expected.astype(object)
        tm.assert_series_equal(result, expected)
    has_numeric_only = (
        'first',
        'last',
        'max',
        'mean',
        'median',
        'min',
        'prod',
        'quantile',
        'sem',
        'skew',
        'kurt',
        'std',
        'sum',
        'var',
        'cummax',
        'cummin',
        'cumprod',
        'cumsum',
    )
    if groupby_func not in has_numeric_only:
        msg = "got an unexpected keyword argument 'numeric_only'"
        with pytest.raises(TypeError, match=msg):
            method(*args, numeric_only=True)
    elif dtype is object:
        msg = '|'.join(
            [
                'SeriesGroupBy.sem called with numeric_only=True and dtype object',
                'Series.skew does not allow numeric_only=True with non-numeric',
                'cum(sum|prod|min|max) is not supported for object dtype',
                'Cannot use numeric_only=True with SeriesGroupBy\\..* and non-numeric',
            ]
        )
        with pytest.raises(TypeError, match=msg):
            method(*args, numeric_only=True)
    elif dtype == bool and groupby_func == 'quantile':
        msg = 'Cannot use quantile with bool dtype'
        with pytest.raises(TypeError, match=msg):
            method(*args, numeric_only=False)
    else:
        result = method(*args, numeric_only=True)
        expected = method(*args, numeric_only=False)
        tm.assert_series_equal(result, expected)
