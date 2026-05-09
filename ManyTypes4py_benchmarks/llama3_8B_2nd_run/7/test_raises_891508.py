import datetime
import re
import numpy as np
import pytest
from pandas import Categorical, DataFrame, Grouper, Series
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args

@pytest.fixture(params=['a', ['a'], ['a', 'b'], Grouper(key='a'), lambda x: x % 2, [0, 0, 0, 1, 2, 2, 2, 3, 3], np.array([0, 0, 0, 1, 2, 2, 2, 3, 3]), dict(zip(range(9), [0, 0, 0, 1, 2, 2, 2, 3, 3])), Series([1, 1, 1, 1, 1, 2, 2, 2, 2]), [Series([1, 1, 1, 1, 1, 2, 2, 2, 2]), Series([3, 3, 4, 4, 4, 4, 4, 3, 3])]])
def by(request: pytest.Parametrize) -> object:
    return request.param

@pytest.fixture(params=[True, False])
def groupby_series(request: pytest.Parametrize) -> bool:
    return request.param

@pytest.fixture
def df_with_string_col() -> DataFrame:
    df = DataFrame({'a': [1, 1, 1, 1, 1, 2, 2, 2, 2], 'b': [3, 3, 4, 4, 4, 4, 4, 3, 3], 'c': range(9), 'd': list('xyzwtyuio')})
    return df

@pytest.fixture
def df_with_datetime_col() -> DataFrame:
    df = DataFrame({'a': [1, 1, 1, 1, 1, 2, 2, 2, 2], 'b': [3, 3, 4, 4, 4, 4, 4, 3, 3], 'c': range(9), 'd': datetime.datetime(2005, 1, 1, 10, 30, 23, 540000)})
    return df

@pytest.fixture
def df_with_cat_col() -> DataFrame:
    df = DataFrame({'a': [1, 1, 1, 1, 1, 2, 2, 2, 2], 'b': [3, 3, 4, 4, 4, 4, 4, 3, 3], 'c': range(9), 'd': Categorical(['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'c'], categories=['a', 'b', 'c', 'd'], ordered=True)})
    return df

def _call_and_check(klass: type, msg: str, how: str, gb: DataFrameGroupBy, groupby_func: str, args: tuple, warn_msg: str = '') -> None:
    warn_klass = None if warn_msg == '' else FutureWarning
    with tm.assert_produces_warning(warn_klass, match=warn_msg, check_stacklevel=False):
        if klass is None:
            if how == 'method':
                getattr(gb, groupby_func)(*args)
            elif how == 'agg':
                gb.agg(groupby_func, *args)
            else:
                gb.transform(groupby_func, *args)
        else:
            with pytest.raises(klass, match=msg):
                if how == 'method':
                    getattr(gb, groupby_func)(*args)
                elif how == 'agg':
                    gb.agg(groupby_func, *args)
                else:
                    gb.transform(groupby_func, *args)

@pytest.mark.parametrize('how', ['method', 'agg', 'transform'])
def test_groupby_raises_string(how: str, by: object, groupby_series: bool, groupby_func: str, df_with_string_col: DataFrame, using_infer_string: bool) -> None:
    df = df_with_string_col
    args = get_groupby_method_args(groupby_func, df)
    gb = df.groupby(by=by)
    if groupby_series:
        gb = gb['d']
        if groupby_func == 'corrwith':
            assert not hasattr(gb, 'corrwith')
            return
    klass, msg = {'all': (None, ''), 'any': (None, ''), 'bfill': (None, ''), 'corrwith': (TypeError, 'Could not convert string .* to numeric|Cannot perform reduction 'mean' with string dtype'), 'count': (None, ''), 'cumcount': (None, ''), 'cummax': ((NotImplementedError, TypeError), '(function|cummax) is not (implemented|supported) for (this|object) dtype'), 'cummin': ((NotImplementedError, TypeError), '(function|cummin) is not (implemented|supported) for (this|object) dtype'), 'cumprod': ((NotImplementedError, TypeError), '(function|cumprod) is not (implemented|supported) for (this|object) dtype'), 'cumsum': ((NotImplementedError, TypeError), '(function|cumsum) is not (implemented|supported) for (this|object) dtype'), 'diff': (TypeError, 'unsupported operand type'), 'ffill': (None, ''), 'first': (None, ''), 'idxmax': (None, ''), 'idxmin': (None, ''), 'last': (None, ''), 'max': (None, ''), 'mean': (TypeError, re.escape('agg function failed [how->mean,dtype->object]')), 'median': (TypeError, re.escape('agg function failed [how->median,dtype->object]')), 'min': (None, ''), 'ngroup': (None, ''), 'nunique': (None, ''), 'pct_change': (TypeError, 'unsupported operand type'), 'prod': (TypeError, re.escape('agg function failed [how->prod,dtype->object]')), 'quantile': (TypeError, "dtype 'object' does not support operation 'quantile'"), 'rank': (None, ''), 'sem': (ValueError, 'could not convert string to float'), 'shift': (None, ''), 'size': (None, ''), 'skew': (ValueError, 'could not convert string to float'), 'kurt': (ValueError, 'could not convert string to float'), 'std': (ValueError, 'could not convert string to float'), 'sum': (None, ''), 'var': (TypeError, re.escape('agg function failed [how->var,dtype->'))}[groupby_func]
    if using_infer_string:
        if groupby_func in ['prod', 'mean', 'median', 'cumsum', 'cumprod', 'std', 'sem', 'var', 'skew', 'kurt', 'quantile']:
            msg = f"dtype 'str' does not support operation '{groupby_func}'"
            if groupby_func in ['sem', 'std', 'skew', 'kurt']:
                klass = TypeError
        elif groupby_func == 'pct_change' and df['d'].dtype.storage == 'pyarrow':
            msg = "operation 'truediv' not supported for dtype 'str' with dtype 'str'"
        elif groupby_func in ['cummin', 'cummax']:
            msg = msg.replace('object', 'str')
        elif groupby_func == 'corrwith':
            msg = "Cannot perform reduction 'mean' with string dtype"
    if groupby_func == 'corrwith':
        warn_msg = 'DataFrameGroupBy.corrwith is deprecated'
    else:
        warn_msg = ''
    _call_and_check(klass, msg, how, gb, groupby_func, args, warn_msg)
