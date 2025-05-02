from __future__ import annotations
import datetime
from functools import partial
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Union, cast
import numpy as np
from pandas._libs.tslibs import Timedelta
import pandas._libs.window.aggregations as window_aggregations
from pandas.util._decorators import doc
from pandas.core.dtypes.common import is_datetime64_dtype, is_numeric_dtype
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.dtypes.missing import isna
from pandas.core import common
from pandas.core.arrays.datetimelike import dtype_to_unit
from pandas.core.indexers.objects import BaseIndexer, ExponentialMovingWindowIndexer, GroupbyIndexer
from pandas.core.util.numba_ import get_jit_arguments, maybe_use_numba
from pandas.core.window.common import zsqrt
from pandas.core.window.doc import _shared_docs, create_section_header, kwargs_numeric_only, numba_notes, template_header, template_returns, template_see_also, window_agg_numba_parameters
from pandas.core.window.numba_ import generate_numba_ewm_func, generate_numba_ewm_table_func
from pandas.core.window.online import EWMMeanState, generate_online_numba_ewma_func
from pandas.core.window.rolling import BaseWindow, BaseWindowGroupby
if TYPE_CHECKING:
    from pandas._typing import TimedeltaConvertibleTypes, npt
    from pandas import DataFrame, Series
    from pandas.core.generic import NDFrame

def get_center_of_mass(
    comass: Optional[float], 
    span: Optional[float], 
    halflife: Optional[float], 
    alpha: Optional[float]
) -> float:
    valid_count = common.count_not_none(comass, span, halflife, alpha)
    if valid_count > 1:
        raise ValueError('comass, span, halflife, and alpha are mutually exclusive')
    if comass is not None:
        if comass < 0:
            raise ValueError('comass must satisfy: comass >= 0')
    elif span is not None:
        if span < 1:
            raise ValueError('span must satisfy: span >= 1')
        comass = (span - 1) / 2
    elif halflife is not None:
        if halflife <= 0:
            raise ValueError('halflife must satisfy: halflife > 0')
        decay = 1 - np.exp(np.log(0.5) / halflife)
        comass = 1 / decay - 1
    elif alpha is not None:
        if alpha <= 0 or alpha > 1:
            raise ValueError('alpha must satisfy: 0 < alpha <= 1')
        comass = (1 - alpha) / alpha
    else:
        raise ValueError('Must pass one of comass, span, halflife, or alpha')
    return float(comass)

def _calculate_deltas(times: Union[np.ndarray, "Series"], halflife: Union[float, str, datetime.timedelta, np.timedelta64]) -> np.ndarray:
    """
    Return the diff of the times divided by the half-life. These values are used in
    the calculation of the ewm mean.

    Parameters
    ----------
    times : np.ndarray, Series
        Times corresponding to the observations. Must be monotonically increasing
        and ``datetime64[ns]`` dtype.
    halflife : float, str, timedelta, optional
        Half-life specifying the decay

    Returns
    -------
    np.ndarray
        Diff of the times divided by the half-life
    """
    unit = dtype_to_unit(times.dtype)
    if isinstance(times, ABCSeries):
        times = times._values
    _times = np.asarray(times.view(np.int64), dtype=np.float64)
    _halflife = float(Timedelta(halflife).as_unit(unit)._value)
    return np.diff(_times) / _halflife

class ExponentialMovingWindow(BaseWindow):
    _attributes = ['com', 'span', 'halflife', 'alpha', 'min_periods', 'adjust', 'ignore_na', 'times', 'method']

    def __init__(
        self,
        obj: Union["DataFrame", "Series"],
        com: Optional[float] = None,
        span: Optional[float] = None,
        halflife: Optional[Union[float, str, datetime.timedelta, np.timedelta64]] = None,
        alpha: Optional[float] = None,
        min_periods: int = 0,
        adjust: bool = True,
        ignore_na: bool = False,
        times: Optional[Union[np.ndarray, "Series"]] = None,
        method: str = 'single',
        *,
        selection=None
    ):
        super().__init__(obj=obj, min_periods=1 if min_periods is None else max(int(min_periods), 1), on=None, center=False, closed=None, method=method, selection=selection)
        self.com = com
        self.span = span
        self.halflife = halflife
        self.alpha = alpha
        self.adjust = adjust
        self.ignore_na = ignore_na
        self.times = times
        if self.times is not None:
            times_dtype = getattr(self.times, 'dtype', None)
            if not (is_datetime64_dtype(times_dtype) or isinstance(times_dtype, DatetimeTZDtype)):
                raise ValueError('times must be datetime64 dtype.')
            if len(self.times) != len(obj):
                raise ValueError('times must be the same length as the object.')
            if not isinstance(self.halflife, (str, datetime.timedelta, np.timedelta64)):
                raise ValueError('halflife must be a timedelta convertible object')
            if isna(self.times).any():
                raise ValueError('Cannot convert NaT values to integer')
            self._deltas = _calculate_deltas(self.times, self.halflife)
            if common.count_not_none(self.com, self.span, self.alpha) > 0:
                if not self.adjust:
                    raise NotImplementedError('None of com, span, or alpha can be specified if times is provided and adjust=False')
                self._com = get_center_of_mass(self.com, self.span, None, self.alpha)
            else:
                self._com = 1.0
        else:
            if self.halflife is not None and isinstance(self.halflife, (str, datetime.timedelta, np.timedelta64)):
                raise ValueError('halflife can only be a timedelta convertible argument if times is not None.')
            self._deltas = np.ones(max(self.obj.shape[0] - 1, 0), dtype=np.float64)
            self._com = get_center_of_mass(self.com, self.span, self.halflife, self.alpha)

    def _check_window_bounds(self, start: np.ndarray, end: np.ndarray, num_vals: int) -> None:
        pass

    def _get_window_indexer(self) -> ExponentialMovingWindowIndexer:
        """
        Return an indexer class that will compute the window start and end bounds
        """
        return ExponentialMovingWindowIndexer()

    def online(self, engine: str = 'numba', engine_kwargs: Optional[Dict[str, bool]] = None) -> "OnlineExponentialMovingWindow":
        """
        Return an ``OnlineExponentialMovingWindow`` object to calculate
        exponentially moving window aggregations in an online method.

        .. versionadded:: 1.3.0

        Parameters
        ----------
        engine: str, default ``'numba'``
            Execution engine to calculate online aggregations.
            Applies to all supported aggregation methods.

        engine_kwargs : dict, default None
            Applies to all supported aggregation methods.

            * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
              and ``parallel`` dictionary keys. The values must either be ``True`` or
              ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
              ``{{'nopython': True, 'nogil': False, 'parallel': False}}`` and will be
              applied to the function

        Returns
        -------
        OnlineExponentialMovingWindow
        """
        return OnlineExponentialMovingWindow(
            obj=self.obj,
            com=self.com,
            span=self.span,
            halflife=self.halflife,
            alpha=self.alpha,
            min_periods=self.min_periods,
            adjust=self.adjust,
            ignore_na=self.ignore_na,
            times=self.times,
            engine=engine,
            engine_kwargs=engine_kwargs,
            selection=self._selection
        )

    @doc(_shared_docs['aggregate'], see_also=dedent('\n        See Also\n        --------\n        pandas.DataFrame.rolling.aggregate\n        '), examples=dedent('\n        Examples\n        --------\n        >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})\n        >>> df\n           A  B  C\n        0  1  4  7\n        1  2  5  8\n        2  3  6  9\n\n        >>> df.ewm(alpha=0.5).mean()\n                  A         B         C\n        0  1.000000  4.000000  7.000000\n        1  1.666667  4.666667  7.666667\n        2  2.428571  5.428571  8.428571\n        '), klass='Series/Dataframe', axis='')
    def aggregate(self, func: Optional[Union[str, Callable]] = None, *args: Any, **kwargs: Any) -> Union["DataFrame", "Series"]:
        return super().aggregate(func, *args, **kwargs)
    agg = aggregate

    @doc(template_header, create_section_header('Parameters'), kwargs_numeric_only, window_agg_numba_parameters(), create_section_header('Returns'), template_returns, create_section_header('See Also'), template_see_also, create_section_header('Notes'), numba_notes, create_section_header('Examples'), dedent('        >>> ser = pd.Series([1, 2, 3, 4])\n        >>> ser.ewm(alpha=.2).mean()\n        0    1.000000\n        1    1.555556\n        2    2.147541\n        3    2.775068\n        dtype: float64\n        '), window_method='ewm', aggregation_description='(exponential weighted moment) mean', agg_method='mean')
    def mean(self, numeric_only: bool = False, engine: Optional[str] = None, engine_kwargs: Optional[Dict[str, bool]] = None) -> Union["DataFrame", "Series"]:
        if maybe_use_numba(engine):
            if self.method == 'single':
                func = generate_numba_ewm_func
            else:
                func = generate_numba_ewm_table_func
            ewm_func = func(**get_jit_arguments(engine_kwargs), com=self._com, adjust=self.adjust, ignore_na=self.ignore_na, deltas=tuple(self._deltas), normalize=True)
            return self._apply(ewm_func, name='mean')
        elif engine in ('cython', None):
            if engine_kwargs is not None:
                raise ValueError('cython engine does not accept engine_kwargs')
            deltas = None if self.times is None else self._deltas
            window_func = partial(window_aggregations.ewm, com=self._com, adjust=self.adjust, ignore_na=self.ignore_na, deltas=deltas, normalize=True)
            return self._apply(window_func, name='mean', numeric_only=numeric_only)
        else:
            raise ValueError("engine must be either 'numba' or 'cython'")

    @doc(template_header, create_section_header('Parameters'), kwargs_numeric_only, window_agg_numba_parameters(), create_section_header('Returns'), template_returns, create_section_header('See Also'), template_see_also, create_section_header('Notes'), numba_notes, create_section_header('Examples'), dedent('        >>> ser = pd.Series([1, 2, 3, 4])\n        >>> ser.ewm(alpha=.2).sum()\n        0    1.000\n        1    2.800\n        2    5.240\n        3    8.192\n        dtype: float64\n        '), window_method='ewm', aggregation_description='(exponential weighted moment) sum', agg_method='sum')
    def sum(self, numeric_only: bool = False, engine: Optional[str] = None, engine_kwargs: Optional[Dict[str, bool]] = None) -> Union["DataFrame", "Series"]:
        if not self.adjust:
            raise NotImplementedError('sum is not implemented with adjust=False')
        if self.times is not None:
            raise NotImplementedError('sum is not implemented with times')
        if maybe_use_numba(engine):
            if self.method == 'single':
                func = generate_numba_ewm_func
            else:
                func = generate_numba_ewm_table_func
            ewm_func = func(**get_jit_arguments(engine_kwargs), com=self._com, adjust=self.adjust, ignore_na=self.ignore_na, deltas=tuple(self._deltas), normalize=False)
            return self._apply(ewm_func, name='sum')
        elif engine in ('cython', None):
            if engine_kwargs is not None:
                raise ValueError('cython engine does not accept engine_kwargs')
            deltas = None if self.times is None else self._deltas
            window_func = partial(window_aggregations.ewm, com=self._com, adjust=self.adjust, ignore_na=self.ignore_na, deltas=deltas, normalize=False)
            return self._apply(window_func, name='sum', numeric_only=numeric_only)
        else:
            raise ValueError("engine must be either 'numba' or 'cython'")

    @doc(template_header, create_section_header('Parameters'), dedent('        bias : bool, default False\n            Use a standard estimation bias correction.\n        '), kwargs_numeric_only, create_section_header('Returns'), template_returns, create_section_header('See Also'), template_see_also, create_section_header('Examples'), dedent('        >>> ser = pd.Series([1, 2, 3, 4])\n        >>> ser.ewm(alpha=.2).std()\n        0         NaN\n        1    0.707107\n        2    0.995893\n        3    1.277320\n        dtype: float64\n        '), window_method='ewm', aggregation_description='(exponential weighted moment) standard deviation', agg_method='std')
    def std(self, bias: bool = False, numeric_only: bool = False) -> Union["DataFrame", "Series"]:
        if numeric_only and self._selected_obj.ndim == 1 and (not is_numeric_dtype(self._selected_obj.dtype)):
            raise NotImplementedError(f'{type(self).__name__}.std does not implement numeric_only')
        if self.times is not None:
            raise NotImplementedError('std is not implemented with times')
        return zsqrt(self.var(bias=bias, numeric_only=numeric_only))

    @doc(template_header, create_section_header('Parameters'), dedent('        bias : bool, default False\n            Use a standard estimation bias correction.\n        '), kwargs_numeric_only, create_section_header('Returns'), template_returns, create_section_header('See Also'), template_see_also, create_section_header('Examples'), dedent('        >>> ser = pd.Series([1, 2, 3, 4])\n        >>> ser.ewm(alpha=.2).var()\n        0         NaN\n        1    0.500000\n        2    0.991803\n        3    1.631547\n        dtype: float64\n        '), window_method='ewm', aggregation_description='(exponential weighted moment) variance', agg_method='var')
    def var(self, bias: bool = False, numeric_only: bool = False) -> Union["DataFrame", "Series"]:
        if self.times is not None:
            raise NotImplementedError('var is not implemented with times')
        window_func = window_aggregations.ewmcov
        wfunc = partial(window_func, com=self._com, adjust=self.adjust, ignore_na=self.ignore_na, bias=bias)

        def var_func(values: np.ndarray, begin: np.ndarray, end: np.ndarray, min_periods: int) -> np.ndarray:
            return wfunc(values, begin, end, min_periods, values)
        return self._apply(var_func, name='var', numeric_only=numeric_only)

    @doc(template_header, create_section_header('Parameters'), dedent('        other : Series or DataFrame , optional\n            If not supplied then will default to self and produce pairwise\n            output.\n        pairwise : bool, default None\n            If False then only matching columns between self and other will be\n            used and the output will be a DataFrame.\n            If True then all pairwise combinations will be calculated and the\n            output will be a MultiIndex DataFrame in the case of DataFrame\n            inputs. In the case of missing elements, only complete pairwise\n            observations will be used.\n        bias : bool, default False\n            Use a standard estimation bias correction.\n        '), kwargs_numeric_only, create_section_header('Returns'), template_returns, create_section_header('See Also'), template_see_also, create_section_header('Examples'), dedent('        >>> ser1 = pd.Series([1, 2, 3, 4])\n        >>> ser2 = pd.Series([10