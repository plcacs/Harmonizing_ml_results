from __future__ import annotations
import datetime
from functools import partial
from textwrap import dedent
from typing import TYPE_CHECKING, Optional, Union
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
    halflife: Optional[Union[float, str, datetime.timedelta]], 
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

def _calculate_deltas(
    times: Union[np.ndarray, Series], 
    halflife: Union[float, str, datetime.timedelta]
) -> np.ndarray:
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
        obj: NDFrame, 
        com: Optional[float] = None, 
        span: Optional[float] = None, 
        halflife: Optional[Union[float, str, datetime.timedelta]] = None, 
        alpha: Optional[float] = None, 
        min_periods: int = 0, 
        adjust: bool = True, 
        ignore_na: bool = False, 
        times: Optional[Union[np.ndarray, Series]] = None, 
        method: str = 'single', 
        *, 
        selection: Optional[str] = None
    ) -> None:
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

    def _check_window_bounds(self, start: int, end: int, num_vals: int) -> None:
        pass

    def _get_window_indexer(self) -> BaseIndexer:
        return ExponentialMovingWindowIndexer()

    def online(
        self, 
        engine: str = 'numba', 
        engine_kwargs: Optional[dict] = None
    ) -> OnlineExponentialMovingWindow:
        return OnlineExponentialMovingWindow(obj=self.obj, com=self.com, span=self.span, halflife=self.halflife, alpha=self.alpha, min_periods=self.min_periods, adjust=self.adjust, ignore_na=self.ignore_na, times=self.times, engine=engine, engine_kwargs=engine_kwargs, selection=self._selection)

    @doc(_shared_docs['aggregate'], see_also=dedent('\n        See Also\n        --------\n        pandas.DataFrame.rolling.aggregate\n        '), examples=dedent('\n        Examples\n        --------\n        >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})\n        >>> df\n           A  B  C\n        0  1  4  7\n        1  2  5  8\n        2  3  6  9\n\n        >>> df.ewm(alpha=0.5).mean()\n                  A         B         C\n        0  1.000000  4.000000  7.000000\n        1  1.666667  4.666667  7.666667\n        2  2.428571  5.428571  8.428571\n        '), klass='Series/Dataframe', axis='')
    def aggregate(self, func=None, *args, **kwargs):
        return super().aggregate(func, *args, **kwargs)
    agg = aggregate

    @doc(template_header, create_section_header('Parameters'), kwargs_numeric_only, window_agg_numba_parameters(), create_section_header('Returns'), template_returns, create_section_header('See Also'), template_see_also, create_section_header('Notes'), numba_notes, create_section_header('Examples'), dedent('        >>> ser = pd.Series([1, 2, 3, 4])\n        >>> ser.ewm(alpha=.2).mean()\n        0    1.000000\n        1    1.555556\n        2    2.147541\n        3    2.775068\n        dtype: float64\n        '), window_method='ewm', aggregation_description='(exponential weighted moment) mean', agg_method='mean')
    def mean(
        self, 
        numeric_only: bool = False, 
        engine: Optional[str] = None, 
        engine_kwargs: Optional[dict] = None
    ) -> NDFrame:
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
    def sum(
        self, 
        numeric_only: bool = False, 
        engine: Optional[str] = None, 
        engine_kwargs: Optional[dict] = None
    ) -> NDFrame:
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
    def std(
        self, 
        bias: bool = False, 
        numeric_only: bool = False
    ) -> NDFrame:
        if numeric_only and self._selected_obj.ndim == 1 and (not is_numeric_dtype(self._selected_obj.dtype)):
            raise NotImplementedError(f'{type(self).__name__}.std does not implement numeric_only')
        if self.times is not None:
            raise NotImplementedError('std is not implemented with times')
        return zsqrt(self.var(bias=bias, numeric_only=numeric_only))

    @doc(template_header, create_section_header('Parameters'), dedent('        bias : bool, default False\n            Use a standard estimation bias correction.\n        '), kwargs_numeric_only, create_section_header('Returns'), template_returns, create_section_header('See Also'), template_see_also, create_section_header('Examples'), dedent('        >>> ser = pd.Series([1, 2, 3, 4])\n        >>> ser.ewm(alpha=.2).var()\n        0         NaN\n        1    0.500000\n        2    0.991803\n        3    1.631547\n        dtype: float64\n        '), window_method='ewm', aggregation_description='(exponential weighted moment) variance', agg_method='var')
    def var(
        self, 
        bias: bool = False, 
        numeric_only: bool = False
    ) -> NDFrame:
        if self.times is not None:
            raise NotImplementedError('var is not implemented with times')
        window_func = window_aggregations.ewmcov
        wfunc = partial(window_func, com=self._com, adjust=self.adjust, ignore_na=self.ignore_na, bias=bias)

        def var_func(values: np.ndarray, begin: int, end: int, min_periods: int) -> np.ndarray:
            return wfunc(values, begin, end, min_periods, values)
        return self._apply(var_func, name='var', numeric_only=numeric_only)

    @doc(template_header, create_section_header('Parameters'), dedent('        other : Series or DataFrame , optional\n            If not supplied then will default to self and produce pairwise\n            output.\n        pairwise : bool, default None\n            If False then only matching columns between self and other will be\n            used and the output will be a DataFrame.\n            If True then all pairwise combinations will be calculated and the\n            output will be a MultiIndex DataFrame in the case of DataFrame\n            inputs. In the case of missing elements, only complete pairwise\n            observations will be used.\n        bias : bool, default False\n            Use a standard estimation bias correction.\n        '), kwargs_numeric_only, create_section_header('Returns'), template_returns, create_section_header('See Also'), template_see_also, create_section_header('Examples'), dedent('        >>> ser1 = pd.Series([1, 2, 3, 4])\n        >>> ser2 = pd.Series([10, 11, 13, 16])\n        >>> ser1.ewm(alpha=.2).cov(ser2)\n        0         NaN\n        1    0.500000\n        2    1.524590\n        3    3.408836\n        dtype: float64\n        '), window_method='ewm', aggregation_description='(exponential weighted moment) sample covariance', agg_method='cov')
    def cov(
        self, 
        other: Optional[Union[Series, DataFrame]] = None, 
        pairwise: Optional[bool] = None, 
        bias: bool = False, 
        numeric_only: bool = False
    ) -> NDFrame:
        if self.times is not None:
            raise NotImplementedError('cov is not implemented with times')
        from pandas import Series
        self._validate_numeric_only('cov', numeric_only)

        def cov_func(x: Series, y: Series) -> Series:
            x_array = self._prep_values(x)
            y_array = self._prep_values(y)
            window_indexer = self._get_window_indexer()
            min_periods = self.min_periods if self.min_periods is not None else window_indexer.window_size
            start, end = window_indexer.get_window_bounds(num_values=len(x_array), min_periods=min_periods, center=self.center, closed=self.closed, step=self.step)
            result = window_aggregations.ewmcov(x_array, start, end, self.min_periods, y_array, self._com, self.adjust, self.ignore_na, bias)
            return Series(result, index=x.index, name=x.name, copy=False)
        return self._apply_pairwise(self._selected_obj, other, pairwise, cov_func, numeric_only)

    @doc(template_header, create_section_header('Parameters'), dedent('        other : Series or DataFrame, optional\n            If not supplied then will default to self and produce pairwise\n            output.\n        pairwise : bool, default None\n            If False then only matching columns between self and other will be\n            used and the output will be a DataFrame.\n            If True then all pairwise combinations will be calculated and the\n            output will be a MultiIndex DataFrame in the case of DataFrame\n            inputs. In the case of missing elements, only complete pairwise\n            observations will be used.\n        '), kwargs_numeric_only, create_section_header('Returns'), template_returns, create_section_header('See Also'), template_see_also, create_section_header('Examples'), dedent('        >>> ser1 = pd.Series([1, 2, 3, 4])\n        >>> ser2 = pd.Series([10, 11, 13, 16])\n        >>> ser1.ewm(alpha=.2).corr(ser2)\n        0         NaN\n        1    1.000000\n        2    0.982821\n        3    0.977802\n        dtype: float64\n        '), window_method='ewm', aggregation_description='(exponential weighted moment) sample correlation', agg_method='corr')
    def corr(
        self, 
        other: Optional[Union[Series, DataFrame]] = None, 
        pairwise: Optional[bool] = None, 
        numeric_only: bool = False
    ) -> NDFrame:
        if self.times is not None:
            raise NotImplementedError('corr is not implemented with times')
        from pandas import Series
        self._validate_numeric_only('corr', numeric_only)

        def cov_func(x: Series, y: Series) -> Series:
            x_array = self._prep_values(x)
            y_array = self._prep_values(y)
            window_indexer = self._get_window_indexer()
            min_periods = self.min_periods if self.min_periods is not None else window_indexer.window_size
            start, end = window_indexer.get_window_bounds(num_values=len(x_array), min_periods=min_periods, center=self.center, closed=self.closed, step=self.step)

            def _cov(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
                return window_aggregations.ewmcov(X, start, end, min_periods, Y, self._com, self.adjust, self.ignore_na, True)
            with np.errstate(all='ignore'):
                cov = _cov(x_array, y_array)
                x_var = _cov(x_array, x_array)
                y_var = _cov(y_array, y_array)
                result = cov / zsqrt(x_var * y_var)
            return Series(result, index=x.index, name=x.name, copy=False)
        return self._apply_pairwise(self._selected_obj, other, pairwise, cov_func, numeric_only)

class ExponentialMovingWindowGroupby(BaseWindowGroupby, ExponentialMovingWindow):
    _attributes = ExponentialMovingWindow._attributes + BaseWindowGroupby._attributes

    def __init__(
        self, 
        obj: NDFrame, 
        *args, 
        _grouper=None, 
        **kwargs
    ) -> None:
        super().__init__(obj, *args, _grouper=_grouper, **kwargs)
        if not obj.empty and self.times is not None:
            groupby_order = np.concatenate(list(self._grouper.indices.values()))
            self._deltas = _calculate_deltas(self.times.take(groupby_order), self.halflife)

    def _get_window_indexer(self) -> GroupbyIndexer:
        window_indexer = GroupbyIndexer(groupby_indices=self._grouper.indices, window_indexer=ExponentialMovingWindowIndexer)
        return window_indexer

class OnlineExponentialMovingWindow(ExponentialMovingWindow):

    def __init__(
        self, 
        obj: NDFrame, 
        com: Optional[float] = None, 
        span: Optional[float] = None, 
        halflife: Optional[Union[float, str, datetime.timedelta]] = None, 
        alpha: Optional[float] = None, 
        min_periods: int = 0, 
        adjust: bool = True, 
        ignore_na: bool = False, 
        times: Optional[Union[np.ndarray, Series]] = None, 
        engine: str = 'numba', 
        engine_kwargs: Optional[dict] = None, 
        *, 
        selection: Optional[str] = None
    ) -> None:
        if times is not None:
            raise NotImplementedError('times is not implemented with online operations.')
        super().__init__(obj=obj, com=com, span=span, halflife=halflife, alpha=alpha, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na, times=times, selection=selection)
        self._mean = EWMMeanState(self._com, self.adjust, self.ignore_na, obj.shape)
        if maybe_use_numba(engine):
            self.engine = engine
            self.engine_kwargs = engine_kwargs
        else:
            raise ValueError("'numba' is the only supported engine")

    def reset(self) -> None:
        self._mean.reset()

    def aggregate(self, func=None, *args, **kwargs):
        raise NotImplementedError('aggregate is not implemented.')

    def std(self, bias=False, *args, **kwargs):
        raise NotImplementedError('std is not implemented.')

    def corr(self, other=None, pairwise=None, numeric_only=False):
        raise NotImplementedError('corr is not implemented.')

    def cov(self, other=None, pairwise=None, bias=False, numeric_only=False):
        raise NotImplementedError('cov is not implemented.')

    def var(self, bias=False, numeric_only=False):
        raise NotImplementedError('var is not implemented.')

    def mean(
        self, 
        *args, 
        update: Optional[Union[DataFrame, Series]] = None, 
        update_times: Optional[Union[Series, np.ndarray]] = None, 
        **kwargs
    ) -> Union[DataFrame, Series]:
        result_kwargs = {}
        is_frame = self._selected_obj.ndim == 2
        if update_times is not None:
            raise NotImplementedError('update_times is not implemented.')
        update_deltas = np.ones(max(self._selected_obj.shape[-1] - 1, 0), dtype=np.float64)
        if update is not None:
            if self._mean.last_ewm is None:
                raise ValueError('Must call mean with update=None first before passing update')
            result_from = 1
            result_kwargs['index'] = update.index
            if is_frame:
                last_value = self._mean.last_ewm[np.newaxis, :]
                result_kwargs['columns'] = update.columns
            else:
                last_value = self._mean.last_ewm
                result_kwargs['name'] = update.name
            np_array = np.concatenate((last_value, update.to_numpy()))
        else:
            result_from = 0
            result_kwargs['index'] = self._selected_obj.index
            if is_frame:
                result_kwargs['columns'] = self._selected_obj.columns
            else:
                result_kwargs['name'] = self._selected_obj.name
            np_array = self._selected_obj.astype(np.float64).to_numpy()
        ewma_func = generate_online_numba_ewma_func(**get_jit_arguments(self.engine_kwargs))
        result = self._mean.run_ewm(np_array if is_frame else np_array[:, np.newaxis], update_deltas, self.min_periods, ewma_func)
        if not is_frame:
            result = result.squeeze()
        result = result[result_from:]
        result = self._selected_obj._constructor(result, **result_kwargs)
        return result
