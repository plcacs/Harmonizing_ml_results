from __future__ import annotations
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Literal, final, overload, Union, Optional, Dict, Tuple, Sequence
from pandas.util._decorators import Appender, Substitution, doc
from pandas.core.indexers.objects import BaseIndexer, ExpandingIndexer, GroupbyIndexer
from pandas.core.window.doc import _shared_docs, create_section_header, kwargs_numeric_only, numba_notes, template_header, template_pipe, template_returns, template_see_also, window_agg_numba_parameters, window_apply_parameters
from pandas.core.window.rolling import BaseWindowGroupby, RollingAndExpandingMixin

if TYPE_CHECKING:
    from collections.abc import Callable
    from pandas._typing import Concatenate, P, QuantileInterpolation, Self, T, WindowingRankType
    from pandas import DataFrame, Series
    from pandas.core.generic import NDFrame

class Expanding(RollingAndExpandingMixin):
    _attributes: list[str] = ['min_periods', 'method']

    def __init__(
        self,
        obj: Union[DataFrame, Series],
        min_periods: int = 1,
        method: str = 'single',
        selection: Optional[Sequence[str]] = None
    ) -> None:
        super().__init__(obj=obj, min_periods=min_periods, method=method, selection=selection)

    def _get_window_indexer(self) -> ExpandingIndexer:
        return ExpandingIndexer()

    @doc(_shared_docs['aggregate'], see_also=dedent('\n        See Also\n        --------\n        DataFrame.aggregate : Similar DataFrame method.\n        Series.aggregate : Similar Series method.\n        '), examples=dedent('\n        Examples\n        --------\n        >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})\n        >>> df\n           A  B  C\n        0  1  4  7\n        1  2  5  8\n        2  3  6  9\n\n        >>> df.ewm(alpha=0.5).mean()\n                  A         B         C\n        0  1.000000  4.000000  7.000000\n        1  1.666667  4.666667  7.666667\n        2  2.428571  5.428571  8.428571\n        '), klass='Series/Dataframe', axis='')
    def aggregate(
        self,
        func: Optional[Union[str, Callable, list[Union[str, Callable]]]] = None,
        *args: Any,
        **kwargs: Any
    ) -> Union[DataFrame, Series]:
        return super().aggregate(func, *args, **kwargs)

    agg: Callable[..., Union[DataFrame, Series]] = aggregate

    @doc(template_header, create_section_header('Parameters'), kwargs_numeric_only, create_section_header('Returns'), template_returns, create_section_header('See Also'), template_see_also, create_section_header('Examples'), dedent("        >>> ser = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])\n        >>> ser.expanding().count()\n        a    1.0\n        b    2.0\n        c    3.0\n        d    4.0\n        dtype: float64\n        "), window_method='expanding', aggregation_description='count of non NaN observations', agg_method='count')
    def count(self, numeric_only: bool = False) -> Union[DataFrame, Series]:
        return super().count(numeric_only=numeric_only)

    @doc(template_header, create_section_header('Parameters'), window_apply_parameters, create_section_header('Returns'), template_returns, create_section_header('See Also'), template_see_also, create_section_header('Examples'), dedent("        >>> ser = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])\n        >>> ser.expanding().apply(lambda s: s.max() - 2 * s.min())\n        a   -1.0\n        b    0.0\n        c    1.0\n        d    2.0\n        dtype: float64\n        "), window_method='expanding', aggregation_description='custom aggregation function', agg_method='apply')
    def apply(
        self,
        func: Callable,
        raw: bool = False,
        engine: Optional[str] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        args: Optional[Tuple[Any, ...]] = None,
        kwargs: Optional[Dict[str, Any]] = None
    ) -> Union[DataFrame, Series]:
        return super().apply(func, raw=raw, engine=engine, engine_kwargs=engine_kwargs, args=args, kwargs=kwargs)

    @overload
    def pipe(
        self,
        func: Callable[Concatenate[Union[DataFrame, Series], P], T],
        *args: P.args,
        **kwargs: P.kwargs
    ) -> T: ...

    @overload
    def pipe(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any
    ) -> T: ...

    @final
    @Substitution(klass='Expanding', examples="\n    >>> df = pd.DataFrame({'A': [1, 2, 3, 4]},\n    ...                   index=pd.date_range('2012-08-02', periods=4))\n    >>> df\n                A\n    2012-08-02  1\n    2012-08-03  2\n    2012-08-04  3\n    2012-08-05  4\n\n    To get the difference between each expanding window's maximum and minimum\n    value in one pass, you can do\n\n    >>> df.expanding().pipe(lambda x: x.max() - x.min())\n                  A\n    2012-08-02  0.0\n    2012-08-03  1.0\n    2012-08-04  2.0\n    2012-08-05  3.0")
    @Appender(template_pipe)
    def pipe(
        self,
        func: Union[Callable[Concatenate[Union[DataFrame, Series], P], T], Callable[..., T]],
        *args: Any,
        **kwargs: Any
    ) -> T:
        return super().pipe(func, *args, **kwargs)

    @doc(template_header, create_section_header('Parameters'), kwargs_numeric_only, window_agg_numba_parameters(), create_section_header('Returns'), template_returns, create_section_header('See Also'), template_see_also, create_section_header('Notes'), numba_notes, create_section_header('Examples'), dedent("        >>> ser = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])\n        >>> ser.expanding().sum()\n        a     1.0\n        b     3.0\n        c     6.0\n        d    10.0\n        dtype: float64\n        "), window_method='expanding', aggregation_description='sum', agg_method='sum')
    def sum(
        self,
        numeric_only: bool = False,
        engine: Optional[str] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None
    ) -> Union[DataFrame, Series]:
        return super().sum(numeric_only=numeric_only, engine=engine, engine_kwargs=engine_kwargs)

    @doc(template_header, create_section_header('Parameters'), kwargs_numeric_only, window_agg_numba_parameters(), create_section_header('Returns'), template_returns, create_section_header('See Also'), template_see_also, create_section_header('Notes'), numba_notes, create_section_header('Examples'), dedent("        >>> ser = pd.Series([3, 2, 1, 4], index=['a', 'b', 'c', 'd'])\n        >>> ser.expanding().max()\n        a    3.0\n        b    3.0\n        c    3.0\n        d    4.0\n        dtype: float64\n        "), window_method='expanding', aggregation_description='maximum', agg_method='max')
    def max(
        self,
        numeric_only: bool = False,
        engine: Optional[str] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None
    ) -> Union[DataFrame, Series]:
        return super().max(numeric_only=numeric_only, engine=engine, engine_kwargs=engine_kwargs)

    @doc(template_header, create_section_header('Parameters'), kwargs_numeric_only, window_agg_numba_parameters(), create_section_header('Returns'), template_returns, create_section_header('See Also'), template_see_also, create_section_header('Notes'), numba_notes, create_section_header('Examples'), dedent("        >>> ser = pd.Series([2, 3, 4, 1], index=['a', 'b', 'c', 'd'])\n        >>> ser.expanding().min()\n        a    2.0\n        b    2.0\n        c    2.0\n        d    1.0\n        dtype: float64\n        "), window_method='expanding', aggregation_description='minimum', agg_method='min')
    def min(
        self,
        numeric_only: bool = False,
        engine: Optional[str] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None
    ) -> Union[DataFrame, Series]:
        return super().min(numeric_only=numeric_only, engine=engine, engine_kwargs=engine_kwargs)

    @doc(template_header, create_section_header('Parameters'), kwargs_numeric_only, window_agg_numba_parameters(), create_section_header('Returns'), template_returns, create_section_header('See Also'), template_see_also, create_section_header('Notes'), numba_notes, create_section_header('Examples'), dedent("        >>> ser = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])\n        >>> ser.expanding().mean()\n        a    1.0\n        b    1.5\n        c    2.0\n        d    2.5\n        dtype: float64\n        "), window_method='expanding', aggregation_description='mean', agg_method='mean')
    def mean(
        self,
        numeric_only: bool = False,
        engine: Optional[str] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None
    ) -> Union[DataFrame, Series]:
        return super().mean(numeric_only=numeric_only, engine=engine, engine_kwargs=engine_kwargs)

    @doc(template_header, create_section_header('Parameters'), kwargs_numeric_only, window_agg_numba_parameters(), create_section_header('Returns'), template_returns, create_section_header('See Also'), template_see_also, create_section_header('Notes'), numba_notes, create_section_header('Examples'), dedent("        >>> ser = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])\n        >>> ser.expanding().median()\n        a    1.0\n        b    1.5\n        c    2.0\n        d    2.5\n        dtype: float64\n        "), window_method='expanding', aggregation_description='median', agg_method='median')
    def median(
        self,
        numeric_only: bool = False,
        engine: Optional[str] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None
    ) -> Union[DataFrame, Series]:
        return super().median(numeric_only=numeric_only, engine=engine, engine_kwargs=engine_kwargs)

    @doc(template_header, create_section_header('Parameters'), dedent('\n        ddof : int, default 1\n            Delta Degrees of Freedom.  The divisor used in calculations\n            is ``N - ddof``, where ``N`` represents the number of elements.\n\n        ').replace('\n', '', 1), kwargs_numeric_only, window_agg_numba_parameters('1.4'), create_section_header('Returns'), template_returns, create_section_header('See Also'), 'numpy.std : Equivalent method for NumPy array.\n', template_see_also, create_section_header('Notes'), dedent('\n        The default ``ddof`` of 1 used in :meth:`Series.std` is different\n        than the default ``ddof`` of 0 in :func:`numpy.std`.\n\n        A minimum of one period is required for the rolling calculation.\n\n        ').replace('\n', '', 1), create_section_header('Examples'), dedent('\n        >>> s = pd.Series([5, 5, 6, 7, 5, 5, 5])\n\n        >>> s.expanding(3).std()\n        0         NaN\n        1         NaN\n        2    0.577350\n        3    0.957427\n        4    0.894427\n        5    0.836660\n        6    0.786796\n        dtype: float64\n        ').replace('\n', '', 1), window_method='expanding', aggregation_description='standard deviation', agg_method='std')
    def std(
        self,
        ddof: int = 1,
        numeric_only: bool = False,
        engine: Optional[str] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None
    ) -> Union[DataFrame, Series]:
        return super().std(ddof=ddof, numeric_only=numeric_only, engine=engine, engine_kwargs=engine_kwargs)

    @doc(template_header, create_section_header('Parameters'), dedent('\n        ddof : int, default 1\n            Delta Degrees of Freedom.  The divisor used in calculations\n            is ``N - ddof``, where ``N`` represents the number of elements.\n\n        ').replace('\n', '', 1), kwargs_numeric_only, window_agg_numba_parameters('1.4'), create_section_header('Returns'), template_returns, create_section_header('See Also'), 'numpy.var : Equivalent method for NumPy array.\n', template_see_also, create_section_header('Notes'), dedent('\n        The default ``ddof`` of 1 used in :meth:`Series.var` is different\n        than the default ``ddof`` of 0 in :func:`numpy.var`.\n\n        A minimum of one period is required for the rolling calculation.\n\n        ').replace('\n', '', 1), create_section_header('Examples'), dedent('\n        >>> s = pd.Series([5, 5, 6, 7, 5, 5, 5])\n\n        >>> s.expanding(3).var()\n        0         NaN\n        1         NaN\n        2    0.333333\n        3    0.916667\n        4    0.800000\n        5    0.700000\n        6    0.619048\n        dtype: float64\n        ').replace('\n', '', 1), window_method='expanding', aggregation_description='variance', agg_method='var')
    def var(
        self,
        ddof: int = 1,
        numeric_only: bool = False,
        engine: Optional[str] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None
    ) -> Union[DataFrame, Series]:
        return super().var(ddof=ddof, numeric_only=numeric_only, engine=engine, engine_kwargs=engine_kwargs)

    @doc(template_header, create_section_header('Parameters'), dedent('\n        ddof : int, default 1\n            Delta Degrees of Freedom.  The divisor used in calculations\n            is ``N - ddof``, where ``N`` represents the number of elements.\n\n        ').replace('\n', '', 1), kwargs_numeric_only, create_section_header('Returns'), template_returns, create_section_header('See Also'), template_see_also, create_section_header('Notes'), 'A minimum of one period is required for the calculation.\n\n', create_section_header('Examples'), dedent('\n        >>> s = pd.Series([0, 1, 2, 3])\n\n        >>> s.expanding().sem()\n        0         NaN\n        1    0.707107\n        2    0.707107\n        3    0.745356\n        dtype: float64\n        ').replace('\n', '', 1), window_method='expanding', aggregation_description='standard error of mean', agg_method='sem')
    def sem(
        self,
        ddof: int = 1,
        numeric_only: bool = False
    ) -> Union[DataFrame, Series]:
        return super().sem(ddof=ddof, numeric_only=numeric_only)

    @doc(template_header, create_section_header('Parameters'), kwargs_numeric_only, create_section_header('Returns'), template_returns, create_section_header('See Also'), 'scipy.stats.skew : Third moment of a probability density.\n', template_see_also, create_section_header('Notes'), 'A minimum of three periods is required for the rolling calculation.\n\n', create_section_header('Examples'), dedent("        >>> ser = pd.Series([-1, 0, 2, -1, 2], index=['a', 'b', 'c', 'd', 'e'])\n        >>> ser.expanding().skew()\n        a         NaN\n        b         NaN\n        c    0.935220\n        d    1.414214\n        e    0.315356\n       