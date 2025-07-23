from __future__ import annotations
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Literal, final, overload, Callable, Union
from pandas.util._decorators import Appender, Substitution, doc
from pandas.core.indexers.objects import BaseIndexer, ExpandingIndexer, GroupbyIndexer
from pandas.core.window.doc import (
    _shared_docs,
    create_section_header,
    kwargs_numeric_only,
    numba_notes,
    template_header,
    template_pipe,
    template_returns,
    template_see_also,
    window_agg_numba_parameters,
    window_apply_parameters,
)
from pandas.core.window.rolling import BaseWindowGroupby, RollingAndExpandingMixin
if TYPE_CHECKING:
    from collections.abc import Callable
    from pandas._typing import Concatenate, P, QuantileInterpolation, Self, T, WindowingRankType
    from pandas import DataFrame, Series
    from pandas.core.generic import NDFrame


class Expanding(RollingAndExpandingMixin):
    """
    Provide expanding window calculations.

    An expanding window yields the value of an aggregation statistic with all the data
    available up to that point in time.

    Parameters
    ----------
    min_periods : int, default 1
        Minimum number of observations in window required to have a value;
        otherwise, result is ``np.nan``.

    method : str {'single', 'table'}, default 'single'
        Execute the rolling operation per single column or row (``'single'``)
        or over the entire object (``'table'``).

        This argument is only implemented when specifying ``engine='numba'``
        in the method call.

        .. versionadded:: 1.3.0

    Returns
    -------
    pandas.api.typing.Expanding
        An instance of Expanding for further expanding window calculations,
        e.g. using the ``sum`` method.

    See Also
    --------
    rolling : Provides rolling window calculations.
    ewm : Provides exponential weighted functions.

    Notes
    -----
    See :ref:`Windowing Operations <window.expanding>` for further usage details
    and examples.

    Examples
    --------
    >>> df = pd.DataFrame({"B": [0, 1, 2, np.nan, 4]})
    >>> df
         B
    0  0.0
    1  1.0
    2  2.0
    3  NaN
    4  4.0

    **min_periods**

    Expanding sum with 1 vs 3 observations needed to calculate a value.

    >>> df.expanding(1).sum()
         B
    0  0.0
    1  1.0
    2  3.0
    3  3.0
    4  7.0
    >>> df.expanding(3).sum()
         B
    0  NaN
    1  NaN
    2  3.0
    3  3.0
    4  7.0
    """
    _attributes: list[str] = ['min_periods', 'method']

    def __init__(
        self,
        obj: Union[Series, DataFrame],
        min_periods: int = 1,
        method: Literal['single', 'table'] = 'single',
        selection: Any = None,
    ) -> None:
        super().__init__(obj=obj, min_periods=min_periods, method=method, selection=selection)

    def _get_window_indexer(self) -> BaseIndexer:
        """
        Return an indexer class that will compute the window start and end bounds
        """
        return ExpandingIndexer()

    @doc(
        _shared_docs['aggregate'],
        see_also=dedent(
            """
        See Also
        --------
        DataFrame.aggregate : Similar DataFrame method.
        Series.aggregate : Similar Series method.
        """
        ),
        examples=dedent(
            """
        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        >>> df
           A  B  C
        0  1  4  7
        1  2  5  8
        2  3  6  9

        >>> df.ewm(alpha=0.5).mean()
                  A         B         C
        0  1.000000  4.000000  7.000000
        1  1.666667  4.666667  7.666667
        2  2.428571  5.428571  8.428571
        """
        ),
        klass='Series/Dataframe',
        axis='',
    )
    def aggregate(
        self,
        func: Union[str, Callable, list, dict, None] = None,
        *args: Any,
        **kwargs: Any,
    ) -> NDFrame:
        return super().aggregate(func, *args, **kwargs)

    agg = aggregate

    @doc(
        template_header,
        create_section_header('Parameters'),
        kwargs_numeric_only,
        create_section_header('Returns'),
        template_returns,
        create_section_header('See Also'),
        template_see_also,
        create_section_header('Examples'),
        dedent(
            """
        >>> ser = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
        >>> ser.expanding().count()
        a    1.0
        b    2.0
        c    3.0
        d    4.0
        dtype: float64
        """
        ),
        window_method='expanding',
        aggregation_description='count of non NaN observations',
        agg_method='count',
    )
    def count(
        self,
        numeric_only: bool = False,
    ) -> NDFrame:
        return super().count(numeric_only=numeric_only)

    @doc(
        template_header,
        create_section_header('Parameters'),
        window_apply_parameters,
        create_section_header('Returns'),
        template_returns,
        create_section_header('See Also'),
        template_see_also,
        create_section_header('Examples'),
        dedent(
            """
        >>> ser = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
        >>> ser.expanding().apply(lambda s: s.max() - 2 * s.min())
        a   -1.0
        b    0.0
        c    1.0
        d    2.0
        dtype: float64
        """
        ),
        window_method='expanding',
        aggregation_description='custom aggregation function',
        agg_method='apply',
    )
    def apply(
        self,
        func: Callable[..., Any],
        raw: bool = False,
        engine: Union[Literal['cython', 'numba'], None] = None,
        engine_kwargs: dict[str, Any] | None = None,
        args: tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> NDFrame:
        return super().apply(func, raw=raw, engine=engine, engine_kwargs=engine_kwargs, args=args, kwargs=kwargs)

    @overload
    def pipe(
        self,
        func: Callable[[Expanding], T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        ...

    @overload
    def pipe(
        self,
        func: Callable[[Expanding, P], T],
        *args: P,
        **kwargs: Any,
    ) -> T:
        ...

    @final
    @Substitution(
        klass='Expanding',
        examples="""
    >>> df = pd.DataFrame({'A': [1, 2, 3, 4]},
    ...                   index=pd.date_range('2012-08-02', periods=4))
    >>> df
                A
    2012-08-02  1
    2012-08-03  2
    2012-08-04  3
    2012-08-05  4

    To get the difference between each expanding window's maximum and minimum
    value in one pass, you can do

    >>> df.expanding().pipe(lambda x: x.max() - x.min())
                  A
    2012-08-02  0.0
    2012-08-03  1.0
    2012-08-04  2.0
    2012-08-05  3.0"""
    )
    @Appender(template_pipe)
    def pipe(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return super().pipe(func, *args, **kwargs)

    @doc(
        template_header,
        create_section_header('Parameters'),
        kwargs_numeric_only,
        window_agg_numba_parameters(),
        create_section_header('Returns'),
        template_returns,
        create_section_header('See Also'),
        template_see_also,
        create_section_header('Notes'),
        numba_notes,
        create_section_header('Examples'),
        dedent(
            """
        >>> ser = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
        >>> ser.expanding().sum()
        a     1.0
        b     3.0
        c     6.0
        d    10.0
        dtype: float64
        """
        ),
        window_method='expanding',
        aggregation_description='sum',
        agg_method='sum',
    )
    def sum(
        self,
        numeric_only: bool = False,
        engine: Union[Literal['cython', 'numba'], None] = None,
        engine_kwargs: dict[str, Any] | None = None,
    ) -> NDFrame:
        return super().sum(numeric_only=numeric_only, engine=engine, engine_kwargs=engine_kwargs)

    @doc(
        template_header,
        create_section_header('Parameters'),
        kwargs_numeric_only,
        window_agg_numba_parameters(),
        create_section_header('Returns'),
        template_returns,
        create_section_header('See Also'),
        template_see_also,
        create_section_header('Notes'),
        numba_notes,
        create_section_header('Examples'),
        dedent(
            """
        >>> ser = pd.Series([3, 2, 1, 4], index=['a', 'b', 'c', 'd'])
        >>> ser.expanding().max()
        a    3.0
        b    3.0
        c    3.0
        d    4.0
        dtype: float64
        """
        ),
        window_method='expanding',
        aggregation_description='maximum',
        agg_method='max',
    )
    def max(
        self,
        numeric_only: bool = False,
        engine: Union[Literal['cython', 'numba'], None] = None,
        engine_kwargs: dict[str, Any] | None = None,
    ) -> NDFrame:
        return super().max(numeric_only=numeric_only, engine=engine, engine_kwargs=engine_kwargs)

    @doc(
        template_header,
        create_section_header('Parameters'),
        kwargs_numeric_only,
        window_agg_numba_parameters(),
        create_section_header('Returns'),
        template_returns,
        create_section_header('See Also'),
        template_see_also,
        create_section_header('Notes'),
        numba_notes,
        create_section_header('Examples'),
        dedent(
            """
        >>> ser = pd.Series([2, 3, 4, 1], index=['a', 'b', 'c', 'd'])
        >>> ser.expanding().min()
        a    2.0
        b    2.0
        c    2.0
        d    1.0
        dtype: float64
        """
        ),
        window_method='expanding',
        aggregation_description='minimum',
        agg_method='min',
    )
    def min(
        self,
        numeric_only: bool = False,
        engine: Union[Literal['cython', 'numba'], None] = None,
        engine_kwargs: dict[str, Any] | None = None,
    ) -> NDFrame:
        return super().min(numeric_only=numeric_only, engine=engine, engine_kwargs=engine_kwargs)

    @doc(
        template_header,
        create_section_header('Parameters'),
        kwargs_numeric_only,
        window_agg_numba_parameters(),
        create_section_header('Returns'),
        template_returns,
        create_section_header('See Also'),
        template_see_also,
        create_section_header('Notes'),
        numba_notes,
        create_section_header('Examples'),
        dedent(
            """
        >>> ser = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
        >>> ser.expanding().mean()
        a    1.0
        b    1.5
        c    2.0
        d    2.5
        dtype: float64
        """
        ),
        window_method='expanding',
        aggregation_description='mean',
        agg_method='mean',
    )
    def mean(
        self,
        numeric_only: bool = False,
        engine: Union[Literal['cython', 'numba'], None] = None,
        engine_kwargs: dict[str, Any] | None = None,
    ) -> NDFrame:
        return super().mean(numeric_only=numeric_only, engine=engine, engine_kwargs=engine_kwargs)

    @doc(
        template_header,
        create_section_header('Parameters'),
        kwargs_numeric_only,
        window_agg_numba_parameters(),
        create_section_header('Returns'),
        template_returns,
        create_section_header('See Also'),
        template_see_also,
        create_section_header('Notes'),
        numba_notes,
        create_section_header('Examples'),
        dedent(
            """
        >>> ser = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
        >>> ser.expanding().median()
        a    1.0
        b    1.5
        c    2.0
        d    2.5
        dtype: float64
        """
        ),
        window_method='expanding',
        aggregation_description='median',
        agg_method='median',
    )
    def median(
        self,
        numeric_only: bool = False,
        engine: Union[Literal['cython', 'numba'], None] = None,
        engine_kwargs: dict[str, Any] | None = None,
    ) -> NDFrame:
        return super().median(numeric_only=numeric_only, engine=engine, engine_kwargs=engine_kwargs)

    @doc(
        template_header,
        create_section_header('Parameters'),
        dedent(
            '\n        ddof : int, default 1\n'
            '            Delta Degrees of Freedom.  The divisor used in calculations\n'
            '            is ``N - ddof``, where ``N`` represents the number of elements.\n\n'
        ).replace('\n', '', 1),
        kwargs_numeric_only,
        window_agg_numba_parameters('1.4'),
        create_section_header('Returns'),
        template_returns,
        create_section_header('See Also'),
        'numpy.std : Equivalent method for NumPy array.\n',
        template_see_also,
        create_section_header('Notes'),
        dedent(
            '\n        The default ``ddof`` of 1 used in :meth:`Series.std` is different\n'
            '        than the default ``ddof`` of 0 in :func:`numpy.std`.\n\n'
            '        A minimum of one period is required for the rolling calculation.\n\n'
        ).replace('\n', '', 1),
        create_section_header('Examples'),
        dedent(
            '\n        >>> s = pd.Series([5, 5, 6, 7, 5, 5, 5])\n\n'
            '        >>> s.expanding(3).std()\n'
            '        0         NaN\n'
            '        1         NaN\n'
            '        2    0.577350\n'
            '        3    0.957427\n'
            '        4    0.894427\n'
            '        5    0.836660\n'
            '        6    0.786796\n'
            '        dtype: float64\n        '
        ).replace('\n', '', 1),
        window_method='expanding',
        aggregation_description='standard deviation',
        agg_method='std',
    )
    def std(
        self,
        ddof: int = 1,
        numeric_only: bool = False,
        engine: Union[Literal['cython', 'numba'], None] = None,
        engine_kwargs: dict[str, Any] | None = None,
    ) -> NDFrame:
        return super().std(ddof=ddof, numeric_only=numeric_only, engine=engine, engine_kwargs=engine_kwargs)

    @doc(
        template_header,
        create_section_header('Parameters'),
        dedent(
            '\n        ddof : int, default 1\n'
            '            Delta Degrees of Freedom.  The divisor used in calculations\n'
            '            is ``N - ddof``, where ``N`` represents the number of elements.\n\n'
        ).replace('\n', '', 1),
        kwargs_numeric_only,
        window_agg_numba_parameters('1.4'),
        create_section_header('Returns'),
        template_returns,
        create_section_header('See Also'),
        'numpy.var : Equivalent method for NumPy array.\n',
        template_see_also,
        create_section_header('Notes'),
        dedent(
            '\n        The default ``ddof`` of 1 used in :meth:`Series.var` is different\n'
            '        than the default ``ddof`` of 0 in :func:`numpy.var`.\n\n'
            '        A minimum of one period is required for the rolling calculation.\n\n'
        ).replace('\n', '', 1),
        create_section_header('Examples'),
        dedent(
            '\n        >>> s = pd.Series([5, 5, 6, 7, 5, 5, 5])\n\n'
            '        >>> s.expanding(3).var()\n'
            '        0         NaN\n'
            '        1         NaN\n'
            '        2    0.333333\n'
            '        3    0.916667\n'
            '        4    0.800000\n'
            '        5    0.700000\n'
            '        6    0.619048\n'
            '        dtype: float64\n        '
        ).replace('\n', '', 1),
        window_method='expanding',
        aggregation_description='variance',
        agg_method='var',
    )
    def var(
        self,
        ddof: int = 1,
        numeric_only: bool = False,
        engine: Union[Literal['cython', 'numba'], None] = None,
        engine_kwargs: dict[str, Any] | None = None,
    ) -> NDFrame:
        return super().var(ddof=ddof, numeric_only=numeric_only, engine=engine, engine_kwargs=engine_kwargs)

    @doc(
        template_header,
        create_section_header('Parameters'),
        dedent(
            '\n        ddof : int, default 1\n'
            '            Delta Degrees of Freedom.  The divisor used in calculations\n'
            '            is ``N - ddof``, where ``N`` represents the number of elements.\n\n'
        ).replace('\n', '', 1),
        kwargs_numeric_only,
        create_section_header('Returns'),
        template_returns,
        create_section_header('See Also'),
        'scipy.stats.skew : Third moment of a probability density.\n',
        template_see_also,
        create_section_header('Notes'),
        'A minimum of three periods is required for the rolling calculation.\n\n',
        create_section_header('Examples'),
        dedent(
            "        >>> ser = pd.Series([-1, 0, 2, -1, 2], index=['a', 'b', 'c', 'd', 'e'])\n"
            "        >>> ser.expanding().skew()\n"
            "        a         NaN\n"
            "        b         NaN\n"
            "        c    0.935220\n"
            "        d    1.414214\n"
            "        e    0.315356\n"
            "        dtype: float64\n        "
        ),
        window_method='expanding',
        aggregation_description='unbiased skewness',
        agg_method='skew',
    )
    def skew(
        self,
        numeric_only: bool = False,
    ) -> NDFrame:
        return super().skew(numeric_only=numeric_only)

    @doc(
        template_header,
        create_section_header('Parameters'),
        kwargs_numeric_only,
        create_section_header('Returns'),
        template_returns,
        create_section_header('See Also'),
        'scipy.stats.kurtosis : Reference SciPy method.\n',
        template_see_also,
        create_section_header('Notes'),
        'A minimum of four periods is required for the calculation.\n\n',
        create_section_header('Examples'),
        dedent(
            '\n        The example below will show a rolling calculation with a window size of\n'
            '        four matching the equivalent function call using `scipy.stats`.\n\n'
            '        >>> arr = [1, 2, 3, 4, 999]\n'
            '        >>> import scipy.stats\n'
            '        >>> print(f"{{scipy.stats.kurtosis(arr[:-1], bias=False):.6f}}")\n'
            '        -1.200000\n'
            '        >>> print(f"{{scipy.stats.kurtosis(arr, bias=False):.6f}}")\n'
            '        4.999874\n'
            '        >>> s = pd.Series(arr)\n'
            '        >>> s.expanding(4).kurt()\n'
            '        0         NaN\n'
            '        1         NaN\n'
            '        2         NaN\n'
            '        3   -1.200000\n'
            '        4    4.999874\n'
            '        dtype: float64\n        '
        ).replace('\n', '', 1),
        window_method='expanding',
        aggregation_description="Fisher's definition of kurtosis without bias",
        agg_method='kurt',
    )
    def kurt(
        self,
        numeric_only: bool = False,
    ) -> NDFrame:
        return super().kurt(numeric_only=numeric_only)

    @doc(
        template_header,
        create_section_header('Parameters'),
        kwargs_numeric_only,
        create_section_header('Returns'),
        template_returns,
        create_section_header('See Also'),
        dedent(
            '\n        GroupBy.first : Similar method for GroupBy objects.\n'
            '        Expanding.last : Method to get the last element in each window.\n\n'
        ).replace('\n', '', 1),
        create_section_header('Examples'),
        dedent(
            '\n        The example below will show an expanding calculation with a window size of\n'
            '        three.\n\n'
            '        >>> s = pd.Series(range(5))\n'
            '        >>> s.expanding(3).first()\n'
            '        0         NaN\n'
            '        1         NaN\n'
            '        2         0.0\n'
            '        3         0.0\n'
            '        4         0.0\n'
            '        dtype: float64\n        '
        ).replace('\n', '', 1),
        window_method='expanding',
        aggregation_description='First (left-most) element of the window',
        agg_method='first',
    )
    def first(
        self,
        numeric_only: bool = False,
    ) -> NDFrame:
        return super().first(numeric_only=numeric_only)

    @doc(
        template_header,
        create_section_header('Parameters'),
        kwargs_numeric_only,
        create_section_header('Returns'),
        template_returns,
        create_section_header('See Also'),
        dedent(
            '\n        GroupBy.last : Similar method for GroupBy objects.\n'
            '        Expanding.first : Method to get the first element in each window.\n\n'
        ).replace('\n', '', 1),
        create_section_header('Examples'),
        dedent(
            '\n        The example below will show an expanding calculation with a window size of\n'
            '        three.\n\n'
            '        >>> s = pd.Series(range(5))\n'
            '        >>> s.expanding(3).last()\n'
            '        0         NaN\n'
            '        1         NaN\n'
            '        2         2.0\n'
            '        3         3.0\n'
            '        4         4.0\n'
            '        dtype: float64\n        '
        ).replace('\n', '', 1),
        window_method='expanding',
        aggregation_description='Last (right-most) element of the window',
        agg_method='last',
    )
    def last(
        self,
        numeric_only: bool = False,
    ) -> NDFrame:
        return super().last(numeric_only=numeric_only)

    @doc(
        template_header,
        create_section_header('Parameters'),
        dedent(
            "\n        q : float\n"
            "            Quantile to compute. 0 <= quantile <= 1.\n\n"
            "            .. deprecated:: 2.1.0\n"
            "                This was renamed from 'quantile' to 'q' in version 2.1.0.\n"
            "        interpolation : {{'linear', 'lower', 'higher', 'midpoint', 'nearest'}}\n"
            "            This optional parameter specifies the interpolation method to use,\n"
            "            when the desired quantile lies between two data points `i` and `j`:\n\n"
            "                * linear: `i + (j - i) * fraction`, where `fraction` is the\n"
            "                  fractional part of the index surrounded by `i` and `j`.\n"
            "                * lower: `i`.\n"
            "                * higher: `j`.\n"
            "                * nearest: `i` or `j` whichever is nearest.\n"
            "                * midpoint: (`i` + `j`) / 2.\n        "
        ).replace('\n', '', 1),
        kwargs_numeric_only,
        create_section_header('Returns'),
        template_returns,
        create_section_header('See Also'),
        template_see_also,
        create_section_header('Examples'),
        dedent(
            "        >>> ser = pd.Series([1, 2, 3, 4, 5, 6], index=['a', 'b', 'c', 'd', 'e', 'f'])\n"
            "        >>> ser.expanding(min_periods=4).quantile(.25)\n"
            "        a     NaN\n"
            "        b     NaN\n"
            "        c     NaN\n"
            "        d    1.75\n"
            "        e    2.00\n"
            "        f    2.25\n"
            "        dtype: float64\n        "
        ),
        window_method='expanding',
        aggregation_description='quantile',
        agg_method='quantile',
    )
    def quantile(
        self,
        q: float,
        interpolation: Literal['linear', 'lower', 'higher', 'midpoint', 'nearest'] = 'linear',
        numeric_only: bool = False,
    ) -> NDFrame:
        return super().quantile(q=q, interpolation=interpolation, numeric_only=numeric_only)

    @doc(
        template_header,
        '.. versionadded:: 1.4.0 \n\n',
        create_section_header('Parameters'),
        dedent(
            "\n        method : {{'average', 'min', 'max'}}, default 'average'\n"
            "            How to rank the group of records that have the same value (i.e. ties):\n\n"
            "            * average: average rank of the group\n"
            "            * min: lowest rank in the group\n"
            "            * max: highest rank in the group\n\n"
            "        ascending : bool, default True\n"
            "            Whether or not the elements should be ranked in ascending order.\n"
            "        pct : bool, default False\n"
            "            Whether or not to display the returned rankings in percentile\n"
            "            form.\n        "
        ).replace('\n', '', 1),
        kwargs_numeric_only,
        create_section_header('Returns'),
        template_returns,
        create_section_header('See Also'),
        template_see_also,
        create_section_header('Examples'),
        dedent(
            '\n        >>> s = pd.Series([1, 4, 2, 3, 5, 3])\n'
            '        >>> s.expanding().rank()\n'
            '        0    1.0\n'
            '        1    2.0\n'
            '        2    2.0\n'
            '        3    3.0\n'
            '        4    5.0\n'
            '        5    3.5\n'
            '        dtype: float64\n\n'
            '        >>> s.expanding().rank(method="max")\n'
            '        0    1.0\n'
            '        1    2.0\n'
            '        2    2.0\n'
            '        3    3.0\n'
            '        4    5.0\n'
            '        5    4.0\n'
            '        dtype: float64\n\n'
            '        >>> s.expanding().rank(method="min")\n'
            '        0    1.0\n'
            '        1    2.0\n'
            '        2    2.0\n'
            '        3    3.0\n'
            '        4    5.0\n'
            '        5    3.0\n'
            '        dtype: float64\n        '
        ).replace('\n', '', 1),
        window_method='expanding',
        aggregation_description='rank',
        agg_method='rank',
    )
    def rank(
        self,
        method: WindowingRankType = 'average',
        ascending: bool = True,
        pct: bool = False,
        numeric_only: bool = False,
    ) -> NDFrame:
        return super().rank(method=method, ascending=ascending, pct=pct, numeric_only=numeric_only)

    @doc(
        template_header,
        create_section_header('Parameters'),
        dedent(
            '\n        other : Series or DataFrame, optional\n'
            '            If not supplied then will default to self and produce pairwise\n'
            '            output.\n'
            '        pairwise : bool, default None\n'
            '            If False then only matching columns between self and other will be\n'
            '            used and the output will be a DataFrame.\n'
            '            If True then all pairwise combinations will be calculated and the\n'
            '            output will be a MultiIndexed DataFrame in the case of DataFrame\n'
            '            inputs. In the case of missing elements, only complete pairwise\n'
            '            observations will be used.\n'
            '        ddof : int, default 1\n'
            '            Delta Degrees of Freedom.  The divisor used in calculations\n'
            '            is ``N - ddof``, where ``N`` represents the number of elements.\n'
            '        '
        ).replace('\n', '', 1),
        kwargs_numeric_only,
        create_section_header('Returns'),
        template_returns,
        create_section_header('See Also'),
        template_see_also,
        create_section_header('Examples'),
        dedent(
            "        >>> ser1 = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])\n"
            "        >>> ser2 = pd.Series([10, 11, 13, 16], index=['a', 'b', 'c', 'd'])\n"
            "        >>> ser1.expanding().cov(ser2)\n"
            "        a         NaN\n"
            "        b    0.500000\n"
            "        c    1.500000\n"
            "        d    3.333333\n"
            "        dtype: float64\n        "
        ).replace('\n', '', 1),
        window_method='expanding',
        aggregation_description='sample covariance',
        agg_method='cov',
    )
    def cov(
        self,
        other: Union[Series, DataFrame, None] = None,
        pairwise: Union[bool, None] = None,
        ddof: int = 1,
        numeric_only: bool = False,
    ) -> Union[Series, DataFrame]:
        return super().cov(other=other, pairwise=pairwise, ddof=ddof, numeric_only=numeric_only)

    @doc(
        template_header,
        create_section_header('Parameters'),
        dedent(
            '\n        other : Series or DataFrame, optional\n'
            '            If not supplied then will default to self and produce pairwise\n'
            '            output.\n'
            '        pairwise : bool, default None\n'
            '            If False then only matching columns between self and other will be\n'
            '            used and the output will be a DataFrame.\n'
            '            If True then all pairwise combinations will be calculated and the\n'
            '            output will be a MultiIndexed DataFrame in the case of DataFrame\n'
            '            inputs. In the case of missing elements, only complete pairwise\n'
            '            observations will be used.\n'
            '        ddof : int, default 1\n'
            '            Delta Degrees of Freedom.  The divisor used in calculations\n'
            '            is ``N - ddof``, where ``N`` represents the number of elements.\n\n'
        ).replace('\n', '', 1),
        kwargs_numeric_only,
        create_section_header('Returns'),
        template_returns,
        create_section_header('See Also'),
        dedent(
            "\n        cov : Similar method to calculate covariance.\n"
            "        numpy.corrcoef : NumPy Pearson's correlation calculation.\n"
            "        "
        ).replace('\n', '', 1),
        template_see_also,
        create_section_header('Notes'),
        dedent(
            "\n        This function uses Pearson's definition of correlation\n"
            "        (https://en.wikipedia.org/wiki/Pearson_correlation_coefficient).\n\n"
            "        When `other` is not specified, the output will be self correlation (e.g.\n"
            "        all 1's), except for :class:`~pandas.DataFrame` inputs with `pairwise`\n"
            "        set to `True`.\n\n"
            "        Function will return ``NaN`` for correlations of equal valued sequences;\n"
            "        this is the result of a 0/0 division error.\n\n"
            "        When `pairwise` is set to `False`, only matching columns between `self` and\n"
            "        `other` will be used.\n\n"
            "        When `pairwise` is set to `True`, the output will be a MultiIndex DataFrame\n"
            "        with the original index on the first level, and the `other` DataFrame\n"
            "        columns on the second level.\n\n"
            "        In the case of missing elements, only complete pairwise observations\n"
            "        will be used.\n\n"
            "        "
        ).replace('\n', '', 1),
        create_section_header('Examples'),
        dedent(
            "        >>> ser1 = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])\n"
            "        >>> ser2 = pd.Series([10, 11, 13, 16], index=['a', 'b', 'c', 'd'])\n"
            "        >>> ser1.expanding().corr(ser2)\n"
            "        a         NaN\n"
            "        b    1.000000\n"
            "        c    0.981981\n"
            "        d    0.975900\n"
            "        dtype: float64\n        "
        ).replace('\n', '', 1),
        window_method='expanding',
        aggregation_description='correlation',
        agg_method='corr',
    )
    def corr(
        self,
        other: Union[Series, DataFrame, None] = None,
        pairwise: Union[bool, None] = None,
        ddof: int = 1,
        numeric_only: bool = False,
    ) -> Union[Series, DataFrame]:
        return super().corr(other=other, pairwise=pairwise, ddof=ddof, numeric_only=numeric_only)


class ExpandingGroupby(BaseWindowGroupby, Expanding):
    """
    Provide a expanding groupby implementation.
    """
    _attributes: list[str] = Expanding._attributes + BaseWindowGroupby._attributes

    def _get_window_indexer(self) -> GroupbyIndexer:
        """
        Return an indexer class that will compute the window start and end bounds

        Returns
        -------
        GroupbyIndexer
        """
        window_indexer = GroupbyIndexer(
            groupby_indices=self._grouper.indices, window_indexer=ExpandingIndexer
        )
        return window_indexer
