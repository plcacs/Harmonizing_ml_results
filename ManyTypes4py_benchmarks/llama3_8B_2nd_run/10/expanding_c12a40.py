from __future__ import annotations
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Literal, final, overload
from pandas.util._decorators import Appender, Substitution, doc
from pandas.core.indexers.objects import BaseIndexer, ExpandingIndexer, GroupbyIndexer
from pandas.core.window.doc import _shared_docs, create_section_header, kwargs_numeric_only, numba_notes, template_header, template_pipe, template_returns, template_see_also, window_agg_numba_parameters, window_apply_parameters
from pandas.core.window.rolling import BaseWindowGroupby, RollingAndExpandingMixin

class Expanding(RollingAndExpandingMixin):
    """Provide expanding window calculations.

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

    _attributes: tuple[str, ...] = ['min_periods', 'method']

    def __init__(self, obj: Any, min_periods: int = 1, method: str = 'single', selection: Any = None) -> None:
        ...

    def _get_window_indexer(self) -> GroupbyIndexer:
        ...

    @doc(_shared_docs['aggregate'], see_also=dedent('\n        See Also\n        --------\n        DataFrame.aggregate : Similar DataFrame method.\n        Series.aggregate : Similar Series method.\n        '), examples=dedent('\n        Examples\n        --------\n        >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})\n        >>> df\n           A  B  C\n        0  1  4  7\n        1  2  5  8\n        2  3  6  9\n\n        >>> df.ewm(alpha=0.5).mean()\n                  A         B         C\n        0  1.000000  4.000000  7.000000\n        1  1.666667  4.666667  7.666667\n        2  2.428571  5.428571  8.428571\n        '), klass='Series/Dataframe', axis='')
    @overload
    def aggregate(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        ...

    @overload
    def aggregate(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        ...

    @final
    @Substitution(klass='Expanding', examples=dedent('\n        >>> df = pd.DataFrame({"A": [1, 2, 3, 4]},\n        ...                   index=pd.date_range("2012-08-02", periods=4))\n        >>> df\n                A\n        2012-08-02  1\n        2012-08-03  2\n        2012-08-04  3\n        2012-08-05  4\n\n        To get the difference between each expanding window\'s maximum and minimum\n        value in one pass, you can do\n\n        >>> df.expanding().pipe(lambda x: x.max() - x.min())\n                  A\n        2012-08-02  0.0\n        2012-08-03  1.0\n        2012-08-04  2.0\n        2012-08-05  3.0'))
    @Appender(template_pipe)
    def pipe(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        ...

    @doc(template_header, create_section_header='Parameters', kwargs_numeric_only, window_agg_numba_parameters(), create_section_header='Returns', template_returns, create_section_header='See Also', template_see_also, create_section_header='Notes', numba_notes, create_section_header='Examples', dedent('\n        >>> ser = pd.Series([1, 2, 3, 4], index=["a", "b", "c", "d"])'))
    def sum(self, numeric_only: bool = False, engine: Any = None, engine_kwargs: Any = None) -> Any:
        ...

    @doc(template_header, create_section_header='Parameters', kwargs_numeric_only,