from __future__ import annotations
import itertools
from typing import TYPE_CHECKING, Literal, cast, Any, Optional, Union
import numpy as np
from pandas._libs import lib
from pandas.core.dtypes.cast import maybe_downcast_to_dtype
from pandas.core.dtypes.common import is_list_like, is_nested_list_like, is_scalar
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import ABCDataFrame, ABCSeries
import pandas.core.common as com
from pandas.core.groupby import Grouper
from pandas.core.indexes.api import Index, MultiIndex, get_objs_combined_axis
from pandas.core.reshape.concat import concat
from pandas.core.series import Series

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable
    from pandas._typing import AggFuncType, AggFuncTypeBase, AggFuncTypeDict, IndexLabel, SequenceNotStr
    from pandas import DataFrame

def pivot_table(
    data: DataFrame, 
    values: Optional[Union[Hashable, list[Hashable]]] = None, 
    index: Optional[Union[Hashable, list[Hashable]]] = None, 
    columns: Optional[Union[Hashable, list[Hashable]]] = None, 
    aggfunc: Union[str, Callable, list[Union[str, Callable]], dict[Hashable, Union[str, Callable, list[Union[str, Callable]]]]] = 'mean', 
    fill_value: Optional[Any] = None, 
    margins: bool = False, 
    dropna: bool = True, 
    margins_name: str = 'All', 
    observed: bool = True, 
    sort: bool = True, 
    **kwargs: Any
) -> DataFrame:
    """
    Create a spreadsheet-style pivot table as a DataFrame.

    The levels in the pivot table will be stored in MultiIndex objects
    (hierarchical indexes) on the index and columns of the result DataFrame.

    Parameters
    ----------
    data : DataFrame
        Input pandas DataFrame object.
    values : list-like or scalar, optional
        Column or columns to aggregate.
    index : column, Grouper, array, or list of the previous
        Keys to group by on the pivot table index. If a list is passed,
        it can contain any of the other types (except list). If an array is
        passed, it must be the same length as the data and will be used in
        the same manner as column values.
    columns : column, Grouper, array, or list of the previous
        Keys to group by on the pivot table column. If a list is passed,
        it can contain any of the other types (except list). If an array is
        passed, it must be the same length as the data and will be used in
        the same manner as column values.
    aggfunc : function, list of functions, dict, default "mean"
        If a list of functions is passed, the resulting pivot table will have
        hierarchical columns whose top level are the function names
        (inferred from the function objects themselves).
        If a dict is passed, the key is column to aggregate and the value is
        function or list of functions. If ``margins=True``, aggfunc will be
        used to calculate the partial aggregates.
    fill_value : scalar, default None
        Value to replace missing values with (in the resulting pivot table,
        after aggregation).
    margins : bool, default False
        If ``margins=True``, special ``All`` columns and rows
        will be added with partial group aggregates across the categories
        on the rows and columns.
    dropna : bool, default True
        Do not include columns whose entries are all NaN. If True,
        rows with a NaN value in any column will be omitted before
        computing margins.
    margins_name : str, default 'All'
        Name of the row / column that will contain the totals
        when margins is True.
    observed : bool, default False
        This only applies if any of the groupers are Categoricals.
        If True: only show observed values for categorical groupers.
        If False: show all values for categorical groupers.

        .. versionchanged:: 3.0.0

            The default value is now ``True``.

    sort : bool, default True
        Specifies if the result should be sorted.

        .. versionadded:: 1.3.0

    **kwargs : dict
        Optional keyword arguments to pass to ``aggfunc``.

        .. versionadded:: 3.0.0

    Returns
    -------
    DataFrame
        An Excel style pivot table.

    See Also
    --------
    DataFrame.pivot : Pivot without aggregation that can handle
        non-numeric data.
    DataFrame.melt: Unpivot a DataFrame from wide to long format,
        optionally leaving identifiers set.
    wide_to_long : Wide panel to long format. Less flexible but more
        user-friendly than melt.

    Notes
    -----
    Reference :ref:`the user guide <reshaping.pivot>` for more examples.

    Examples
    --------
    >>> df = pd.DataFrame(
    ...     {
    ...         "A": ["foo", "foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar"],
    ...         "B": ["one", "one", "one", "two", "two", "one", "one", "two", "two"],
    ...         "C": [
    ...             "small",
    ...             "large",
    ...             "large",
    ...             "small",
    ...             "small",
    ...             "large",
    ...             "small",
    ...             "small",
    ...             "large",
    ...         ],
    ...         "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
    ...         "E": [2, 4, 5, 5, 6, 6, 8, 9, 9],
    ...     }
    ... )
    >>> df
         A    B      C  D  E
    0  foo  one  small  1  2
    1  foo  one  large  2  4
    2  foo  one  large  2  5
    3  foo  two  small  3  5
    4  foo  two  small  3  6
    5  bar  one  large  4  6
    6  bar  one  small  5  8
    7  bar  two  small  6  9
    8  bar  two  large  7  9

    This first example aggregates values by taking the sum.

    >>> table = pd.pivot_table(
    ...     df, values="D", index=["A", "B"], columns=["C"], aggfunc="sum"
    ... )
    >>> table
    C        large  small
    A   B
    bar one    4.0    5.0
        two    7.0    6.0
    foo one    4.0    1.0
        two    NaN    6.0

    We can also fill missing values using the `fill_value` parameter.

    >>> table = pd.pivot_table(
    ...     df, values="D", index=["A", "B"], columns=["C"], aggfunc="sum", fill_value=0
    ... )
    >>> table
    C        large  small
    A   B
    bar one      4      5
        two      7      6
    foo one      4      1
        two      0      6

    The next example aggregates by taking the mean across multiple columns.

    >>> table = pd.pivot_table(
    ...     df, values=["D", "E"], index=["A", "C"], aggfunc={"D": "mean", "E": "mean"}
    ... )
    >>> table
                    D         E
    A   C
    bar large  5.500000  7.500000
        small  5.500000  8.500000
    foo large  2.000000  4.500000
        small  2.333333  4.333333

    We can also calculate multiple types of aggregations for any given
    value column.

    >>> table = pd.pivot_table(
    ...     df,
    ...     values=["D", "E"],
    ...     index=["A", "C"],
    ...     aggfunc={"D": "mean", "E": ["min", "max", "mean"]},
    ... )
    >>> table
                      D   E
                   mean max      mean  min
    A   C
    bar large  5.500000   9  7.500000    6
        small  5.500000   9  8.500000    8
    foo large  2.000000   5  4.500000    4
        small  2.333333   6  4.333333    2
    """
    index = _convert_by(index)
    columns = _convert_by(columns)
    if isinstance(aggfunc, list):
        pieces: list[DataFrame] = []
        keys = []
        for func in aggfunc:
            _table = __internal_pivot_table(data, values=values, index=index, columns=columns, fill_value=fill_value, aggfunc=func, margins=margins, dropna=dropna, margins_name=margins_name, observed=observed, sort=sort, kwargs=kwargs)
            pieces.append(_table)
            keys.append(getattr(func, '__name__', func))
        table = concat(pieces, keys=keys, axis=1)
        return table.__finalize__(data, method='pivot_table')
    table = __internal_pivot_table(data, values, index, columns, aggfunc, fill_value, margins, dropna, margins_name, observed, sort, kwargs)
    return table.__finalize__(data, method='pivot_table')

def __internal_pivot_table(
    data: DataFrame, 
    values: Optional[Union[Hashable, list[Hashable]]], 
    index: list[Hashable], 
    columns: list[Hashable], 
    aggfunc: Union[str, Callable, dict[Hashable, Union[str, Callable, list[Union[str, Callable]]]], 
    fill_value: Optional[Any], 
    margins: bool, 
    dropna: bool, 
    margins_name: str, 
    observed: bool, 
    sort: bool, 
    kwargs: dict[str, Any]
) -> DataFrame:
    """
    Helper of :func:`pandas.pivot_table` for any non-list ``aggfunc``.
    """
    keys = index + columns
    values_passed = values is not None
    if values_passed:
        if is_list_like(values):
            values_multi = True
            values = list(values)
        else:
            values_multi = False
            values = [values]
        for i in values:
            if i not in data:
                raise KeyError(i)
        to_filter = []
        for x in keys + values:
            if isinstance(x, Grouper):
                x = x.key
            try:
                if x in data:
                    to_filter.append(x)
            except TypeError:
                pass
        if len(to_filter) < len(data.columns):
            data = data[to_filter]
    else:
        values = data.columns
        for key in keys:
            try:
                values = values.drop(key)
            except (TypeError, ValueError, KeyError):
                pass
        values = list(values)
    grouped = data.groupby(keys, observed=observed, sort=sort, dropna=dropna)
    agged = grouped.agg(aggfunc, **kwargs)
    if dropna and isinstance(agged, ABCDataFrame) and len(agged.columns):
        agged = agged.dropna(how='all')
    table = agged
    if table.index.nlevels > 1 and index:
        index_names = agged.index.names[:len(index)]
        to_unstack = []
        for i in range(len(index), len(keys)):
            name = agged.index.names[i]
            if name is None or name in index_names:
                to_unstack.append(i)
            else:
                to_unstack.append(name)
        table = agged.unstack(to_unstack, fill_value=fill_value)
    if not dropna:
        if isinstance(table.index, MultiIndex):
            m = MultiIndex.from_product(table.index.levels, names=table.index.names)
            table = table.reindex(m, axis=0, fill_value=fill_value)
        if isinstance(table.columns, MultiIndex):
            m = MultiIndex.from_product(table.columns.levels, names=table.columns.names)
            table = table.reindex(m, axis=1, fill_value=fill_value)
    if sort is True and isinstance(table, ABCDataFrame):
        table = table.sort_index(axis=1)
    if fill_value is not None:
        table = table.fillna(fill_value)
        if aggfunc is len and (not observed) and lib.is_integer(fill_value):
            table = table.astype(np.int64)
    if margins:
        if dropna:
            data = data[data.notna().all(axis=1)]
        table = _add_margins(table, data, values, rows=index, cols=columns, aggfunc=aggfunc, kwargs=kwargs, observed=dropna, margins_name=margins_name, fill_value=fill_value)
    if values_passed and (not values_multi) and (table.columns.nlevels > 1):
        table.columns = table.columns.droplevel(0)
    if len(index) == 0 and len(columns) > 0:
        table = table.T
    if isinstance(table, ABCDataFrame) and dropna:
        table = table.dropna(how='all', axis=1)
    return table

def _add_margins(
    table: Union[DataFrame, Series], 
    data: DataFrame, 
    values: list[Hashable], 
    rows: list[Hashable], 
    cols: list[Hashable], 
    aggfunc: Union[str, Callable, dict[Hashable, Union[str, Callable, list[Union[str, Callable]]]], 
    kwargs: dict[str, Any], 
    observed: bool, 
    margins_name: Hashable = 'All', 
    fill_value: Optional[Any] = None
) -> Union[DataFrame, Series]:
    if not isinstance(margins_name, str):
        raise ValueError('margins_name argument must be a string')
    msg = f'Conflicting name "{margins_name}" in margins'
    for level in table.index.names:
        if margins_name in table.index.get_level_values(level):
            raise ValueError(msg)
    grand_margin = _compute_grand_margin(data, values, aggfunc, kwargs, margins_name)
    if table.ndim == 2:
        for level in table.columns.names[1:]:
            if margins_name in table.columns.get_level_values(level):
                raise ValueError(msg)
    key: Union[str, tuple[str, ...]]
    if len(rows) > 1:
        key = (margins_name,) + ('',) * (len(rows) - 1)
    else:
        key = margins_name
    if not values and isinstance(table, ABCSeries):
        return table._append(table._constructor({key: grand_margin[margins_name]}))
    elif values:
        marginal_result_set = _generate_marginal_results(table, data, values, rows, cols, aggfunc, kwargs, observed, margins_name)
        if not isinstance(marginal_result_set, tuple):
            return marginal_result_set
        (result, margin_keys, row_margin) = marginal_result_set
    else:
        assert isinstance(table, ABCDataFrame)
        marginal_result_set = _generate_marginal_results_without_values(table, data, rows, cols, aggfunc, kwargs, observed, margins_name)
        if not isinstance(marginal_result_set, tuple):
            return marginal_result_set
        (result, margin_keys, row_margin) = marginal_result_set
    row_margin = row_margin.reindex(result.columns, fill_value=fill_value)
    for k in margin_keys:
        if isinstance(k, str):
            row_margin[k] = grand_margin[k]
        else:
            row_margin[k] = grand_margin[k[0]]
    from pandas import DataFrame
    margin_dummy = DataFrame(row_margin, columns=Index([key])).T
    row_names = result.index.names
    for dtype in set(result.dtypes):
        if isinstance(dtype, ExtensionDtype):
            continue
        cols = result.select_dtypes([dtype]).columns
        margin_dummy[cols] = margin_dummy[cols].apply(maybe_downcast_to_dtype, args=(dtype,))
    result = result._append(margin_dummy)
    result.index.names = row_names
    return result

def _compute_grand_margin(
    data: DataFrame, 
    values: list[Hashable], 
    aggfunc: Union[str, Callable, dict[Hashable, Union[str, Callable, list[Union[str, Callable]]]], 
    kwargs: dict[str, Any], 
    margins_name: Hashable = 'All'
) -> dict[Hashable, Any]:
    if values:
        grand_margin = {}
        for (k, v) in data[values].items():
            try:
                if isinstance(aggfunc, str):
                    grand_margin[k] = getattr(v, aggfunc)(**kwargs)
                elif isinstance(aggfunc, dict):
                    if isinstance(aggfunc[k], str):
                        grand_margin[k] = getattr(v, aggfunc[k])(**kwargs)
                    else:
                        grand_margin[k] = aggfunc[k](v, **kwargs)
                else:
                    grand_margin[k] = aggfunc(v, **kwargs)
            except TypeError:
                pass
        return grand_margin
    else:
        return {margins_name: aggfunc(data.index, **kwargs)}

def _generate_marginal_results(
    table: Union[DataFrame, Series], 
    data: DataFrame, 
    values: list[Hashable], 
    rows: list[Hashable], 
    cols: list[Hashable], 
    aggfunc: Union[str, Callable, dict[Hashable, Union[str, Callable, list[Union[str, Callable]]]], 
    kwargs: dict[str, Any], 
    observed: bool, 
    margins_name: Hashable = 'All'
) -> Union[Union[DataFrame, Series], tuple[Union[DataFrame, Series], Union[list, Index], Union[