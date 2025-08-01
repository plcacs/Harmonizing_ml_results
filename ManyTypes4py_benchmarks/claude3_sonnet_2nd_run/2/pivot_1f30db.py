from __future__ import annotations
import itertools
from typing import TYPE_CHECKING, Literal, cast, Dict, List, Optional, Tuple, Union, Any, Callable, Hashable, Sequence, TypeVar, overload
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
    values: Optional[Union[str, Sequence[str]]] = None,
    index: Optional[Union[str, Grouper, np.ndarray, List]] = None,
    columns: Optional[Union[str, Grouper, np.ndarray, List]] = None,
    aggfunc: Union[str, Callable, List[Callable], Dict[str, Callable]] = 'mean',
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
        pieces: List[DataFrame] = []
        keys: List[str] = []
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
    values: Optional[Union[str, Sequence[str]]],
    index: List,
    columns: List,
    aggfunc: Union[str, Callable, Dict[str, Callable]],
    fill_value: Optional[Any],
    margins: bool,
    dropna: bool,
    margins_name: str,
    observed: bool,
    sort: bool,
    kwargs: Dict[str, Any]
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
        to_filter: List[Union[str, Grouper]] = []
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
        to_unstack: List[Union[int, str]] = []
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
    values: List[str],
    rows: List,
    cols: List,
    aggfunc: Union[str, Callable, Dict[str, Callable]],
    kwargs: Dict[str, Any],
    observed: bool,
    margins_name: str = 'All',
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
        result, margin_keys, row_margin = marginal_result_set
    else:
        assert isinstance(table, ABCDataFrame)
        marginal_result_set = _generate_marginal_results_without_values(table, data, rows, cols, aggfunc, kwargs, observed, margins_name)
        if not isinstance(marginal_result_set, tuple):
            return marginal_result_set
        result, margin_keys, row_margin = marginal_result_set
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
    values: List[str],
    aggfunc: Union[str, Callable, Dict[str, Callable]],
    kwargs: Dict[str, Any],
    margins_name: str = 'All'
) -> Dict[str, Any]:
    if values:
        grand_margin: Dict[str, Any] = {}
        for k, v in data[values].items():
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
    table: DataFrame,
    data: DataFrame,
    values: List[str],
    rows: List,
    cols: List,
    aggfunc: Union[str, Callable, Dict[str, Callable]],
    kwargs: Dict[str, Any],
    observed: bool,
    margins_name: str = 'All'
) -> Union[DataFrame, Tuple[DataFrame, List, Series]]:
    if len(cols) > 0:
        table_pieces: List[DataFrame] = []
        margin_keys: List = []

        def _all_key(key: Any) -> Tuple:
            return (key, margins_name) + ('',) * (len(cols) - 1)
        if len(rows) > 0:
            margin = data[rows + values].groupby(rows, observed=observed).agg(aggfunc, **kwargs)
            cat_axis = 1
            for key, piece in table.T.groupby(level=0, observed=observed):
                piece = piece.T
                all_key = _all_key(key)
                piece[all_key] = margin[key]
                table_pieces.append(piece)
                margin_keys.append(all_key)
        else:
            margin = data[cols[:1] + values].groupby(cols[:1], observed=observed).agg(aggfunc, **kwargs).T
            cat_axis = 0
            for key, piece in table.groupby(level=0, observed=observed):
                if len(cols) > 1:
                    all_key = _all_key(key)
                else:
                    all_key = margins_name
                table_pieces.append(piece)
                transformed_piece = margin[key].to_frame().T
                if isinstance(piece.index, MultiIndex):
                    transformed_piece.index = MultiIndex.from_tuples([all_key], names=piece.index.names + [None])
                else:
                    transformed_piece.index = Index([all_key], name=piece.index.name)
                table_pieces.append(transformed_piece)
                margin_keys.append(all_key)
        if not table_pieces:
            return table
        else:
            result = concat(table_pieces, axis=cat_axis)
        if len(rows) == 0:
            return result
    else:
        result = table
        margin_keys = table.columns
    if len(cols) > 0:
        row_margin = data[cols + values].groupby(cols, observed=observed).agg(aggfunc, **kwargs)
        row_margin = row_margin.stack()
        new_order_indices = itertools.chain([len(cols)], range(len(cols)))
        new_order_names = [row_margin.index.names[i] for i in new_order_indices]
        row_margin.index = row_margin.index.reorder_levels(new_order_names)
    else:
        row_margin = data._constructor_sliced(np.nan, index=result.columns)
    return (result, margin_keys, row_margin)

def _generate_marginal_results_without_values(
    table: DataFrame,
    data: DataFrame,
    rows: List,
    cols: List,
    aggfunc: Union[str, Callable, Dict[str, Callable]],
    kwargs: Dict[str, Any],
    observed: bool,
    margins_name: str = 'All'
) -> Union[DataFrame, Tuple[DataFrame, List, Series]]:
    if len(cols) > 0:
        margin_keys: List = []

        def _all_key() -> Union[str, Tuple]:
            if len(cols) == 1:
                return margins_name
            return (margins_name,) + ('',) * (len(cols) - 1)
        if len(rows) > 0:
            margin = data.groupby(rows, observed=observed)[rows].apply(aggfunc, **kwargs)
            all_key = _all_key()
            table[all_key] = margin
            result = table
            margin_keys.append(all_key)
        else:
            margin = data.groupby(level=0, observed=observed).apply(aggfunc, **kwargs)
            all_key = _all_key()
            table[all_key] = margin
            result = table
            margin_keys.append(all_key)
            return result
    else:
        result = table
        margin_keys = table.columns
    if len(cols):
        row_margin = data.groupby(cols, observed=observed)[cols].apply(aggfunc, **kwargs)
    else:
        row_margin = Series(np.nan, index=result.columns)
    return (result, margin_keys, row_margin)

def _convert_by(by: Optional[Union[str, Grouper, np.ndarray, List]]) -> List:
    if by is None:
        by = []
    elif is_scalar(by) or isinstance(by, (np.ndarray, Index, ABCSeries, Grouper)) or callable(by):
        by = [by]
    else:
        by = list(by)
    return by

def pivot(
    data: DataFrame,
    *,
    columns: Union[str, List[str]],
    index: Union[str, List[str], lib._NoDefault] = lib.no_default,
    values: Union[str, List[str], lib._NoDefault] = lib.no_default
) -> DataFrame:
    """
    Return reshaped DataFrame organized by given index / column values.

    Reshape data (produce a "pivot" table) based on column values. Uses
    unique values from specified `index` / `columns` to form axes of the
    resulting DataFrame. This function does not support data
    aggregation, multiple values will result in a MultiIndex in the
    columns. See the :ref:`User Guide <reshaping>` for more on reshaping.

    Parameters
    ----------
    data : DataFrame
        Input pandas DataFrame object.
    columns : str or object or a list of str
        Column to use to make new frame's columns.
    index : str or object or a list of str, optional
        Column to use to make new frame's index. If not given, uses existing index.
    values : str, object or a list of the previous, optional
        Column(s) to use for populating new frame's values. If not
        specified, all remaining columns will be used and the result will
        have hierarchically indexed columns.

    Returns
    -------
    DataFrame
        Returns reshaped DataFrame.

    Raises
    ------
    ValueError:
        When there are any `index`, `columns` combinations with multiple
        values. `DataFrame.pivot_table` when you need to aggregate.

    See Also
    --------
    DataFrame.pivot_table : Generalization of pivot that can handle
        duplicate values for one index/column pair.
    DataFrame.unstack : Pivot based on the index values instead of a
        column.
    wide_to_long : Wide panel to long format. Less flexible but more
        user-friendly than melt.

    Notes
    -----
    For finer-tuned control, see hierarchical indexing documentation along
    with the related stack/unstack methods.

    Reference :ref:`the user guide <reshaping.pivot>` for more examples.

    Examples
    --------
    >>> df = pd.DataFrame(
    ...     {
    ...         "foo": ["one", "one", "one", "two", "two", "two"],
    ...         "bar": ["A", "B", "C", "A", "B", "C"],
    ...         "baz": [1, 2, 3, 4, 5, 6],
    ...         "zoo": ["x", "y", "z", "q", "w", "t"],
    ...     }
    ... )
    >>> df
        foo   bar  baz  zoo
    0   one   A    1    x
    1   one   B    2    y
    2   one   C    3    z
    3   two   A    4    q
    4   two   B    5    w
    5   two   C    6    t

    >>> df.pivot(index="foo", columns="bar", values="baz")
    bar  A   B   C
    foo
    one  1   2   3
    two  4   5   6

    >>> df.pivot(index="foo", columns="bar")["baz"]
    bar  A   B   C
    foo
    one  1   2   3
    two  4   5   6

    >>> df.pivot(index="foo", columns="bar", values=["baz", "zoo"])
          baz       zoo
    bar   A  B  C   A  B  C
    foo
    one   1  2  3   x  y  z
    two   4  5  6   q  w  t

    You could also assign a list of column names or a list of index names.

    >>> df = pd.DataFrame(
    ...     {
    ...         "lev1": [1, 1, 1, 2, 2, 2],
    ...         "lev2": [1, 1, 2, 1, 1, 2],
    ...         "lev3": [1, 2, 1, 2, 1, 2],
    ...         "lev4": [1, 2, 3, 4, 5, 6],
    ...         "values": [0, 1, 2, 3, 4, 5],
    ...     }
    ... )
    >>> df
        lev1 lev2 lev3 lev4 values
    0   1    1    1    1    0
    1   1    1    2    2    1
    2   1    2    1    3    2
    3   2    1    2    4    3
    4   2    1    1    5    4
    5   2    2    2    6    5

    >>> df.pivot(index="lev1", columns=["lev2", "lev3"], values="values")
    lev2    1         2
    lev3    1    2    1    2
    lev1
    1     0.0  1.0  2.0  NaN
    2     4.0  3.0  NaN  5.0

    >>> df.pivot(index=["lev1", "lev2"], columns=["lev3"], values="values")
          lev3    1    2
    lev1  lev2
       1     1  0.0  1.0
             2  2.0  NaN
       2     1  4.0  3.0
             2  NaN  5.0

    A ValueError is raised if there are any duplicates.

    >>> df = pd.DataFrame(
    ...     {
    ...         "foo": ["one", "one", "two", "two"],
    ...         "bar": ["A", "A", "B", "C"],
    ...         "baz": [1, 2, 3, 4],
    ...     }
    ... )
    >>> df
       foo bar  baz
    0  one   A    1
    1  one   A    2
    2  two   B    3
    3  two   C    4

    Notice that the first two rows are the same for our `index`
    and `columns` arguments.

    >>> df.pivot(index="foo", columns="bar", values="baz")
    Traceback (most recent call last):
       ...
    ValueError: Index contains duplicate entries, cannot reshape
    """
    columns_listlike = com.convert_to_list_like(columns)
    if any((name is None for name in data.index.names)):
        data = data.copy(deep=False)
        data.index.names = [name if name is not None else lib.no_default for name in data.index.names]
    if values is lib.no_default:
        if index is not lib.no_default:
            cols = com.convert_to_list_like(index)
        else:
            cols = []
        append = index is lib.no_default
        indexed = data.set_index(cols + columns_listlike, append=append)
    else:
        if index is lib.no_default:
            if isinstance(data.index, MultiIndex):
                index_list = [data.index.get_level_values(i) for i in range(data.index.nlevels)]
            else:
                index_list = [data._constructor_sliced(data.index, name=data.index.name)]
        else:
            index_list = [data[idx] for idx in com.convert_to_list_like(index)]
        data_columns = [data[col] for col in columns_listlike]
        index_list.extend(data_columns)
        multiindex = MultiIndex.from_arrays(index_list)
        if is_list_like(values) and (not isinstance(values, tuple)):
            indexed = data._constructor(data[values]._values, index=multiindex, columns=cast('SequenceNotStr', values))
        else:
            indexed = data._constructor_sliced(data[values]._values, index=multiindex)
    result = cast('DataFrame', indexed.unstack(columns_listlike))
    result.index.names = [name if name is not lib.no_default else None for name in result.index.names]
    return result

def crosstab(
    index: Union[np.ndarray, Series, List],
    columns: Union[np.ndarray, Series, List],
    values: Optional[np.ndarray] = None,
    rownames: Optional[Sequence[str]] = None,
    colnames: Optional[Sequence[str]] = None,
    aggfunc: Optional[Callable] = None,
    margins: bool = False,
    margins_name: str = 'All',
    dropna: bool = True,
    normalize: Union[bool, str, int] = False
) -> DataFrame:
    """
    Compute a simple cross tabulation of two (or more) factors.

    By default, computes a frequency table of the factors unless an
    array of values and an aggregation function are passed.

    Parameters
    ----------
    index : array-like, Series, or list of arrays/Series
        Values to group by in the rows.
    columns : array-like, Series, or list of arrays/Series
        Values to group by in the columns.
    values : array-like, optional
        Array of values to aggregate according to the factors.
        Requires `aggfunc` be specified.
    rownames : sequence, default None
        If passed, must match number of row arrays passed.
    colnames : sequence, default None
        If passed, must match number of column arrays passed.
    aggfunc : function, optional
        If specified, requires `values` be specified as well.
    margins : bool, default False
        Add row/column margins (subtotals).
    margins_name : str, default 'All'
        Name of the row/column that will contain the totals
        when margins is True.
    dropna : bool, default True
        Do not include columns whose entries are all NaN.
    normalize : bool, {'all', 'index', 'columns'}, or {0,1}, default False
        Normalize by dividing all values by the sum of values.

        - If passed 'all' or `True`, will normalize over all values.
        - If passed 'index' will normalize over each row.
        - If passed 'columns' will normalize over each column.
        - If margins is `True`, will also normalize margin values.

    Returns
    -------
    DataFrame
        Cross tabulation of the data.

    See Also
    --------
    DataFrame.pivot : Reshape data based on column values.
    pivot_table : Create a pivot table as a DataFrame.

    Notes
    -----
    Any Series passed will have their name attributes used unless row or column
    names for the cross-tabulation are specified.

    Any input passed containing Categorical data will have **all** of its
    categories included in the cross-tabulation, even if the actual data does
    not contain any instances of a particular category.

    In the event that there aren't overlapping indexes an empty DataFrame will
    be returned.

    Reference :ref:`the user guide <reshaping.crosstabulations>` for more examples.

    Examples
    --------
    >>> a = np.array(
    ...     [
    ...         "foo",
    ...         "foo",
    ...         "foo",
    ...         "foo",
    ...         "bar",
    ...         "bar",
    ...         "bar",
    ...         "bar",
    ...         "foo",
    ...         "foo",
    ...         "foo",
    ...     ],
    ...     dtype=object,
    ... )
    >>> b = np.array(
    ...     [
    ...         "one",
    ...         "one",
    ...         "one",
    ...         "two",
    ...         "one",
    ...         "one",
    ...         "one",
    ...         "two",
    ...         "two",
    ...         "two",
    ...         "one",
    ...     ],
    ...     dtype=object,
    ... )
    >>> c = np.array(
    ...     [
    ...         "dull",
    ...         "dull",
    ...         "shiny",
    ...         "dull",
    ...         "dull",
    ...         "shiny",
    ...         "shiny",
    ...         "dull",
    ...         "shiny",
    ...         "shiny",
    ...         "shiny",
    ...     ],
    ...     dtype=object,
    ... )
    >>> pd.crosstab(a, [b, c], rownames=["a"], colnames=["b", "c"])
    b   one        two
    c   dull shiny dull shiny
    a
    bar    1     2    1     0
    foo    2     2    1     2

    Here 'c' and 'f' are not represented in the data and will not be
    shown in the output because dropna is True by default. Set
    dropna=False to preserve categories with no data.

    >>> foo = pd.Categorical(["a", "b"], categories=["a", "b", "c"])
    >>> bar = pd.Categorical(["d", "e"], categories=["d", "e", "f"])
    >>> pd.crosstab(foo, bar)
    col_0  d  e
    row_0
    a      1  0
    b      0  1
    >>> pd.crosstab(foo, bar, dropna=False)
    col_0  d  e  f
    row_0
    a      1  0  0
    b      0  1  0
    c      0  0  0
    """
    if values is None and aggfunc is not None:
        raise ValueError('aggfunc cannot be used without values.')
    if values is not None and aggfunc is None:
        raise ValueError('values cannot be used without an aggfunc.')
    if not is_nested_list_like(index):
        index = [index]
    if not is_nested_list_like(columns):
        columns = [columns]
    common_idx = None
    pass_objs = [x for x in index + columns if isinstance(x, (ABCSeries, ABCDataFrame))]
    if pass_objs:
        common_idx = get_objs_combined_axis(pass_objs, intersect=True, sort=False)
    rownames = _get_names(index, rownames, prefix='row')
    colnames = _get_names(columns, colnames, prefix='col')
    rownames_mapper, unique_rownames, colnames_mapper, unique_colnames = _build_names_mapper(rownames, colnames)
    from pandas import DataFrame
    data = {**dict(zip(unique_rownames, index)), **dict(zip(unique_colnames, columns))}
    df = DataFrame(data, index=common_idx)
    if values is None:
        df['__dummy__'] = 0
        kwargs: Dict[str, Any] = {'aggfunc': len, 'fill_value': 0}
    else:
        df['__dummy__'] = values
        kwargs = {'aggfunc': aggfunc}
    table = df.pivot_table('__dummy__', index=unique_rownames, columns=unique_colnames, margins=margins, margins_name=margins_name, dropna=dropna, observed=dropna, **kwargs)
    if normalize is not False:
        table = _normalize(table, normalize=normalize, margins=margins, margins_name=margins_name)
    table = table.rename_axis(index=rownames_mapper, axis=0)
    table = table.rename_axis(columns=colnames_mapper, axis=1)
    return table

def _normalize(
    table: DataFrame,
    normalize: Union[bool, str, int],
    margins: bool,
    margins_name: str = 'All'
) -> DataFrame:
    if not isinstance(normalize, (bool, str)):
        axis_subs = {0: 'index', 1: 'columns'}
        try:
            normalize = axis_subs[normalize]
        except KeyError as err:
            raise ValueError('Not a valid normalize argument') from err
    if margins is False:
        normalizers: Dict[Union[bool, str], Callable[[DataFrame], DataFrame]] = {
            'all': lambda x: x / x.sum(axis=1).sum(axis=0),
            'columns': lambda x: x / x.sum(),
            'index': lambda x: x.div(x.sum(axis=1), axis=0)
        }
        normalizers[True] = normalizers['all']
        try:
            f = normalizers[normalize]
        except KeyError as err:
            raise ValueError('Not a valid normalize argument') from err
        table = f(table)
        table = table.fillna(0)
    elif margins is True:
        table_index = table.index
        table_columns = table.columns
        last_ind_or_col = table.iloc[-1, :].name
        if (margins_name not in last_ind_or_col) & (margins_name != last_ind_or_col):
            raise ValueError(f'{margins_name} not in pivoted DataFrame')
        column_margin = table.iloc[:-1, -1]
        index_margin = table.iloc[-1, :-1]
        table = table.iloc[:-1, :-1]
        table = _normalize(table, normalize=normalize, margins=False)
        if normalize == 'columns':
            column_margin = column_margin / column_margin.sum()
            table = concat([table, column_margin], axis=1)
            table = table.fillna(0)
            table.columns = table_columns
        elif normalize == 'index':
            index_margin = index_margin / index_margin.sum()
            table = table._append(index_margin, ignore_index=True)
            table = table.fillna(0)
            table.index = table_index
        elif normalize == 'all' or normalize is True:
            column_margin = column_margin / column_margin.sum()
            index_margin = index_margin / index_margin.sum()
            index_margin.loc[margins_name] = 1
            table = concat([table, column_margin], axis=1)
            table = table._append(index_margin, ignore_index=True)
            table = table.fillna(0)
            table.index = table_index
            table.columns = table_columns
        else:
            raise ValueError('Not a valid normalize argument')
    else:
        raise ValueError('Not a valid margins argument')
    return table

def _get_names(
    arrs: List,
    names: Optional[Sequence[str]],
    prefix: str = 'row'
) -> List[str]:
    if names is None:
        names = []
        for i, arr in enumerate(arrs):
            if isinstance(arr, ABCSeries) and arr.name is not None:
                names.append(arr.name)
            else:
                names.append(f'{prefix}_{i}')
    else:
        if len(names) != len(arrs):
            raise AssertionError('arrays and names must have the same length')
        if not isinstance(names, list):
            names = list(names)
    return names

def _build_names_mapper(
    rownames: List[str],
    colnames: List[str]
) -> Tuple[Dict[str, str], List[str], Dict[str, str], List[str]]:
    """
    Given the names of a DataFrame's rows and columns, returns a set of unique row
    and column names and mappers that convert to original names.

    A row or column name is replaced if it is duplicate among the rows of the inputs,
    among the columns of the inputs or between the rows and the columns.

    Parameters
    ----------
    rownames: list[str]
    colnames: list[str]

    Returns
    -------
    Tuple(Dict[str, str], List[str], Dict[str, str], List[str])

    rownames_mapper: dict[str, str]
        a dictionary with new row names as keys and original rownames as values
    unique_rownames: list[str]
        a list of rownames with duplicate names replaced by dummy names
    colnames_mapper: dict[str, str]
        a dictionary with new column names as keys and original column names as values
    unique_colnames: list[str]
        a list of column names with duplicate names replaced by dummy names

    """
    dup_names = set(rownames) | set(colnames)
    rownames_mapper: Dict[str, str] = {f'row_{i}': name for i, name in enumerate(rownames) if name in dup_names}
    unique_rownames: List[str] = [f'row_{i}' if name in dup_names else name for i, name in enumerate(rownames)]
    colnames_mapper: Dict[str, str] = {f'col_{i}': name for i, name in enumerate(colnames) if name in dup_names}
    unique_colnames: List[str] = [f'col_{i}' if name in dup_names else name for i, name in enumerate(colnames)]
    return (rownames_mapper, unique_rownames, colnames_mapper, unique_colnames)
