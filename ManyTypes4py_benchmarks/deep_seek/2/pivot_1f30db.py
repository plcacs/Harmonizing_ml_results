from __future__ import annotations
import itertools
from typing import TYPE_CHECKING, Literal, cast, Any, Optional, Union, Dict, List, Tuple, Sequence, Set
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
    from pandas._typing import (
        AggFuncType, 
        AggFuncTypeBase, 
        AggFuncTypeDict, 
        IndexLabel, 
        SequenceNotStr,
        Axis,
        Level,
    )
    from pandas import DataFrame

def pivot_table(
    data: DataFrame,
    values: Optional[Union[SequenceNotStr[Hashable], Hashable]] = None,
    index: Optional[Union[SequenceNotStr[Hashable], Hashable]] = None,
    columns: Optional[Union[SequenceNotStr[Hashable], Hashable]] = None,
    aggfunc: Union[str, Callable, List[Union[str, Callable]], Dict[Hashable, Union[str, Callable, List[Union[str, Callable]]]] = 'mean',
    fill_value: Optional[Any] = None,
    margins: bool = False,
    dropna: bool = True,
    margins_name: str = 'All',
    observed: bool = True,
    sort: bool = True,
    **kwargs: Any
) -> DataFrame:
    index = _convert_by(index)
    columns = _convert_by(columns)
    if isinstance(aggfunc, list):
        pieces = []
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
    values: Optional[Union[SequenceNotStr[Hashable], Hashable]],
    index: List[Hashable],
    columns: List[Hashable],
    aggfunc: Union[str, Callable, Dict[Hashable, Union[str, Callable]]],
    fill_value: Optional[Any],
    margins: bool,
    dropna: bool,
    margins_name: str,
    observed: bool,
    sort: bool,
    kwargs: Dict[str, Any]
) -> DataFrame:
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
    values: List[Hashable],
    rows: List[Hashable],
    cols: List[Hashable],
    aggfunc: Union[str, Callable, Dict[Hashable, Union[str, Callable]]],
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
    values: List[Hashable],
    aggfunc: Union[str, Callable, Dict[Hashable, Union[str, Callable]]],
    kwargs: Dict[str, Any],
    margins_name: str = 'All'
) -> Dict[Hashable, Any]:
    if values:
        grand_margin = {}
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
    table: Union[DataFrame, Series],
    data: DataFrame,
    values: List[Hashable],
    rows: List[Hashable],
    cols: List[Hashable],
    aggfunc: Union[str, Callable, Dict[Hashable, Union[str, Callable]]],
    kwargs: Dict[str, Any],
    observed: bool,
    margins_name: str = 'All'
) -> Union[DataFrame, Tuple[DataFrame, List[Union[str, Tuple[str, ...]]], Series]:
    if len(cols) > 0:
        table_pieces = []
        margin_keys = []

        def _all_key(key: Hashable) -> Tuple[Hashable, ...]:
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
    rows: List[Hashable],
    cols: List[Hashable],
    aggfunc: Union[str, Callable],
    kwargs: Dict[str, Any],
    observed: bool,
    margins_name: str = 'All'
) -> Union[DataFrame, Tuple[DataFrame, List[Union[str, Tuple[str, ...]]], Series]]:
    if len(cols) > 0:
        margin_keys = []

        def _all_key() -> Union[str, Tuple[str, ...]]:
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

def _convert_by(by: Optional[Union[SequenceNotStr[Hashable], Hashable]]) -> List[Hashable]:
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
    columns: Union[Hashable, Sequence[Hashable]],
    index: Union[Hashable, Sequence[Hashable]] = lib.no_default,
    values: Union[Hashable, Sequence[Hashable]] = lib.no_default
) -> DataFrame:
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
    index: Union[Sequence[Any], Series, List[Union[Sequence[Any], Series]]],
    columns: Union[Sequence[Any], Series, List[Union[Sequence[Any], Series]]],
    values: Optional[Union[Sequence[Any], Series]] = None,
    rownames: Optional[Sequence[str]] = None,
    colnames: Optional[Sequence[str]] = None,
    aggfunc: Optional[Union[str, Callable]] = None,
    margins: bool = False,
    margins_name: str = 'All',
    dropna: bool = True,
    normalize: Union[bool, str, int] = False
) -> DataFrame:
    if values is None and aggfunc is not None:
        raise ValueError('aggfunc cannot be used without values.')
    if values is not None and aggfunc is None:
        raise ValueError('values cannot be used without an aggfunc.')
    if not is_nested_list_like(index):
        index = [index]
    if not is_nested_list_like(columns):
        columns = [columns]
    common_idx = None
    pass_objs = [x for x in index +