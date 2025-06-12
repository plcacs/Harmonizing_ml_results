from __future__ import annotations
import itertools
from typing import TYPE_CHECKING, Literal, cast, Union, Optional, List, Dict, Tuple
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
    values: Optional[Union[List[str], str]] = None,
    index: Optional[Union[str, Grouper, np.ndarray, List[Union[str, Grouper, np.ndarray]]]] = None,
    columns: Optional[Union[str, Grouper, np.ndarray, List[Union[str, Grouper, np.ndarray]]]] = None,
    aggfunc: Union[str, Callable, List[Callable], Dict[str, Union[str, Callable, List[Callable]]]] = 'mean',
    fill_value: Optional[Union[int, float, str]] = None,
    margins: bool = False,
    dropna: bool = True,
    margins_name: str = 'All',
    observed: bool = True,
    sort: bool = True,
    **kwargs
) -> DataFrame:
    index = _convert_by(index)
    columns = _convert_by(columns)
    if isinstance(aggfunc, list):
        pieces = []
        keys = []
        for func in aggfunc:
            _table = __internal_pivot_table(
                data, values=values, index=index, columns=columns, fill_value=fill_value,
                aggfunc=func, margins=margins, dropna=dropna, margins_name=margins_name,
                observed=observed, sort=sort, kwargs=kwargs
            )
            pieces.append(_table)
            keys.append(getattr(func, '__name__', func))
        table = concat(pieces, keys=keys, axis=1)
        return table.__finalize__(data, method='pivot_table')
    table = __internal_pivot_table(
        data, values, index, columns, aggfunc, fill_value, margins, dropna,
        margins_name, observed, sort, kwargs
    )
    return table.__finalize__(data, method='pivot_table')

def __internal_pivot_table(
    data: DataFrame,
    values: Optional[Union[List[str], str]],
    index: List[Union[str, Grouper, np.ndarray]],
    columns: List[Union[str, Grouper, np.ndarray]],
    aggfunc: Union[str, Callable],
    fill_value: Optional[Union[int, float, str]],
    margins: bool,
    dropna: bool,
    margins_name: str,
    observed: bool,
    sort: bool,
    kwargs: dict
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
        table = _add_margins(
            table, data, values, rows=index, cols=columns, aggfunc=aggfunc,
            kwargs=kwargs, observed=dropna, margins_name=margins_name, fill_value=fill_value
        )
    if values_passed and (not values_multi) and (table.columns.nlevels > 1):
        table.columns = table.columns.droplevel(0)
    if len(index) == 0 and len(columns) > 0:
        table = table.T
    if isinstance(table, ABCDataFrame) and dropna:
        table = table.dropna(how='all', axis=1)
    return table

def _add_margins(
    table: DataFrame,
    data: DataFrame,
    values: List[str],
    rows: List[Union[str, Grouper, np.ndarray]],
    cols: List[Union[str, Grouper, np.ndarray]],
    aggfunc: Union[str, Callable],
    kwargs: dict,
    observed: bool,
    margins_name: str = 'All',
    fill_value: Optional[Union[int, float, str]] = None
) -> DataFrame:
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
        marginal_result_set = _generate_marginal_results(
            table, data, values, rows, cols, aggfunc, kwargs, observed, margins_name
        )
        if not isinstance(marginal_result_set, tuple):
            return marginal_result_set
        result, margin_keys, row_margin = marginal_result_set
    else:
        assert isinstance(table, ABCDataFrame)
        marginal_result_set = _generate_marginal_results_without_values(
            table, data, rows, cols, aggfunc, kwargs, observed, margins_name
        )
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
    aggfunc: Union[str, Callable],
    kwargs: dict,
    margins_name: str = 'All'
) -> Dict[str, Union[int, float]]:
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
    table: DataFrame,
    data: DataFrame,
    values: List[str],
    rows: List[Union[str, Grouper, np.ndarray]],
    cols: List[Union[str, Grouper, np.ndarray]],
    aggfunc: Union[str, Callable],
    kwargs: dict,
    observed: bool,
    margins_name: str = 'All'
) -> Union[DataFrame, Tuple[DataFrame, List[Union[str, Tuple[str, ...]]], Series]]:
    if len(cols) > 0:
        table_pieces = []
        margin_keys = []

        def _all_key(key: str) -> Union[str, Tuple[str, ...]]:
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
    rows: List[Union[str, Grouper, np.ndarray]],
    cols: List[Union[str, Grouper, np.ndarray]],
    aggfunc: Union[str, Callable],
    kwargs: dict,
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

def _convert_by(by: Optional[Union[str, np.ndarray, Index, ABCSeries, Grouper, Callable]]) -> List[Union[str, np.ndarray, Index, ABCSeries, Grouper, Callable]]:
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
    index: Union[str, List[str]] = lib.no_default,
    values: Union[str, List[str]] = lib.no_default
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
    index: Union[np.ndarray, Series, List[Union[np.ndarray, Series]]],
    columns: Union[np.ndarray, Series, List[Union[np.ndarray, Series]]],
    values: Optional[np.ndarray] = None,
    rownames: Optional[List[str]] = None,
    colnames: Optional[List[str]] = None,
    aggfunc: Optional[Callable] = None,
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
        kwargs = {'aggfunc': len, 'fill_value': 0}
    else:
        df['__dummy__'] = values
        kwargs = {'aggfunc': aggfunc}
    table = df.pivot_table(
        '__dummy__', index=unique_rownames, columns=unique_colnames, margins=margins,
        margins_name=margins_name, dropna=dropna, observed=dropna, **kwargs
    )
    if normalize is not False:
        table = _normalize(table, normalize=normalize, margins=margins, margins_name=margins_name)
    table = table.rename_axis(index=rownames_mapper, axis=0)
    table = table.rename_axis(columns=colnames_mapper, axis=1)
    return table

def _normalize(
    table: DataFrame,
    normalize: Union[bool, str],
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
        normalizers = {
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
    arrs: List[Union[np.ndarray, Series]],
    names: Optional[List[str]],
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
    dup_names = set(rownames) | set(colnames)
    rownames_mapper = {f'row_{i}': name for i, name in enumerate(rownames) if name in dup_names}
    unique_rownames = [f'row_{i}' if name in dup_names else name for i, name in enumerate(rownames)]
    colnames_mapper = {f'col_{i}': name for i, name in enumerate(colnames) if name in dup_names}
    unique_colnames = [f'col_{i}' if name in dup_names else name for i, name in enumerate(colnames)]
    return (rownames_mapper, unique_rownames, colnames_mapper, unique_colnames)
