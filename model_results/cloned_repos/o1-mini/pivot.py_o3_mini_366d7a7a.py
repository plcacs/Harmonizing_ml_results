from __future__ import annotations

import itertools
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Hashable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numpy as np

from pandas._libs import lib

from pandas.core.dtypes.cast import maybe_downcast_to_dtype
from pandas.core.dtypes.common import (
    is_list_like,
    is_nested_list_like,
    is_scalar,
)
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)

import pandas.core.common as com
from pandas.core.groupby import Grouper
from pandas.core.indexes.api import (
    Index,
    MultiIndex,
    get_objs_combined_axis,
)
from pandas.core.reshape.concat import concat
from pandas.core.series import Series

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Hashable,
    )

    from pandas._typing import (
        AggFuncType,
        AggFuncTypeBase,
        AggFuncTypeDict,
        IndexLabel,
        SequenceNotStr,
    )

    from pandas import DataFrame


def pivot_table(
    data: DataFrame,
    values: Optional[Union[str, List[str], Series, Index]] = None,
    index: Optional[Union[str, List[str], Series, Index, Grouper]] = None,
    columns: Optional[Union[str, List[str], Series, Index, Grouper]] = None,
    aggfunc: AggFuncType = "mean",
    fill_value: Optional[Union[int, float, str]] = None,
    margins: bool = False,
    dropna: bool = True,
    margins_name: Hashable = "All",
    observed: bool = True,
    sort: bool = True,
    **kwargs: Any,
) -> DataFrame:
    """
    Create a spreadsheet-style pivot table as a DataFrame.

    [Docstring omitted for brevity]
    """
    index_converted: List[Union[str, Series, Index, Grouper]] = _convert_by(index)
    columns_converted: List[Union[str, Series, Index, Grouper]] = _convert_by(columns)

    if isinstance(aggfunc, list):
        pieces: List[DataFrame] = []
        keys: List[str] = []
        for func in aggfunc:
            _table: DataFrame = __internal_pivot_table(
                data,
                values=values,
                index=index_converted,
                columns=columns_converted,
                fill_value=fill_value,
                aggfunc=func,
                margins=margins,
                dropna=dropna,
                margins_name=margins_name,
                observed=observed,
                sort=sort,
                kwargs=kwargs,
            )
            pieces.append(_table)
            keys.append(getattr(func, "__name__", str(func)))

        table: DataFrame = cast(DataFrame, concat(pieces, keys=keys, axis=1))
        return cast(DataFrame, table.__finalize__(data, method="pivot_table"))

    table: DataFrame = __internal_pivot_table(
        data,
        values,
        index_converted,
        columns_converted,
        aggfunc,
        fill_value,
        margins,
        dropna,
        margins_name,
        observed,
        sort,
        kwargs,
    )
    return cast(DataFrame, table.__finalize__(data, method="pivot_table"))


def __internal_pivot_table(
    data: DataFrame,
    values: Optional[Union[str, List[str], Series, Index]],
    index: List[Union[str, Series, Index, Grouper]],
    columns: List[Union[str, Series, Index, Grouper]],
    aggfunc: Union[AggFuncTypeBase, AggFuncTypeDict],
    fill_value: Optional[Union[int, float, str]],
    margins: bool,
    dropna: bool,
    margins_name: Hashable,
    observed: bool,
    sort: bool,
    kwargs: Dict[str, Any],
) -> DataFrame:
    """
    Helper of :func:`pandas.pivot_table` for any non-list ``aggfunc``.
    """
    keys: List[Union[str, Series, Index, Grouper]] = index + columns

    values_passed: bool = values is not None
    if values_passed:
        if is_list_like(values):
            values_multi: bool = True
            values_converted: List[str] = list(cast(List[str], values))
        else:
            values_multi = False
            values_converted = [cast(str, values)]

        # GH14938 Make sure value labels are in data
        for i in values_converted:
            if i not in data:
                raise KeyError(i)

        to_filter: List[str] = []
        for x in keys + values_converted:
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
        values_multi = False
        values_converted = list(data.columns)
        for key in keys:
            try:
                values_converted.remove(key)
            except (TypeError, ValueError, KeyError):
                pass
        values_converted = list(values_converted)

    if values_passed:
        values = values_converted
    else:
        values = values_converted

    grouped: DataFrameGroupBy = data.groupby(keys, observed=observed, sort=sort, dropna=dropna)
    agged: DataFrame | DataFrameGroupBy = grouped.agg(aggfunc, **kwargs)

    if dropna and isinstance(agged, ABCDataFrame) and len(agged.columns):
        agged = agged.dropna(how="all")

    table: DataFrame | Series = agged

    # GH17038, this check should only happen if index is defined (not None)
    if isinstance(table.index, MultiIndex) and table.index.nlevels > 1 and index:
        # Related GH #17123
        # If index_names are integers, determine whether the integers refer
        # to the level position or name.
        index_names = agged.index.names[: len(index)]
        to_unstack: List[Union[str, int]] = []
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

    if sort and isinstance(table, ABCDataFrame):
        table = table.sort_index(axis=1)

    if fill_value is not None:
        table = table.fillna(fill_value)
        if aggfunc is len and not observed and lib.is_integer(fill_value):
            # TODO: can we avoid this?  this used to be handled by
            #  downcast="infer" in fillna
            table = table.astype(np.int64)

    if margins:
        if dropna:
            data = data[data.notna().all(axis=1)]
        table = _add_margins(
            table,
            data,
            values,
            rows=index,
            cols=columns,
            aggfunc=aggfunc,
            kwargs=kwargs,
            observed=dropna,
            margins_name=margins_name,
            fill_value=fill_value,
        )

    # discard the top level
    if values_passed and not values_multi and isinstance(table.columns, MultiIndex) and table.columns.nlevels > 1:
        table.columns = cast(MultiIndex, table.columns).droplevel(0)
    if len(index) == 0 and len(columns) > 0:
        table = table.T

    # GH 15193 Make sure empty columns are removed if dropna=True
    if isinstance(table, ABCDataFrame) and dropna:
        table = table.dropna(how="all", axis=1)

    return table


def _add_margins(
    table: DataFrame | Series,
    data: DataFrame,
    values: Optional[Union[str, List[str], Series, Index]],
    rows: List[Union[str, Series, Index, Grouper]],
    cols: List[Union[str, Series, Index, Grouper]],
    aggfunc: Union[AggFuncTypeBase, AggFuncTypeDict],
    kwargs: Dict[str, Any],
    observed: bool,
    margins_name: Hashable = "All",
    fill_value: Optional[Union[int, float, str]] = None,
) -> DataFrame | Series:
    if not isinstance(margins_name, str):
        raise ValueError("margins_name argument must be a string")

    msg: str = f'Conflicting name "{margins_name}" in margins'
    for level in table.index.names:
        if margins_name in table.index.get_level_values(level):
            raise ValueError(msg)

    grand_margin: Dict[Hashable, Any] = _compute_grand_margin(data, values, aggfunc, kwargs, margins_name)

    if table.ndim == 2:
        # i.e. DataFrame
        for level in table.columns.names[1:]:
            if margins_name in table.columns.get_level_values(level):
                raise ValueError(msg)

    key: Union[str, Tuple[str, ...]]
    if len(rows) > 1:
        key = (margins_name,) + ("",) * (len(rows) - 1)
    else:
        key = margins_name

    from pandas import DataFrame

    if not values and isinstance(table, ABCSeries):
        # If there are no values and the table is a series, then there is only
        # one column in the data. Compute grand margin and return it.
        return table._append(table._constructor({key: grand_margin[margins_name]}))

    elif values:
        marginal_result_set: Union[Tuple[DataFrame, List[Union[str, Tuple[str, ...]]], Series], DataFrame | Series] = _generate_marginal_results(
            table,
            data,
            values,
            rows,
            cols,
            aggfunc,
            kwargs,
            observed,
            margins_name,
        )
        if not isinstance(marginal_result_set, tuple):
            return cast(DataFrame | Series, marginal_result_set)
        result, margin_keys, row_margin = marginal_result_set
    else:
        # no values, and table is a DataFrame
        assert isinstance(table, ABCDataFrame)
        marginal_result_set = _generate_marginal_results_without_values(
            table, data, rows, cols, aggfunc, kwargs, observed, margins_name
        )
        if not isinstance(marginal_result_set, tuple):
            return cast(DataFrame | Series, marginal_result_set)
        result, margin_keys, row_margin = cast(Tuple[DataFrame, List[Union[str, Tuple[str, ...]]], Series], marginal_result_set)

    row_margin = row_margin.reindex(result.columns, fill_value=fill_value)
    # populate grand margin
    for k in margin_keys:
        if isinstance(k, str):
            row_margin[k] = grand_margin[k]
        else:
            row_margin[k] = grand_margin[k[0]]

    margin_dummy: DataFrame = DataFrame(row_margin, columns=Index([key])).T

    row_names: Optional[List[str]] = result.index.names
    # check the result column and leave floats

    dtypes_set = set(result.dtypes)
    for dtype in dtypes_set:
        if isinstance(dtype, ExtensionDtype):
            # Can hold NA already
            continue

        cols_to_cast = result.select_dtypes([dtype]).columns
        margin_dummy[cols_to_cast] = margin_dummy[cols_to_cast].apply(
            maybe_downcast_to_dtype, args=(dtype,)
        )
    result = result._append(margin_dummy)
    result.index.names = row_names

    return result


def _compute_grand_margin(
    data: DataFrame, values: Optional[Union[str, List[str], Series, Index]], aggfunc: Union[AggFuncTypeBase, AggFuncTypeDict], kwargs: Dict[str, Any], margins_name: Hashable = "All"
) -> Dict[Hashable, Any]:
    if values:
        grand_margin: Dict[Hashable, Any] = {}
        for k, v in data[cast(List[str], values)].items():
            try:
                if isinstance(aggfunc, str):
                    grand_margin[k] = getattr(v, aggfunc)(**kwargs)
                elif isinstance(aggfunc, dict):
                    if isinstance(aggfunc.get(k), str):
                        grand_margin[k] = getattr(v, aggfunc[k])(**kwargs)
                    else:
                        func: Callable = cast(Callable, aggfunc[k])
                        grand_margin[k] = func(v, **kwargs)
                else:
                    func: Callable = cast(Callable, aggfunc)
                    grand_margin[k] = func(v, **kwargs)
            except TypeError:
                pass
        return grand_margin
    else:
        return {margins_name: cast(Callable, aggfunc)(data.index, **kwargs)}


def _generate_marginal_results(
    table: DataFrame | Series,
    data: DataFrame,
    values: Optional[Union[str, List[str], Series, Index]],
    rows: List[Union[str, Series, Index, Grouper]],
    cols: List[Union[str, Series, Index, Grouper]],
    aggfunc: Union[AggFuncTypeBase, AggFuncTypeDict],
    kwargs: Dict[str, Any],
    observed: bool,
    margins_name: Hashable = "All",
) -> Union[Tuple[DataFrame, List[Union[str, Tuple[str, ...]]], Series], DataFrame | Series]:
    margin_keys: List[Union[str, Tuple[str, ...]]] = []
    if len(cols) > 0:
        # need to "interleave" the margins
        table_pieces: List[DataFrame] = []
        margin_keys = []

        def _all_key(key: Union[str, Tuple[str, ...]]) -> Union[str, Tuple[str, ...]]:
            return (key, margins_name) + ("",) * (len(cols) - 1)

        if len(rows) > 0:
            margin = (
                data[rows + cast(List[str], values)]
                .groupby(rows, observed=observed)
                .agg(aggfunc, **kwargs)
            )
            cat_axis = 1

            for key, piece in table.T.groupby(level=0, observed=observed):
                piece = cast(DataFrame, piece).T
                all_key = _all_key(key)

                piece[all_key] = margin[key]

                table_pieces.append(piece)
                margin_keys.append(all_key)
        else:
            margin = (
                data[cols[:1] + cast(List[str], values)]
                .groupby(cols[:1], observed=observed)
                .agg(aggfunc, **kwargs)
                .T
            )

            cat_axis = 0
            for key, piece in table.groupby(level=0, observed=observed):
                if len(cols) > 1:
                    all_key = _all_key(key)
                else:
                    all_key = margins_name
                table_pieces.append(piece)
                transformed_piece = cast(DataFrame, margin[key]).to_frame().T
                if isinstance(piece.index, MultiIndex):
                    # We are adding an empty level
                    transformed_piece.index = MultiIndex.from_tuples(
                        [all_key],
                        names=piece.index.names
                        + [
                            None,
                        ],
                    )
                else:
                    transformed_piece.index = Index([all_key], name=piece.index.name)

                # append piece for margin into table_piece
                table_pieces.append(transformed_piece)
                margin_keys.append(all_key)

        if not table_pieces:
            # GH 49240
            return table
        else:
            result = cast(DataFrame, concat(table_pieces, axis=cat_axis))

        if len(rows) == 0:
            return result
    else:
        result = table
        margin_keys = cast(List[Union[str, Tuple[str, ...]]], table.columns)

    if len(cols) > 0:
        row_margin = (
            data[cols + cast(List[str], values)].groupby(cols, observed=observed).agg(aggfunc, **kwargs)
        )
        row_margin = cast(Series, row_margin.stack())

        # GH#26568. Use names instead of indices in case of numeric names
        new_order_indices = itertools.chain([len(cols)], range(len(cols)))
        new_order_names = [row_margin.index.names[i] for i in new_order_indices]
        row_margin.index = cast(MultiIndex, row_margin.index).reorder_levels(new_order_names)
    else:
        row_margin = cast(Series, data._constructor_sliced(np.nan, index=cast(Index, result.columns)))

    return result, margin_keys, row_margin


def _generate_marginal_results_without_values(
    table: DataFrame,
    data: DataFrame,
    rows: List[Union[str, Series, Index, Grouper]],
    cols: List[Union[str, Series, Index, Grouper]],
    aggfunc: Union[AggFuncTypeBase, AggFuncTypeDict],
    kwargs: Dict[str, Any],
    observed: bool,
    margins_name: Hashable = "All",
) -> Union[Tuple[DataFrame, List[Union[str, Tuple[str, ...]]], Series], DataFrame | Series]:
    margin_keys: List[Union[str, Tuple[str, ...]]] = []
    if len(cols) > 0:
        # need to "interleave" the margins
        if len(rows) > 0:
            margin = data.groupby(rows, observed=observed)[rows].apply(
                cast(Callable, aggfunc), **kwargs
            )
            all_key: Union[str, Tuple[str, ...]] = margins_name if len(cols) == 1 else (margins_name,) + ("",) * (len(cols) - 1)
            table[all_key] = margin
            result = table
            margin_keys.append(all_key)

        else:
            margin = data.groupby(level=0, observed=observed).apply(cast(Callable, aggfunc), **kwargs)
            all_key: Union[str, Tuple[str, ...]] = margins_name if len(cols) == 1 else (margins_name,) + ("",) * (len(cols) - 1)
            table[all_key] = margin
            result = table
            margin_keys.append(all_key)
            return result
    else:
        result = table
        margin_keys = cast(List[Union[str, Tuple[str, ...]]], table.columns)

    if len(cols):
        row_margin = data.groupby(cols, observed=observed)[cols].apply(cast(Callable, aggfunc), **kwargs)
    else:
        row_margin = cast(Series, Series(np.nan, index=cast(Index, result.columns)))

    return result, margin_keys, row_margin


def _convert_by(by: Optional[Union[str, List[str], Series, Index, Grouper]]) -> List[Union[str, Series, Index, Grouper]]:
    if by is None:
        return []
    elif (
        is_scalar(by)
        or isinstance(by, (np.ndarray, Index, ABCSeries, Grouper))
        or callable(by)
    ):
        return [by]
    else:
        return list(by)


def pivot(
    data: DataFrame,
    *,
    columns: IndexLabel,
    index: Union[IndexLabel, lib.NoDefault] = lib.no_default,
    values: Union[IndexLabel, lib.NoDefault] = lib.no_default,
) -> DataFrame:
    """
    Return reshaped DataFrame organized by given index / column values.

    [Docstring omitted for brevity]
    """
    columns_listlike: List[str] = com.convert_to_list_like(columns)

    # If columns is None we will create a MultiIndex level with None as name
    # which might cause duplicated names because None is the default for
    # level names
    if any(name is None for name in data.index.names):
        data = data.copy(deep=False)
        data.index.names = [
            name if name is not None else lib.no_default for name in data.index.names
        ]

    indexed: DataFrame | Series
    if values is lib.no_default:
        if index is not lib.no_default:
            cols: List[str] = com.convert_to_list_like(index)
        else:
            cols = []

        append: bool = index is lib.no_default
        # error: Unsupported operand types for + ("List[Any]" and "ExtensionArray")
        # error: Unsupported left operand type for + ("ExtensionArray")
        indexed = data.set_index(
            cast(List[Union[str, Series, Index, Grouper]], cols + columns_listlike),  # type: ignore[operator]
            append=append,
        )
    else:
        index_list: List[Union[Index, Series]] = []
        if index is lib.no_default:
            if isinstance(data.index, MultiIndex):
                # GH 23955
                index_list = [
                    data.index.get_level_values(i) for i in range(data.index.nlevels)
                ]
            else:
                index_list = [
                    cast(Series, data._constructor_sliced(data.index, name=data.index.name))
                ]
        else:
            index_list = [cast(Series, data[idx]) for idx in com.convert_to_list_like(index)]

        data_columns: List[Union[Series, Index]] = [data[col] for col in columns_listlike]
        index_list.extend(data_columns)
        multiindex: MultiIndex = MultiIndex.from_arrays(index_list)

        if is_list_like(values) and not isinstance(values, tuple):
            # Exclude tuple because it is seen as a single column name
            indexed = data._constructor(
                data[cast(List[str], values)]._values,
                index=multiindex,
                columns=cast(SequenceNotStr, values),
            )
        else:
            indexed = data._constructor_sliced(data[cast(IndexLabel, values)]._values, index=multiindex)
    # error: Argument 1 to "unstack" of "DataFrame" has incompatible type "Union[List[Any], ExtensionArray, ndarray[Any, Any], Index, Series]"; expected "Hashable"
    # unstack with a MultiIndex returns a DataFrame
    result: DataFrame = cast(DataFrame, indexed.unstack(columns_listlike))  # type: ignore[arg-type]
    result.index.names = [
        name if name is not lib.no_default else None for name in result.index.names
    ]

    return result


def crosstab(
    index: Union[
        Hashable,
        Sequence[Hashable],
        Series,
        Sequence[Series],
        Index,
        Sequence[Index],
        ABCDataFrame,
    ],
    columns: Union[
        Hashable,
        Sequence[Hashable],
        Series,
        Sequence[Series],
        Index,
        Sequence[Index],
        ABCDataFrame,
    ],
    values: Optional[Union[str, List[str], Series, Index]] = None,
    rownames: Optional[Sequence[str]] = None,
    colnames: Optional[Sequence[str]] = None,
    aggfunc: Optional[AggFuncType] = None,
    margins: bool = False,
    margins_name: Hashable = "All",
    dropna: bool = True,
    normalize: Union[
        bool, Literal[0, 1, "all", "index", "columns"]
    ] = False,
) -> DataFrame:
    """
    Compute a simple cross tabulation of two (or more) factors.

    [Docstring omitted for brevity]
    """
    if values is None and aggfunc is not None:
        raise ValueError("aggfunc cannot be used without values.")

    if values is not None and aggfunc is None:
        raise ValueError("values cannot be used without an aggfunc.")

    if not is_nested_list_like(index):
        index = [index]
    if not is_nested_list_like(columns):
        columns = [columns]

    common_idx: Optional[Index] = None
    pass_objs: List[Union[Series, DataFrame]] = [x for x in index + columns if isinstance(x, (ABCSeries, ABCDataFrame))]
    if pass_objs:
        common_idx = get_objs_combined_axis(pass_objs, intersect=True, sort=False)

    rownames_final: List[str] = _get_names(index, rownames, prefix="row")
    colnames_final: List[str] = _get_names(columns, colnames, prefix="col")

    # duplicate names mapped to unique names for pivot op
    (
        rownames_mapper,
        unique_rownames,
        colnames_mapper,
        unique_colnames,
    ) = _build_names_mapper(rownames_final, colnames_final)

    from pandas import DataFrame

    data_dict: Dict[str, Union[Series, Index]] = {
        **dict(zip(unique_rownames, index)),
        **dict(zip(unique_colnames, columns)),
    }
    df: DataFrame = DataFrame(data_dict, index=common_idx)

    if values is None:
        df["__dummy__"] = 0
        kwargs: Dict[str, Any] = {"aggfunc": len, "fill_value": 0}
    else:
        df["__dummy__"] = values
        kwargs = {"aggfunc": aggfunc}

    # error: Argument 7 to "pivot_table" of "DataFrame" has incompatible type
    # "**Dict[str, object]"; expected "Union[...]"
    table: DataFrame = cast(DataFrame, df.pivot_table(
        "__dummy__",
        index=unique_rownames,
        columns=unique_colnames,
        margins=margins,
        margins_name=margins_name,
        dropna=dropna,
        observed=dropna,
        **kwargs,  # type: ignore[arg-type]
    ))

    # Post-process
    if normalize is not False:
        table = _normalize(
            table, normalize=normalize, margins=margins, margins_name=margins_name
        )

    table = table.rename_axis(index=rownames_mapper, axis=0)
    table = table.rename_axis(columns=colnames_mapper, axis=1)

    return table


def _normalize(
    table: DataFrame,
    normalize: Union[bool, Literal[0, 1, "all", "index", "columns"]],
    margins: bool,
    margins_name: Hashable = "All",
) -> DataFrame:
    if not isinstance(normalize, (bool, str)):
        axis_subs: Dict[int, str] = {0: "index", 1: "columns"}
        try:
            normalize = axis_subs[normalize]
        except KeyError as err:
            raise ValueError("Not a valid normalize argument") from err

    if not margins:
        # Actual Normalizations
        normalizers: Dict[Union[Literal[True], str], Callable[[DataFrame], DataFrame]] = {
            "all": lambda x: x / x.sum(axis=1).sum(axis=0),
            "columns": lambda x: x / x.sum(),
            "index": lambda x: x.div(x.sum(axis=1), axis=0),
        }

        normalizers[True] = normalizers["all"]

        try:
            f: Callable[[DataFrame], DataFrame] = normalizers[normalize]
        except KeyError as err:
            raise ValueError("Not a valid normalize argument") from err

        table = f(table)
        table = table.fillna(0)

    elif margins:
        # keep index and column of pivoted table
        table_index: Index = table.index
        table_columns: Index = table.columns
        last_ind_or_col: Hashable = table.iloc[-1, :].name

        # check if margin name is not in (for MI cases) and not equal to last
        # index/column and save the column and index margin
        if (margins_name not in last_ind_or_col) & (margins_name != last_ind_or_col):
            raise ValueError(f"{margins_name} not in pivoted DataFrame")
        column_margin: Series = table.iloc[:-1, -1]
        index_margin: Series = table.iloc[-1, :-1]

        # keep the core table
        table = table.iloc[:-1, :-1]

        # Normalize core
        table = _normalize(table, normalize=normalize, margins=False)

        # Fix Margins
        if normalize == "columns":
            column_margin = column_margin / column_margin.sum()
            table = cast(DataFrame, concat([table, column_margin], axis=1))
            table = table.fillna(0)
            table.columns = table_columns

        elif normalize == "index":
            index_margin = index_margin / index_margin.sum()
            table = cast(DataFrame, table._append(index_margin, ignore_index=True))
            table = table.fillna(0)
            table.index = table_index

        elif normalize == "all" or normalize is True:
            column_margin = column_margin / column_margin.sum()
            index_margin = index_margin / index_margin.sum()
            index_margin.loc[margins_name] = 1
            table = cast(DataFrame, concat([table, column_margin], axis=1))
            table = cast(DataFrame, table._append(index_margin, ignore_index=True))

            table = table.fillna(0)
            table.index = table_index
            table.columns = table_columns

        else:
            raise ValueError("Not a valid normalize argument")
    else:
        raise ValueError("Not a valid margins argument")

    return table


def _get_names(arrs: Sequence[Union[str, Series, Index, Grouper]], names: Optional[Sequence[str]], prefix: str = "row") -> List[str]:
    if names is None:
        names_final: List[str] = []
        for i, arr in enumerate(arrs):
            if isinstance(arr, ABCSeries) and arr.name is not None:
                names_final.append(arr.name)
            else:
                names_final.append(f"{prefix}_{i}")
    else:
        if len(names) != len(arrs):
            raise AssertionError("arrays and names must have the same length")
        if not isinstance(names, list):
            names_final = list(names)
        else:
            names_final = names

    return names_final


def _build_names_mapper(
    rownames: List[str], colnames: List[str]
) -> Tuple[Dict[str, str], List[str], Dict[str, str], List[str]]:
    """
    Given the names of a DataFrame's rows and columns, returns a set of unique row
    and column names and mappers that convert to original names.

    [Docstring omitted for brevity]
    """
    dup_names: set = set(rownames) | set(colnames)

    rownames_mapper: Dict[str, str] = {
        f"row_{i}": name for i, name in enumerate(rownames) if name in dup_names
    }
    unique_rownames: List[str] = [
        f"row_{i}" if name in dup_names else name for i, name in enumerate(rownames)
    ]

    colnames_mapper: Dict[str, str] = {
        f"col_{i}": name for i, name in enumerate(colnames) if name in dup_names
    }
    unique_colnames: List[str] = [
        f"col_{i}" if name in dup_names else name for i, name in enumerate(colnames)
    ]

    return rownames_mapper, unique_rownames, colnames_mapper, unique_colnames
