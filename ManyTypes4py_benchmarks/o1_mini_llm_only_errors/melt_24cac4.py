from __future__ import annotations
import re
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Union,
)
import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_iterator, is_list_like
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.missing import notna
import pandas.core.algorithms as algos
from pandas.core.indexes.api import MultiIndex
from pandas.core.reshape.concat import concat
from pandas.core.tools.numeric import to_numeric

if TYPE_CHECKING:
    from collections.abc import Hashable
    from pandas._typing import AnyArrayLike
    from pandas import DataFrame, Index


def ensure_list_vars(
    arg_vars: Optional[Union[Hashable, Sequence[Hashable]]],
    variable: str,
    columns: Union[Index, MultiIndex],
) -> List[Hashable]:
    if arg_vars is not None:
        if not is_list_like(arg_vars):
            return [arg_vars]
        elif isinstance(columns, MultiIndex) and (not isinstance(arg_vars, list)):
            raise ValueError(f'{variable} must be a list of tuples when columns are a MultiIndex')
        else:
            return list(arg_vars)
    else:
        return []


def melt(
    frame: pd.DataFrame,
    id_vars: Optional[Union[Hashable, Sequence[Hashable]]] = None,
    value_vars: Optional[Union[Hashable, Sequence[Hashable]]] = None,
    var_name: Optional[Union[str, Sequence[str]]] = None,
    value_name: str = 'value',
    col_level: Optional[int] = None,
    ignore_index: bool = True,
) -> pd.DataFrame:
    """
    Unpivot a DataFrame from wide to long format, optionally leaving identifiers set.

    [Docstring omitted for brevity]
    """
    if value_name in frame.columns:
        raise ValueError(f'value_name ({value_name}) cannot match an element in the DataFrame columns.')
    id_vars_list: List[Hashable] = ensure_list_vars(id_vars, 'id_vars', frame.columns)
    value_vars_was_not_none: bool = value_vars is not None
    value_vars_list: List[Hashable] = ensure_list_vars(value_vars, 'value_vars', frame.columns)
    if id_vars_list or value_vars_list:
        if col_level is not None:
            level = frame.columns.get_level_values(col_level)
        else:
            level = frame.columns
        labels = id_vars_list + value_vars_list
        idx = level.get_indexer_for(labels)
        missing = idx == -1
        if missing.any():
            missing_labels = [lab for lab, not_found in zip(labels, missing) if not_found]
            raise KeyError(f'The following id_vars or value_vars are not present in the DataFrame: {missing_labels}')
        if value_vars_was_not_none:
            frame = frame.iloc[:, algos.unique(idx)]
        else:
            frame = frame.copy(deep=False)
    else:
        frame = frame.copy(deep=False)
    if col_level is not None:
        frame.columns = frame.columns.get_level_values(col_level)
    if var_name is None:
        if isinstance(frame.columns, MultiIndex):
            if len(frame.columns.names) == len(set(frame.columns.names)):
                var_name = frame.columns.names
            else:
                var_name = [f'variable_{i}' for i in range(len(frame.columns.names))]
        else:
            var_name = [frame.columns.name if frame.columns.name is not None else 'variable']
    elif is_list_like(var_name):
        if isinstance(frame.columns, MultiIndex):
            if is_iterator(var_name):
                var_name = list(var_name)
            if len(var_name) > len(frame.columns):
                raise ValueError(f'var_name={var_name!r} has {len(var_name)} items, but the dataframe columns only have {len(frame.columns)} levels.')
        else:
            raise ValueError(f'var_name={var_name!r} must be a scalar.')
    else:
        var_name = [var_name]
    num_rows, K = frame.shape
    num_cols_adjusted = K - len(id_vars_list)
    mdata: Dict[Any, AnyArrayLike] = {}
    for col in id_vars_list:
        id_data = frame.pop(col)
        if not isinstance(id_data.dtype, np.dtype):
            if num_cols_adjusted > 0:
                mdata[col] = concat([id_data] * num_cols_adjusted, ignore_index=True)
            else:
                mdata[col] = type(id_data)([], name=id_data.name, dtype=id_data.dtype)
        else:
            mdata[col] = np.tile(id_data._values, num_cols_adjusted)
    mcolumns: List[Union[str, List[str]]] = id_vars_list + var_name + [value_name]
    if frame.shape[1] > 0 and (not any((not isinstance(dt, np.dtype) and dt._supports_2d for dt in frame.dtypes))):
        mdata[value_name] = concat([frame.iloc[:, i] for i in range(frame.shape[1])], ignore_index=True).values
    else:
        mdata[value_name] = frame._values.ravel('F')
    for i, col in enumerate(var_name):
        mdata[col] = frame.columns._get_level_values(i).repeat(num_rows)
    result = frame._constructor(mdata, columns=mcolumns)
    if not ignore_index:
        taker = np.tile(np.arange(len(frame)), num_cols_adjusted)
        result.index = frame.index.take(taker)
    return result


def lreshape(
    data: pd.DataFrame,
    groups: Dict[str, List[str]],
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Reshape wide-format data to long. Generalized inverse of DataFrame.pivot.

    [Docstring omitted for brevity]
    """
    mdata: Dict[str, AnyArrayLike] = {}
    pivot_cols: List[str] = []
    all_cols: set = set()
    K: int = len(next(iter(groups.values())))
    for target, names in groups.items():
        if len(names) != K:
            raise ValueError('All column lists must be same length')
        to_concat = [data[col]._values for col in names]
        mdata[target] = concat_compat(to_concat)
        pivot_cols.append(target)
        all_cols = all_cols.union(names)
    id_cols = list(data.columns.difference(all_cols))
    for col in id_cols:
        mdata[col] = np.tile(data[col]._values, K)
    if dropna:
        mask = np.ones(len(mdata[pivot_cols[0]]), dtype=bool)
        for c in pivot_cols:
            mask &= notna(mdata[c])
        if not mask.all():
            mdata = {k: v[mask] for k, v in mdata.items()}
    return data._constructor(mdata, columns=id_cols + pivot_cols)


def wide_to_long(
    df: pd.DataFrame,
    stubnames: Union[str, Sequence[str]],
    i: Union[str, Sequence[str]],
    j: str,
    sep: str = '',
    suffix: str = '\\d+',
) -> pd.DataFrame:
    """
    Unpivot a DataFrame from wide to long format.

    [Docstring omitted for brevity]
    """

    def get_var_names(df: pd.DataFrame, stub: str, sep: str, suffix: str) -> Index:
        regex = f'^{re.escape(stub)}{re.escape(sep)}{suffix}$'
        return df.columns[df.columns.str.match(regex)]

    def melt_stub(
        df: pd.DataFrame,
        stub: str,
        i: List[str],
        j: str,
        value_vars: Index,
        sep: str,
    ) -> pd.DataFrame:
        newdf = melt(
            df,
            id_vars=i,
            value_vars=value_vars,
            value_name=stub.rstrip(sep),
            var_name=j,
            ignore_index=True,
        )
        newdf[j] = newdf[j].str.replace(re.escape(stub + sep), '', regex=True)  # type: ignore
        try:
            newdf[j] = to_numeric(newdf[j])
        except (TypeError, ValueError, OverflowError):
            pass
        return newdf.set_index(i + [j])

    if not is_list_like(stubnames):
        stubnames = [stubnames]
    else:
        stubnames = list(stubnames)
    if df.columns.isin(stubnames).any():
        raise ValueError("stubname can't be identical to a column name")
    if not is_list_like(i):
        i = [i]
    else:
        i = list(i)
    if df[i].duplicated().any():
        raise ValueError('the id variables need to uniquely identify each row')
    _melted: List[pd.DataFrame] = []
    value_vars_flattened: List[Any] = []
    for stub in stubnames:
        value_var = get_var_names(df, stub, sep, suffix)
        value_vars_flattened.extend(value_var)
        _melted.append(melt_stub(df, stub, i, j, value_var, sep))
    melted = concat(_melted, axis=1)
    id_vars = df.columns.difference(value_vars_flattened)
    new = df[id_vars]
    if len(i) == 1:
        return new.set_index(i).join(melted)
    else:
        return new.merge(melted.reset_index(), on=i).set_index(i + [j])
