from __future__ import annotations
import re
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union
from collections.abc import Hashable
import numpy as np
from pandas.core.dtypes.common import is_iterator, is_list_like
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.missing import notna
import pandas.core.algorithms as algos
from pandas.core.indexes.api import MultiIndex
from pandas.core.reshape.concat import concat
from pandas.core.tools.numeric import to_numeric

if TYPE_CHECKING:
    from pandas import DataFrame
    from pandas._typing import AnyArrayLike


def ensure_list_vars(
    arg_vars: Optional[Union[Hashable, Sequence[Hashable], np.ndarray]],
    variable: str,
    columns: object,
) -> list[Hashable]:
    if arg_vars is not None:
        if not is_list_like(arg_vars):
            return [arg_vars]  # type: ignore[list-item]
        elif isinstance(columns, MultiIndex) and (not isinstance(arg_vars, list)):
            raise ValueError(f'{variable} must be a list of tuples when columns are a MultiIndex')
        else:
            return list(arg_vars)  # type: ignore[list-item]
    else:
        return []


def melt(
    frame: DataFrame,
    id_vars: Optional[Union[Hashable, Sequence[Hashable], np.ndarray]] = None,
    value_vars: Optional[Union[Hashable, Sequence[Hashable], np.ndarray]] = None,
    var_name: Optional[Union[Hashable, Sequence[Hashable]]] = None,
    value_name: Hashable = 'value',
    col_level: Optional[Union[int, Hashable]] = None,
    ignore_index: bool = True,
) -> DataFrame:
    """
    Unpivot a DataFrame from wide to long format, optionally leaving identifiers set.

    This function is useful to reshape a DataFrame into a format where one
    or more columns are identifier variables (`id_vars`), while all other
    columns are considered measured variables (`value_vars`), and are "unpivoted" to
    the row axis, leaving just two non-identifier columns, 'variable' and
    'value'.

    Parameters
    ----------
    frame : DataFrame
        The DataFrame to unpivot.
    id_vars : scalar, tuple, list, or ndarray, optional
        Column(s) to use as identifier variables.
    value_vars : scalar, tuple, list, or ndarray, optional
        Column(s) to unpivot. If not specified, uses all columns that
        are not set as `id_vars`.
    var_name : scalar, tuple, list, or ndarray, optional
        Name to use for the 'variable' column. If None it uses
        ``frame.columns.name`` or 'variable'. Must be a scalar if columns are a
        MultiIndex.
    value_name : scalar, default 'value'
        Name to use for the 'value' column, can't be an existing column label.
    col_level : scalar, optional
        If columns are a MultiIndex then use this level to melt.
    ignore_index : bool, default True
        If True, original index is ignored. If False, the original index is retained.
        Index labels will be repeated as necessary.

    Returns
    -------
    DataFrame
        Unpivoted DataFrame.

    See Also
    --------
    DataFrame.melt : Identical method.
    pivot_table : Create a spreadsheet-style pivot table as a DataFrame.
    DataFrame.pivot : Return reshaped DataFrame organized
        by given index / column values.
    DataFrame.explode : Explode a DataFrame from list-like
            columns to long format.

    Notes
    -----
    Reference :ref:`the user guide <reshaping.melt>` for more examples.

    Examples
    --------
    >>> df = pd.DataFrame(
    ...     {
    ...         "A": {0: "a", 1: "b", 2: "c"},
    ...         "B": {0: 1, 1: 3, 2: 5},
    ...         "C": {0: 2, 1: 4, 2: 6},
    ...     }
    ... )
    >>> df
    A  B  C
    0  a  1  2
    1  b  3  4
    2  c  5  6

    >>> pd.melt(df, id_vars=["A"], value_vars=["B"])
    A variable  value
    0  a        B      1
    1  b        B      3
    2  c        B      5

    >>> pd.melt(df, id_vars=["A"], value_vars=["B", "C"])
    A variable  value
    0  a        B      1
    1  b        B      3
    2  c        B      5
    3  a        C      2
    4  b        C      4
    5  c        C      6

    The names of 'variable' and 'value' columns can be customized:

    >>> pd.melt(
    ...     df,
    ...     id_vars=["A"],
    ...     value_vars=["B"],
    ...     var_name="myVarname",
    ...     value_name="myValname",
    ... )
    A myVarname  myValname
    0  a         B          1
    1  b         B          3
    2  c         B          5

    Original index values can be kept around:

    >>> pd.melt(df, id_vars=["A"], value_vars=["B", "C"], ignore_index=False)
    A variable  value
    0  a        B      1
    1  b        B      3
    2  c        B      5
    0  a        C      2
    1  b        C      4
    2  c        C      6

    If you have multi-index columns:

    >>> df.columns = [list("ABC"), list("DEF")]
    >>> df
    A  B  C
    D  E  F
    0  a  1  2
    1  b  3  4
    2  c  5  6

    >>> pd.melt(df, col_level=0, id_vars=["A"], value_vars=["B"])
    A variable  value
    0  a        B      1
    1  b        B      3
    2  c        B      5

    >>> pd.melt(df, id_vars=[("A", "D")], value_vars=[("B", "E")])
    (A, D) variable_0 variable_1  value
    0      a          B          E      1
    1      b          B          E      3
    2      c          B          E      5
    """
    if value_name in frame.columns:
        raise ValueError(f'value_name ({value_name}) cannot match an element in the DataFrame columns.')

    id_vars_list: list[Hashable] = ensure_list_vars(id_vars, 'id_vars', frame.columns)
    value_vars_was_not_none: bool = value_vars is not None
    value_vars_list: list[Hashable] = ensure_list_vars(value_vars, 'value_vars', frame.columns)

    if id_vars_list or value_vars_list:
        if col_level is not None:
            level = frame.columns.get_level_values(col_level)
        else:
            level = frame.columns

        labels: list[Hashable] = id_vars_list + value_vars_list
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
                var_name_list: list[Hashable] = frame.columns.names  # type: ignore[assignment]
            else:
                var_name_list = [f'variable_{i}' for i in range(len(frame.columns.names))]
        else:
            var_name_list = [frame.columns.name if frame.columns.name is not None else 'variable']
    elif is_list_like(var_name):
        if isinstance(frame.columns, MultiIndex):
            if is_iterator(var_name):
                var_name = list(var_name)  # type: ignore[assignment]
            if len(var_name) > len(frame.columns):  # type: ignore[arg-type]
                raise ValueError(f'var_name={var_name!r} has {len(var_name)} items, but the dataframe columns only have {len(frame.columns)} levels.')
            var_name_list = list(var_name)  # type: ignore[list-item]
        else:
            raise ValueError(f'var_name={var_name!r} must be a scalar.')
    else:
        var_name_list = [var_name]

    num_rows, K = frame.shape
    num_cols_adjusted: int = K - len(id_vars_list)

    mdata: dict[Hashable, Any] = {}
    for col in id_vars_list:
        id_data = frame.pop(col)
        if not isinstance(id_data.dtype, np.dtype):
            if num_cols_adjusted > 0:
                mdata[col] = concat([id_data] * num_cols_adjusted, ignore_index=True)
            else:
                mdata[col] = type(id_data)([], name=id_data.name, dtype=id_data.dtype)
        else:
            mdata[col] = np.tile(id_data._values, num_cols_adjusted)

    mcolumns: list[Hashable] = id_vars_list + var_name_list + [value_name]

    if frame.shape[1] > 0 and (not any((not isinstance(dt, np.dtype) and dt._supports_2d for dt in frame.dtypes))):
        mdata[value_name] = concat([frame.iloc[:, i] for i in range(frame.shape[1])], ignore_index=True).values
    else:
        mdata[value_name] = frame._values.ravel('F')

    for i, col in enumerate(var_name_list):
        mdata[col] = frame.columns._get_level_values(i).repeat(num_rows)

    result: DataFrame = frame._constructor(mdata, columns=mcolumns)  # type: ignore[assignment]

    if not ignore_index:
        taker = np.tile(np.arange(len(frame)), num_cols_adjusted)
        result.index = frame.index.take(taker)

    return result


def lreshape(
    data: DataFrame,
    groups: dict[Hashable, Sequence[Hashable]],
    dropna: bool = True,
) -> DataFrame:
    """
    Reshape wide-format data to long. Generalized inverse of DataFrame.pivot.

    Accepts a dictionary, ``groups``, in which each key is a new column name
    and each value is a list of old column names that will be "melted" under
    the new column name as part of the reshape.

    Parameters
    ----------
    data : DataFrame
        The wide-format DataFrame.
    groups : dict
        {new_name : list_of_columns}.
    dropna : bool, default True
        Do not include columns whose entries are all NaN.

    Returns
    -------
    DataFrame
        Reshaped DataFrame.

    See Also
    --------
    melt : Unpivot a DataFrame from wide to long format, optionally leaving
        identifiers set.
    pivot : Create a spreadsheet-style pivot table as a DataFrame.
    DataFrame.pivot : Pivot without aggregation that can handle
        non-numeric data.
    DataFrame.pivot_table : Generalization of pivot that can handle
        duplicate values for one index/column pair.
    DataFrame.unstack : Pivot based on the index values instead of a
        column.
    wide_to_long : Wide panel to long format. Less flexible but more
        user-friendly than melt.
    """
    mdata: dict[Hashable, Any] = {}
    pivot_cols: list[Hashable] = []
    all_cols: set[Hashable] = set()
    K = len(next(iter(groups.values())))
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

    return data._constructor(mdata, columns=id_cols + pivot_cols)  # type: ignore[return-value]


def wide_to_long(
    df: DataFrame,
    stubnames: Union[str, Sequence[str]],
    i: Union[Hashable, Sequence[Hashable]],
    j: str,
    sep: str = '',
    suffix: str = '\\d+',
) -> DataFrame:
    """
    Unpivot a DataFrame from wide to long format.

    Less flexible but more user-friendly than melt.

    With stubnames ['A', 'B'], this function expects to find one or more
    group of columns with format
    A-suffix1, A-suffix2,..., B-suffix1, B-suffix2,...
    You specify what you want to call this suffix in the resulting long format
    with `j` (for example `j='year'`)

    Each row of these wide variables are assumed to be uniquely identified by
    `i` (can be a single column name or a list of column names)

    All remaining variables in the data frame are left intact.

    Parameters
    ----------
    df : DataFrame
        The wide-format DataFrame.
    stubnames : str or list-like
        The stub name(s). The wide format variables are assumed to
        start with the stub names.
    i : str or list-like
        Column(s) to use as id variable(s).
    j : str
        The name of the sub-observation variable. What you wish to name your
        suffix in the long format.
    sep : str, default ""
        A character indicating the separation of the variable names
        in the wide format, to be stripped from the names in the long format.
        For example, if your column names are A-suffix1, A-suffix2, you
        can strip the hyphen by specifying `sep='-'`.
    suffix : str, default '\\\\d+'
        A regular expression capturing the wanted suffixes. '\\\\d+' captures
        numeric suffixes. Suffixes with no numbers could be specified with the
        negated character class '\\\\D+'. You can also further disambiguate
        suffixes, for example, if your wide variables are of the form A-one,
        B-two,.., and you have an unrelated column A-rating, you can ignore the
        last one by specifying `suffix='(!?one|two)'`. When all suffixes are
        numeric, they are cast to int64/float64.

    Returns
    -------
    DataFrame
        A DataFrame that contains each stub name as a variable, with new index
        (i, j).

    See Also
    --------
    melt : Unpivot a DataFrame from wide to long format, optionally leaving
        identifiers set.
    pivot : Create a spreadsheet-style pivot table as a DataFrame.
    DataFrame.pivot : Pivot without aggregation that can handle
        non-numeric data.
    DataFrame.pivot_table : Generalization of pivot that can handle
        duplicate values for one index/column pair.
    DataFrame.unstack : Pivot based on the index values instead of a
        column.

    Notes
    -----
    All extra variables are left untouched. This simply uses
    `pandas.melt` under the hood, but is hard-coded to "do the right thing"
    in a typical case.
    """

    def get_var_names(df: DataFrame, stub: str, sep: str, suffix: str) -> Any:
        regex = f'^{re.escape(stub)}{re.escape(sep)}{suffix}$'
        return df.columns[df.columns.str.match(regex)]

    def melt_stub(
        df: DataFrame,
        stub: str,
        i: list[Hashable],
        j: str,
        value_vars: Any,
        sep: str,
    ) -> DataFrame:
        newdf = melt(df, id_vars=i, value_vars=value_vars, value_name=stub.rstrip(sep), var_name=j)
        newdf[j] = newdf[j].str.replace(re.escape(stub + sep), '', regex=True)
        try:
            newdf[j] = to_numeric(newdf[j])
        except (TypeError, ValueError, OverflowError):
            pass
        return newdf.set_index(i + [j])

    if not is_list_like(stubnames):
        stubnames_list: list[str] = [stubnames]  # type: ignore[list-item]
    else:
        stubnames_list = list(stubnames)  # type: ignore[arg-type]

    if df.columns.isin(stubnames_list).any():
        raise ValueError("stubname can't be identical to a column name")

    if not is_list_like(i):
        i_list: list[Hashable] = [i]  # type: ignore[list-item]
    else:
        i_list = list(i)  # type: ignore[arg-type]

    if df[i_list].duplicated().any():
        raise ValueError('the id variables need to uniquely identify each row')

    _melted: list[DataFrame] = []
    value_vars_flattened: list[Hashable] = []
    for stub in stubnames_list:
        value_var = get_var_names(df, stub, sep, suffix)
        value_vars_flattened.extend(value_var)
        _melted.append(melt_stub(df, stub, i_list, j, value_var, sep))

    melted = concat(_melted, axis=1)
    id_vars = df.columns.difference(value_vars_flattened)
    new = df[id_vars]
    if len(i_list) == 1:
        return new.set_index(i_list).join(melted)
    else:
        return new.merge(melted.reset_index(), on=i_list).set_index(i_list + [j])