from __future__ import annotations
import codecs
from functools import wraps
import re
from typing import Any, Callable, List, Optional, Sequence, Union, cast
import warnings
import numpy as np
from pandas._config import get_option
from pandas._libs import lib
from pandas._typing import AlignJoin, DtypeObj, F, Scalar, npt
from pandas.util._decorators import Appender
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (ensure_object, is_bool_dtype, is_extension_array_dtype,
                                       is_integer, is_list_like, is_numeric_dtype, is_object_dtype, is_re)
from pandas.core.dtypes.dtypes import ArrowDtype, CategoricalDtype
from pandas.core.dtypes.generic import ABCDataFrame, ABCIndex, ABCMultiIndex, ABCSeries
from pandas.core.dtypes.missing import isna
from pandas.core.arrays import ExtensionArray
from pandas.core.base import NoNewAttributesMixin
from pandas.core.construction import extract_array
if False:  # TYPE_CHECKING
    from collections.abc import Callable, Hashable, Iterator
    from pandas._typing import NpDtype
    from pandas import DataFrame, Index, Series

def cat_safe(list_of_columns: List[np.ndarray], sep: str) -> np.ndarray:
    """
    Auxiliary function for :meth:`str.cat`.

    Same signature as cat_core, but handles TypeErrors in concatenation, which
    happen if the arrays in list_of_columns have the wrong dtypes or content.

    Parameters
    ----------
    list_of_columns : list of numpy arrays
        List of arrays to be concatenated with sep;
        these arrays may not contain NaNs!
    sep : str
        The separator string for concatenating the columns.

    Returns
    -------
    np.ndarray
        The concatenation of list_of_columns with sep.
    """
    try:
        result = cat_core(list_of_columns, sep)
    except TypeError:
        for column in list_of_columns:
            dtype = lib.infer_dtype(column, skipna=True)
            if dtype not in ['string', 'empty']:
                raise TypeError(
                    f"Concatenation requires list-likes containing only strings (or missing values). Offending values found in column {dtype}"
                ) from None
    return result

def cat_core(list_of_columns: List[np.ndarray], sep: str) -> np.ndarray:
    """
    Auxiliary function for :meth:`str.cat`

    Parameters
    ----------
    list_of_columns : list of numpy arrays
        List of arrays to be concatenated with sep;
        these arrays may not contain NaNs!
    sep : str
        The separator string for concatenating the columns.

    Returns
    -------
    np.ndarray
        The concatenation of list_of_columns with sep.
    """
    if sep == '':
        arr_of_cols = np.asarray(list_of_columns, dtype=object)
        return np.sum(arr_of_cols, axis=0)
    list_with_sep: List[Any] = [sep] * (2 * len(list_of_columns) - 1)
    list_with_sep[::2] = list_of_columns
    arr_with_sep = np.asarray(list_with_sep, dtype=object)
    return np.sum(arr_with_sep, axis=0)

def _result_dtype(arr: Any) -> Any:
    from pandas.core.arrays.string_ import StringDtype
    if isinstance(arr.dtype, (ArrowDtype, StringDtype)):
        return arr.dtype
    return object

def _get_single_group_name(regex: re.Pattern) -> Optional[str]:
    if regex.groupindex:
        return next(iter(regex.groupindex))
    else:
        return None

def _get_group_names(regex: re.Pattern) -> Union[Sequence[Union[str, int]], range]:
    """
    Get named groups from compiled regex.

    Unnamed groups are numbered.

    Parameters
    ----------
    regex : re.Pattern
        Compiled regular expression.

    Returns
    -------
    Union[Sequence[Union[str, int]], range]
        List of column labels or a range object.
    """
    rng = range(regex.groups)
    names: dict[int, str] = {v: k for k, v in regex.groupindex.items()}
    if not names:
        return rng
    result = [names.get(1 + i, i) for i in range(regex.groups)]
    arr = np.array(result)
    if arr.dtype.kind == 'i' and lib.is_range_indexer(arr, len(arr)):
        return rng
    return result

def str_extractall(arr: Union[ABCSeries, ABCIndex], pat: str, flags: int = 0) -> Any:
    """
    Extract capture groups in the regex `pat` as columns in DataFrame.

    For each subject string in the Series, extract groups from all
    matches of regular expression pat. When each subject string in the
    Series has exactly one match, extractall(pat).xs(0, level='match')
    is the same as extract(pat).

    Parameters
    ----------
    arr : Union[ABCSeries, ABCIndex]
        Series or Index of strings.
    pat : str
        Regular expression pattern with capturing groups.
    flags : int, default 0
        Flags from the re module.

    Returns
    -------
    DataFrame
        A DataFrame with one row for each match and columns for each capture group.
    """
    regex = re.compile(pat, flags=flags)
    if regex.groups == 0:
        raise ValueError('pattern contains no capture groups')
    from pandas import DataFrame, MultiIndex
    if isinstance(arr, ABCIndex):
        arr = arr.to_series().reset_index(drop=True).astype(arr.dtype)
    columns = _get_group_names(regex)
    match_list: List[List[Any]] = []
    index_list: List[tuple] = []
    is_mi = arr.index.nlevels > 1
    for subject_key, subject in arr.items():
        if isinstance(subject, str):
            if not is_mi:
                subject_key = (subject_key,)
            for match_i, match_tuple in enumerate(regex.findall(subject)):
                if isinstance(match_tuple, str):
                    match_tuple = (match_tuple,)
                na_tuple = [np.nan if group == '' else group for group in match_tuple]
                match_list.append(na_tuple)
                result_key = tuple(subject_key + (match_i,))
                index_list.append(result_key)
    index = MultiIndex.from_tuples(index_list, names=arr.index.names + ['match'])
    dtype = _result_dtype(arr)
    result = arr._constructor_expanddim(match_list, index=index, columns=columns, dtype=dtype)
    return result

# The remainder of the module (including the StringMethods class and its methods)
# is assumed to be part of the codebase and should be similarly annotated.
# Only the functions above are provided here with type annotations.
