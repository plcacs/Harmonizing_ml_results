from typing import List, Any, Optional, Union, Hashable, Tuple
import numpy as np
import re
from pandas import DataFrame, Series, Index, MultiIndex
from pandas._libs import lib
from pandas.core.dtypes.dtypes import ArrowDtype
# Note: StringDtype will be imported locally where needed

def cat_safe(list_of_columns: List[np.ndarray[Any, Any]], sep: str) -> np.ndarray[Any, Any]:
    """
    Auxiliary function for :meth:`str.cat`.

    Parameters
    ----------
    list_of_columns : list of np.ndarray
        List of arrays to be concatenated with sep; these arrays may not contain NaNs!
    sep : str
        The separator string for concatenating the columns.

    Returns
    -------
    np.ndarray
        The concatenation of list_of_columns with sep.
    """
    try:
        result: np.ndarray[Any, Any] = cat_core(list_of_columns, sep)
    except TypeError as err:
        for column in list_of_columns:
            dtype = lib.infer_dtype(column, skipna=True)
            if dtype not in ['string', 'empty']:
                raise TypeError(
                    f'Concatenation requires list-likes containing only strings (or missing values). '
                    f'Offending values found in column {dtype}'
                ) from None
        raise err
    return result

def cat_core(list_of_columns: List[np.ndarray[Any, Any]], sep: str) -> np.ndarray[Any, Any]:
    """
    Auxiliary function for :meth:`str.cat`

    Parameters
    ----------
    list_of_columns : list of np.ndarray
        List of arrays to be concatenated with sep; these arrays may not contain NaNs!
    sep : str
        The separator string for concatenating the columns.

    Returns
    -------
    np.ndarray
        The concatenation of list_of_columns with sep.
    """
    if sep == '':
        arr_of_cols: np.ndarray[Any, Any] = np.asarray(list_of_columns, dtype=object)
        return np.sum(arr_of_cols, axis=0)
    list_with_sep: List[Any] = [sep] * (2 * len(list_of_columns) - 1)
    list_with_sep[::2] = list_of_columns  # type: ignore
    arr_with_sep: np.ndarray[Any, Any] = np.asarray(list_with_sep, dtype=object)
    return np.sum(arr_with_sep, axis=0)

def _result_dtype(arr: Any) -> Any:
    """
    Get the result dtype for the array.

    Parameters
    ----------
    arr : array-like
        Input array.

    Returns
    -------
    Any
        The dtype of the result.
    """
    from pandas.core.arrays.string_ import StringDtype
    if isinstance(arr.dtype, (ArrowDtype, StringDtype)):
        return arr.dtype
    return object

def _get_single_group_name(regex: re.Pattern) -> Optional[str]:
    """
    Get the single group name from the compiled regex.

    Parameters
    ----------
    regex : re.Pattern
        Regular expression pattern.

    Returns
    -------
    Optional[str]
        The name of the single capture group, if one exists.
    """
    if regex.groupindex:
        return next(iter(regex.groupindex))
    else:
        return None

def _get_group_names(regex: re.Pattern) -> List[Hashable]:
    """
    Get named groups from compiled regex.

    Unnamed groups are numbered.

    Parameters
    ----------
    regex : re.Pattern
        Compiled regex.

    Returns
    -------
    List[Hashable]
        List of column labels.
    """
    rng = range(regex.groups)
    names: dict[int, str] = {v: k for (k, v) in regex.groupindex.items()}
    if not names:
        return list(rng)
    result: List[Hashable] = [names.get(1 + i, i) for i in rng]
    arr = np.array(result)
    if arr.dtype.kind == 'i' and lib.is_range_indexer(arr, len(arr)):
        return list(rng)
    return result

def str_extractall(arr: Union[Index, Series], pat: str, flags: int = 0) -> DataFrame:
    """
    Extract capture groups in the regex `pat` as columns in DataFrame.
    
    Parameters
    ----------
    arr : Index or Series
        Input array-like of strings.
    pat : str
        Regular expression pattern with capture groups.
    flags : int, optional
        Regex flags (default is 0).

    Returns
    -------
    DataFrame
        A DataFrame with one row per match, with a MultiIndex including the match number.
    """
    regex: re.Pattern = re.compile(pat, flags=flags)
    if regex.groups == 0:
        raise ValueError('pattern contains no capture groups')
    # If arr is an Index, convert to Series with reset index.
    if isinstance(arr, Index):
        arr = arr.to_series().reset_index(drop=True).astype(arr.dtype)
    columns: List[Hashable] = _get_group_names(regex)
    match_list: List[List[Any]] = []
    index_list: List[Tuple[Any, ...]] = []
    is_mi: bool = arr.index.nlevels > 1
    for subject_key, subject in arr.items():
        if isinstance(subject, str):
            key_tuple: Tuple[Any, ...]
            if not is_mi:
                key_tuple = (subject_key,)
            else:
                key_tuple = subject_key  # type: ignore
            for match_i, match_tuple in enumerate(regex.findall(subject)):
                if isinstance(match_tuple, str):
                    match_tuple = (match_tuple,)
                na_tuple: List[Any] = [np.nan if group == '' else group for group in match_tuple]
                match_list.append(na_tuple)
                result_key: Tuple[Any, ...] = key_tuple + (match_i,)
                index_list.append(result_key)
    index: MultiIndex = MultiIndex.from_tuples(index_list, names=arr.index.names + ['match'])
    dtype = _result_dtype(arr)
    result = arr._constructor_expanddim(match_list, index=index, columns=columns, dtype=dtype)
    return result