from __future__ import annotations
from typing import List, Any, Union, Optional, Hashable
import re
import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, Series, Index, MultiIndex
from pandas._libs import lib


def cat_safe(list_of_columns: List[NDArray[np.object_]], sep: str) -> NDArray[np.object_]:
    """
    Auxiliary function for :meth:`str.cat`.
    """
    try:
        result = cat_core(list_of_columns, sep)
    except TypeError:
        for column in list_of_columns:
            dtype = lib.infer_dtype(column, skipna=True)
            if dtype not in ["string", "empty"]:
                raise TypeError(
                    "Concatenation requires list-likes containing only "
                    "strings (or missing values). Offending values found in "
                    f"column {dtype}"
                ) from None
    return result


def cat_core(list_of_columns: List[NDArray[np.object_]], sep: str) -> NDArray[np.object_]:
    """
    Auxiliary function for :meth:`str.cat`
    """
    if sep == "":
        # no need to interleave sep if it is empty
        arr_of_cols = np.asarray(list_of_columns, dtype=object)
        return np.sum(arr_of_cols, axis=0)
    list_with_sep: List[Any] = [sep] * (2 * len(list_of_columns) - 1)
    list_with_sep[::2] = list_of_columns
    arr_with_sep: NDArray[np.object_] = np.asarray(list_with_sep, dtype=object)
    return np.sum(arr_with_sep, axis=0)


def _result_dtype(arr: Any) -> Any:
    # workaround #27953
    from pandas.core.arrays.string_ import StringDtype
    from pandas.core.dtypes.dtypes import ArrowDtype

    if isinstance(arr.dtype, (ArrowDtype, StringDtype)):
        return arr.dtype
    return object


def _get_single_group_name(regex: re.Pattern) -> Optional[Hashable]:
    if regex.groupindex:
        return next(iter(regex.groupindex))
    else:
        return None


def _get_group_names(regex: re.Pattern) -> Union[List[Hashable], range]:
    """
    Get named groups from compiled regex.
    Unnamed groups are numbered.
    """
    rng: range = range(regex.groups)
    names: dict[int, str] = {v: k for k, v in regex.groupindex.items()}
    if not names:
        return rng
    result: List[Hashable] = [names.get(1 + i, i) for i in rng]
    arr = np.array(result)
    if arr.dtype.kind == "i" and lib.is_range_indexer(arr, len(arr)):
        return rng
    return result


def str_extractall(arr: Union[Series, Index], pat: str, flags: int = 0) -> DataFrame:
    regex: re.Pattern = re.compile(pat, flags=flags)
    if regex.groups == 0:
        raise ValueError("pattern contains no capture groups")
    if isinstance(arr, Index):
        arr = arr.to_series().reset_index(drop=True).astype(arr.dtype)
    columns: Union[List[Hashable], range] = _get_group_names(regex)
    match_list: List[List[Any]] = []
    index_list: List[tuple] = []
    is_mi: bool = arr.index.nlevels > 1

    for subject_key, subject in arr.items():
        if isinstance(subject, str):
            if not is_mi:
                subject_key = (subject_key,)
            for match_i, match_tuple in enumerate(regex.findall(subject)):
                if isinstance(match_tuple, str):
                    match_tuple = (match_tuple,)
                na_tuple: List[Any] = [np.nan if group == "" else group for group in match_tuple]
                match_list.append(na_tuple)
                result_key: tuple = tuple(subject_key + (match_i,))
                index_list.append(result_key)

    index_mi: MultiIndex = MultiIndex.from_tuples(index_list, names=arr.index.names + ["match"])
    dtype = _result_dtype(arr)
    result = arr._constructor_expanddim(match_list, index=index_mi, columns=columns, dtype=dtype)
    return result