#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fully annotated string methods utilities.
"""

from __future__ import annotations

import codecs
import re
import numpy as np
from typing import Any, Callable, Hashable, Iterator, Literal, Union, cast

from pandas._libs import lib
from pandas._typing import DtypeObj, F, NpDtype, Scalar
from pandas.core.arrays.string_ import StringDtype
from pandas.core.dtypes.common import is_bool_dtype, is_numeric_dtype, is_object_dtype
from pandas.core.dtypes.missing import isna
from pandas.core.dtypes.generic import ABCDataFrame, ABCIndex, ABCSeries
from pandas.core.dtypes.dtypes import ArrowDtype
from pandas.util._decorators import Appender

# Shared documentation strings
_shared_docs: dict[str, str] = {}

def _map_and_wrap(name: str, docstring: str) -> Callable[[Any], Any]:
    def wrapper(self: StringMethods, *args: Any, **kwargs: Any) -> Any:
        result = getattr(self._data.array, f"_str_{name}")(*args, **kwargs)
        return self._wrap_result(result)
    wrapper.__doc__ = docstring
    return wrapper

class StringMethods:
    def __init__(self, data: Any) -> None:
        self._data = data

    def _wrap_result(self,
                     result: Any,
                     name: Any = None,
                     expand: bool | None = None,
                     fill_value: Any = np.nan,
                     returns_string: bool = True,
                     dtype: DtypeObj | str | None = None) -> Any:
        # Dummy implementation for annotation purposes.
        return result

def cat_safe(list_of_columns: list[np.ndarray], sep: str) -> np.ndarray:
    """
    Auxiliary function for :meth:`str.cat`.

    Same signature as cat_core, but handles TypeErrors in concatenation, which
    happen if the arrays in list_of_columns have the wrong dtypes or content.

    Parameters
    ----------
    list_of_columns : list of numpy arrays
        List of arrays to be concatenated with sep;
        these arrays may not contain NaNs!
    sep : string
        The separator string for concatenating the columns.

    Returns
    -------
    np.ndarray
        The concatenation of list_of_columns with sep.
    """
    try:
        result = cat_core(list_of_columns, sep)
    except TypeError:
        # if there are any non-string values (wrong dtype or hidden behind
        # object dtype), np.sum will fail; catch and return with better message
        for column in list_of_columns:
            dtype = lib.infer_dtype(column, skipna=True)
            if dtype not in ["string", "empty"]:
                raise TypeError(
                    "Concatenation requires list-likes containing only "
                    "strings (or missing values). Offending values found in "
                    f"column {dtype}"
                ) from None
        raise
    return result

def cat_core(list_of_columns: list[np.ndarray], sep: str) -> np.ndarray:
    """
    Auxiliary function for :meth:`str.cat`

    Parameters
    ----------
    list_of_columns : list of numpy arrays
        List of arrays to be concatenated with sep;
        these arrays may not contain NaNs!
    sep : string
        The separator string for concatenating the columns.

    Returns
    -------
    np.ndarray
        The concatenation of list_of_columns with sep.
    """
    if sep == "":
        # no need to interleave sep if it is empty
        arr_of_cols = np.asarray(list_of_columns, dtype=object)
        return np.sum(arr_of_cols, axis=0)
    list_with_sep = [sep] * (2 * len(list_of_columns) - 1)
    list_with_sep[::2] = list_of_columns
    arr_with_sep = np.asarray(list_with_sep, dtype=object)
    return np.sum(arr_with_sep, axis=0)

def _result_dtype(arr: Any) -> Any:
    # workaround #27953
    # ideally we just pass `dtype=arr.dtype` unconditionally, but this fails
    # when the list of values is empty.
    from pandas.core.arrays.string_ import StringDtype
    if isinstance(arr.dtype, (ArrowDtype, StringDtype)):
        return arr.dtype
    return object

def _get_single_group_name(regex: re.Pattern) -> Hashable:
    if regex.groupindex:
        return next(iter(regex.groupindex))
    else:
        return None

def _get_group_names(regex: re.Pattern) -> list[Hashable] | range:
    """
    Get named groups from compiled regex.

    Unnamed groups are numbered.

    Parameters
    ----------
    regex : compiled regex

    Returns
    -------
    list of column labels
    """
    rng = range(regex.groups)
    names = {v: k for k, v in regex.groupindex.items()}
    if not names:
        return rng
    result: list[Hashable] = [names.get(1 + i, i) for i in rng]
    arr = np.array(result)
    if arr.dtype.kind == "i" and lib.is_range_indexer(arr, len(arr)):
        return rng
    return result

def str_extractall(arr: Any, pat: str, flags: int = 0) -> Any:
    """
    Extract capture groups in the regex `pat` as columns in DataFrame.

    For each subject string in the Series, extract groups from all matches
    of regular expression pat. When each subject string in the Series has exactly one match,
    extractall(pat).xs(0, level='match') is the same as extract(pat).

    Parameters
    ----------
    arr : Any
        Subject array (Series or Index) to extract from.
    pat : str
        Regular expression pattern with capturing groups.
    flags : int, default 0 (no flags)
        Flags for the re module (e.g., re.IGNORECASE).

    Returns
    -------
    DataFrame
        A DataFrame with one row for each match and one column for each group.
    """
    regex = re.compile(pat, flags=flags)
    # the regex must contain capture groups.
    if regex.groups == 0:
        raise ValueError("pattern contains no capture groups")

    from pandas import DataFrame, MultiIndex

    if hasattr(arr, "to_series") and isinstance(arr, ABCIndex):
        arr = arr.to_series().reset_index(drop=True).astype(arr.dtype)

    columns = _get_group_names(regex)
    match_list: list[list[Any]] = []
    index_list: list[tuple[Any, ...]] = []
    is_mi = hasattr(arr, "index") and (arr.index.nlevels > 1)

    for subject_key, subject in arr.items():
        if isinstance(subject, str):
            if not is_mi:
                subject_key = (subject_key,)
            for match_i, match_tuple in enumerate(regex.findall(subject)):
                if isinstance(match_tuple, str):
                    match_tuple = (match_tuple,)
                na_tuple = [np.nan if group == "" else group for group in match_tuple]
                match_list.append(na_tuple)
                result_key = tuple(subject_key + (match_i,))
                index_list.append(result_key)

    index = MultiIndex.from_tuples(index_list, names=getattr(arr.index, "names", None) or [] + ["match"])
    dtype = _result_dtype(arr)
    result = arr._constructor_expanddim(  # type: ignore[attr-defined]
        match_list, index=index, columns=columns, dtype=dtype
    )
    return result

_doc_args: dict[str, dict[str, str]] = {}
_doc_args["lower"] = {"type": "lowercase", "method": "lower", "version": ""}
_doc_args["upper"] = {"type": "uppercase", "method": "upper", "version": ""}
_doc_args["title"] = {"type": "titlecase", "method": "title", "version": ""}
_doc_args["capitalize"] = {"type": "be capitalized", "method": "capitalize", "version": ""}
_doc_args["swapcase"] = {"type": "be swapcased", "method": "swapcase", "version": ""}
_doc_args["casefold"] = {"type": "be casefolded", "method": "casefold", "version": ""}

_shared_docs["casemethods"] = """
Convert strings in the Series/Index to %(type)s.
%(version)s
Equivalent to :meth:`str.%(method)s`.

Returns
-------
Series or Index of objects
    A Series or Index where the strings are modified by :meth:`str.%(method)s`.

See Also
--------
Series.str.lower : Converts all characters to lowercase.
Series.str.upper : Converts all characters to uppercase.
Series.str.title : Converts first character of each word to uppercase and
    remaining to lowercase.
Series.str.capitalize : Converts first character to uppercase and
    remaining to lowercase.
Series.str.swapcase : Converts uppercase to lowercase and lowercase to
    uppercase.
Series.str.casefold: Removes all case distinctions in the string.

Examples
--------
>>> s = pd.Series(['lower', 'CAPITALS', 'this is a sentence', 'SwApCaSe'])
>>> s
0                 lower
1              CAPITALS
2    this is a sentence
3              SwApCaSe
dtype: object

>>> s.str.lower()
0                 lower
1              capitals
2    this is a sentence
3              swapcase
dtype: object

>>> s.str.upper()
0                 LOWER
1              CAPITALS
2    THIS IS A SENTENCE
3              SWAPCASE
dtype: object

>>> s.str.title()
0                 Lower
1              Capitals
2    This Is A Sentence
3              Swapcase
dtype: object

>>> s.str.capitalize()
0                 Lower
1              Capitals
2    This is a sentence
3              Swapcase
dtype: object

>>> s.str.swapcase()
0                 LOWER
1              capitals
2    THIS IS A SENTENCE
3              sWaPcAsE
dtype: object
"""

# Mapping of methods using _map_and_wrap for various is* methods.
isalnum = _map_and_wrap(
    "isalnum",
    docstring=_shared_docs["casemethods"] % _doc_args["isalnum"] + "\n" + "See Also\n--------\nSeries.str.isalnum : Check whether all characters are alphanumeric.\n"
)
isalpha = _map_and_wrap(
    "isalpha",
    docstring=_shared_docs["casemethods"] % _doc_args["isalpha"] + "\n" + "See Also\n--------\nSeries.str.isalpha : Check whether all characters are alphabetic.\n"
)
isdigit = _map_and_wrap(
    "isdigit",
    docstring=_shared_docs["casemethods"] % _doc_args["isdigit"] + "\n" + "See Also\n--------\nSeries.str.isdigit : Check whether all characters are digits.\n"
)
isspace = _map_and_wrap(
    "isspace",
    docstring=_shared_docs["casemethods"] % _doc_args["isspace"] + "\n" + "See Also\n--------\nSeries.str.isspace : Check whether all characters are whitespace.\n"
)
islower = _map_and_wrap(
    "islower",
    docstring=_shared_docs["casemethods"] % _doc_args["islower"] + "\n" + "See Also\n--------\nSeries.str.islower : Check whether all characters are lowercase.\n"
)
isascii = _map_and_wrap(
    "isascii",
    docstring=_shared_docs["casemethods"] % _doc_args["isascii"] + "\n" + "See Also\n--------\nSeries.str.isascii : Check whether all characters are ascii.\n"
)
isupper = _map_and_wrap(
    "isupper",
    docstring=_shared_docs["casemethods"] % _doc_args["isupper"] + "\n" + "See Also\n--------\nSeries.str.isupper : Check whether all characters are uppercase.\n"
)
istitle = _map_and_wrap(
    "istitle",
    docstring=_shared_docs["casemethods"] % _doc_args["istitle"] + "\n" + "See Also\n--------\nSeries.str.istitle : Check whether all characters are titlecase.\n"
)
isnumeric = _map_and_wrap(
    "isnumeric",
    docstring=_shared_docs["casemethods"] % _doc_args["isnumeric"] + "\n" + "See Also\n--------\nSeries.str.isnumeric : Check whether all characters are numeric.\n"
)
isdecimal = _map_and_wrap(
    "isdecimal",
    docstring=_shared_docs["casemethods"] % _doc_args["isdecimal"] + "\n" + "See Also\n--------\nSeries.str.isdecimal : Check whether all characters are decimal.\n"
)

# End of fully annotated code.
