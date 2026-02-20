#!/usr/bin/env python
from __future__ import annotations

import codecs
import re
import warnings
from functools import wraps
from typing import Any, Callable, Hashable, Iterator, Literal, Union

import numpy as np
import numpy.typing as npt

from pandas._config import get_option
from pandas._libs import lib
from pandas._typing import DtypeObj, F, NpDtype, Scalar
from pandas.core.dtypes.common import (
    ensure_object,
    is_bool_dtype,
    is_integer,
    is_list_like,
    is_numeric_dtype,
    is_object_dtype,
    is_re,
)
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas.core.arrays import ExtensionArray
from pandas.core.base import NoNewAttributesMixin
from pandas.core.construction import extract_array
from pandas.core.arrays.string_ import StringDtype
from pandas.core.dtypes.generic import ABCDataFrame, ABCIndex, ABCMultiIndex, ABCSeries
from pandas.core.arrays.string_ import ArrowDtype

if __debug__:
    from pandas._typing import TypeCallable  # type: ignore[misc]


def forbid_nonstring_types(
    forbidden: list[str] | None, name: str | None = None
) -> Callable[[F], F]:
    """
    Decorator to forbid non-string types from being passed to a string method.

    Parameters
    ----------
    forbidden : list of str or None
        List of types that are forbidden.
    name : str or None, optional
        Method name for error messages, by default None.
    
    Returns
    -------
    A decorator that wraps a function to forbid non-string types.
    """
    def _forbid_nonstring_types(func: F) -> F:
        @wraps(func)
        def inner(self, *args, **kwargs):
            if not isinstance(self._data, (ABCSeries, ABCIndex)):
                raise TypeError("Only Series and Index are supported")
            return func(self, *args, **kwargs)
        return inner  # type: ignore[return-value]
    return _forbid_nonstring_types


def _map_and_wrap(name: str, docstring: str) -> Callable[[Any], Any]:
    def wrapper(self, *args, **kwargs):
        result = getattr(self._data.array, f"_str_{name}")(*args, **kwargs)
        return self._wrap_result(result)
    wrapper.__doc__ = docstring
    return wrapper


class StringMethods(NoNewAttributesMixin):
    """
    Vectorized string functions for Series and Index objects.
    """

    _inferred_dtype: str

    def __init__(self, data) -> None:
        self._data = data
        self._inferred_dtype = lib.infer_dtype(extract_array(data))
        self._orig = data

    def _wrap_result(
        self,
        result,
        name: str | None = None,
        expand: bool = False,
        fill_value=np.nan,
        returns_string: bool = True,
        dtype: DtypeObj | None = None,
    ):
        # simplified wrap for demonstration purposes
        return result

    @forbid_nonstring_types(["bytes"])
    def cat(
        self,
        others=None,
        sep: str = "",
        na_rep=None,
        join="left",
    ) -> str | ABCSeries | ABCIndex:
        from pandas import Index, Series, concat

        if isinstance(others, str):
            raise ValueError("Did you mean to supply a `sep` keyword?")
        if sep is None:
            sep = ""
        if others is None:
            data = ensure_object(self._data)
            na_mask = isna(data)
            if na_rep is None and na_mask.any():
                return sep.join(data[~na_mask])
            elif na_rep is not None and na_mask.any():
                return sep.join(np.where(na_mask, na_rep, data))
            else:
                return sep.join(data)
        try:
            others = self._get_series_list(others)
        except ValueError as err:
            raise ValueError(
                "If `others` contains arrays or lists (or other "
                "list-likes without an index), these must all be of the same "
                "length as the calling Series/Index."
            ) from err
        if any(not self._data.index.equals(x.index) for x in others):
            others = concat(
                others,
                axis=1,
                join=("inner" if join == "inner" else "outer"),
                keys=range(len(others)),
                sort=False,
            )
            data, others = self._data.align(others, join=join)
            others = [others[x] for x in others]
        all_cols = [ensure_object(x) for x in [self._data] + others]
        na_masks = np.array([isna(x) for x in all_cols])
        union_mask = np.logical_or.reduce(na_masks, axis=0)
        if na_rep is None and union_mask.any():
            result = np.empty(len(self._data), dtype=object)
            np.putmask(result, union_mask, np.nan)
            not_masked = ~union_mask
            result[not_masked] = cat_safe([x[not_masked] for x in all_cols], sep)
        elif na_rep is not None and union_mask.any():
            all_cols = [
                np.where(nm, na_rep, col) for nm, col in zip(na_masks, all_cols)
            ]
            result = cat_safe(all_cols, sep)
        else:
            result = cat_safe(all_cols, sep)
        from pandas import Index, Series
        if isinstance(self._data, ABCIndex):
            if isna(result).all():
                dtype = object
            else:
                dtype = self._data.dtype
            out = Index(result, dtype=dtype, name=self._data.name)
        else:
            index = self._data.index
            cons = self._data._constructor
            out = cons(result, name=self._data.name, index=index, dtype=self._data.dtype)
            out = out.__finalize__(self._data, method="str_cat")
        return out

    def _get_series_list(self, others) -> list[ABCSeries]:
        from pandas import Series, Index, DataFrame
        idx = self._data if isinstance(self._data, ABCIndex) else self._data.index
        if isinstance(others, ABCSeries):
            return [others]
        elif isinstance(others, ABCIndex):
            return [Series(others, index=idx, dtype=others.dtype)]
        elif isinstance(others, DataFrame):
            return [others[x] for x in others]
        elif isinstance(others, np.ndarray) and others.ndim == 2:
            others = DataFrame(others, index=idx)
            return [others[x] for x in others]
        elif is_list_like(others, allow_sets=False):
            try:
                others = list(others)
            except TypeError:
                pass
            else:
                if all(
                    isinstance(x, (ABCSeries, ABCIndex, ExtensionArray))
                    or (isinstance(x, np.ndarray) and x.ndim == 1)
                    for x in others
                ):
                    los: list[ABCSeries] = []
                    while others:
                        los = los + self._get_series_list(others.pop(0))
                    return los
                elif all(not is_list_like(x) for x in others):
                    from pandas import Series
                    return [Series(others, index=idx)]
        raise TypeError(
            "others must be Series, Index, DataFrame, np.ndarray or list-like "
            "containing only strings or objects of type Series/Index/np.ndarray[1-dim]"
        )

    @forbid_nonstring_types(["bytes"])
    def split(
        self,
        pat: str | re.Pattern | None = None,
        *,
        n: int = -1,
        expand: bool = False,
        regex: bool | None = None,
    ):
        if regex is False and is_re(pat):
            raise ValueError(
                "Cannot use a compiled regex as replacement pattern with regex=False"
            )
        if is_re(pat):
            regex = True
        result = self._data.array._str_split(pat, n, expand, regex)
        if self._data.dtype == "category":
            dtype = self._data.dtype.categories.dtype
        else:
            dtype = object if self._data.dtype == object else None
        return self._wrap_result(result, expand=expand, returns_string=expand, dtype=dtype)

    @forbid_nonstring_types(["bytes"])
    def rsplit(self, pat=None, *, n: int = -1, expand: bool = False):
        result = self._data.array._str_rsplit(pat, n=n)
        dtype = object if self._data.dtype == object else None
        return self._wrap_result(result, expand=expand, returns_string=expand, dtype=dtype)

    @forbid_nonstring_types(["bytes"])
    def partition(self, sep: str = " ", expand: bool = True):
        result = self._data.array._str_partition(sep, expand)
        if self._data.dtype == "category":
            dtype = self._data.dtype.categories.dtype
        else:
            dtype = object if self._data.dtype == object else None
        return self._wrap_result(result, expand=expand, returns_string=expand, dtype=dtype)

    @forbid_nonstring_types(["bytes"])
    def rpartition(self, sep: str = " ", expand: bool = True):
        result = self._data.array._str_rpartition(sep, expand)
        if self._data.dtype == "category":
            dtype = self._data.dtype.categories.dtype
        else:
            dtype = object if self._data.dtype == object else None
        return self._wrap_result(result, expand=expand, returns_string=expand, dtype=dtype)

    def get(self, i):
        result = self._data.array._str_get(i)
        return self._wrap_result(result)

    @forbid_nonstring_types(["bytes"])
    def join(self, sep: str):
        result = self._data.array._str_join(sep)
        return self._wrap_result(result)

    @forbid_nonstring_types(["bytes"])
    def contains(
        self,
        pat,
        case: bool = True,
        flags: int = 0,
        na=lib.no_default,
        regex: bool = True,
    ):
        if regex and re.compile(pat).groups:
            warnings.warn(
                "This pattern is interpreted as a regular expression, and has "
                "match groups. To actually get the groups, use str.extract.",
                UserWarning,
                stacklevel=find_stack_level(),
            )
        result = self._data.array._str_contains(pat, case, flags, na, regex)
        return self._wrap_result(result, fill_value=na, returns_string=False)

    @forbid_nonstring_types(["bytes"])
    def match(self, pat: str, case: bool = True, flags: int = 0, na=lib.no_default):
        result = self._data.array._str_match(pat, case=case, flags=flags, na=na)
        return self._wrap_result(result, fill_value=na, returns_string=False)

    @forbid_nonstring_types(["bytes"])
    def fullmatch(self, pat, case: bool = True, flags: int = 0, na=lib.no_default):
        result = self._data.array._str_fullmatch(pat, case=case, flags=flags, na=na)
        return self._wrap_result(result, fill_value=na, returns_string=False)

    @forbid_nonstring_types(["bytes"])
    def replace(
        self,
        pat: str | re.Pattern | dict,
        repl = None,
        n: int = -1,
        case: bool | None = None,
        flags: int = 0,
        regex: bool = False,
    ):
        if isinstance(pat, dict) and repl is not None:
            raise ValueError("repl cannot be used when pat is a dictionary")
        if not isinstance(pat, dict) and not (isinstance(repl, str) or callable(repl)):
            raise TypeError("repl must be a string or callable")
        is_compiled_re = is_re(pat)
        if regex or regex is None:
            if is_compiled_re and (case is not None or flags != 0):
                raise ValueError(
                    "case and flags cannot be set when pat is a compiled regex"
                )
        elif is_compiled_re:
            raise ValueError(
                "Cannot use a compiled regex as replacement pattern with regex=False"
            )
        elif callable(repl):
            raise ValueError("Cannot use a callable replacement when regex=False")
        if case is None:
            case = True
        res_output = self._data
        if not isinstance(pat, dict):
            pat = {pat: repl}
        for key, value in pat.items():
            result = res_output.array._str_replace(
                key, value, n=n, case=case, flags=flags, regex=regex
            )
            res_output = self._wrap_result(result)
        return res_output

    @forbid_nonstring_types(["bytes"])
    def repeat(self, repeats) -> ABCSeries | ABCIndex:
        result = self._data.array._str_repeat(repeats)
        return self._wrap_result(result)

    @forbid_nonstring_types(["bytes"])
    def pad(
        self,
        width,
        side: Literal["left", "right", "both"] = "left",
        fillchar: str = " ",
    ) -> ABCSeries | ABCIndex:
        if not isinstance(fillchar, str):
            msg = f"fillchar must be a character, not {type(fillchar).__name__}"
            raise TypeError(msg)
        if len(fillchar) != 1:
            raise TypeError("fillchar must be a character, not str")
        if not is_integer(width):
            msg = f"width must be of integer type, not {type(width).__name__}"
            raise TypeError(msg)
        result = self._data.array._str_pad(width, side=side, fillchar=fillchar)
        return self._wrap_result(result)

    @forbid_nonstring_types(["bytes"])
    def center(self, width: int, fillchar: str = " ") -> ABCSeries | ABCIndex:
        return self.pad(width, side="both", fillchar=fillchar)

    @forbid_nonstring_types(["bytes"])
    def ljust(self, width: int, fillchar: str = " ") -> ABCSeries | ABCIndex:
        return self.pad(width, side="right", fillchar=fillchar)

    @forbid_nonstring_types(["bytes"])
    def rjust(self, width: int, fillchar: str = " ") -> ABCSeries | ABCIndex:
        return self.pad(width, side="left", fillchar=fillchar)

    @forbid_nonstring_types(["bytes"])
    def zfill(self, width: int) -> ABCSeries | ABCIndex:
        if not is_integer(width):
            msg = f"width must be of integer type, not {type(width).__name__}"
            raise TypeError(msg)
        f = lambda x: x.zfill(width)
        result = self._data.array._str_map(f)
        return self._wrap_result(result)

    def slice(self, start=None, stop=None, step=None) -> ABCSeries | ABCIndex:
        result = self._data.array._str_slice(start, stop, step)
        return self._wrap_result(result)

    @forbid_nonstring_types(["bytes"])
    def slice_replace(self, start=None, stop=None, repl=None) -> ABCSeries | ABCIndex:
        result = self._data.array._str_slice_replace(start, stop, repl)
        return self._wrap_result(result)

    def decode(self, encoding, errors: str = "strict") -> ABCSeries | ABCIndex:
        if encoding in _cpython_optimized_decoders:
            f = lambda x: x.decode(encoding, errors)
        else:
            decoder = codecs.getdecoder(encoding)
            f = lambda x: decoder(x, errors)[0]
        arr = self._data.array
        result = arr._str_map(f)
        dtype = "str" if get_option("future.infer_string") else None
        return self._wrap_result(result, dtype=dtype)

    @forbid_nonstring_types(["bytes"])
    def encode(self, encoding, errors: str = "strict") -> ABCSeries | ABCIndex:
        result = self._data.array._str_encode(encoding, errors)
        return self._wrap_result(result, returns_string=False)

    @forbid_nonstring_types(["bytes"])
    def strip(self, to_strip=None) -> ABCSeries | ABCIndex:
        result = self._data.array._str_strip(to_strip)
        return self._wrap_result(result)

    @forbid_nonstring_types(["bytes"])
    def lstrip(self, to_strip=None) -> ABCSeries | ABCIndex:
        result = self._data.array._str_lstrip(to_strip)
        return self._wrap_result(result)

    @forbid_nonstring_types(["bytes"])
    def rstrip(self, to_strip=None) -> ABCSeries | ABCIndex:
        result = self._data.array._str_rstrip(to_strip)
        return self._wrap_result(result)

    @forbid_nonstring_types(["bytes"])
    def removeprefix(self, prefix: str) -> ABCSeries | ABCIndex:
        result = self._data.array._str_removeprefix(prefix)
        return self._wrap_result(result)

    @forbid_nonstring_types(["bytes"])
    def removesuffix(self, suffix) -> ABCSeries | ABCIndex:
        result = self._data.array._str_removesuffix(suffix)
        return self._wrap_result(result)

    @forbid_nonstring_types(["bytes"])
    def wrap(
        self,
        width,
        expand_tabs: bool = True,
        tabsize: int = 8,
        replace_whitespace: bool = True,
        drop_whitespace: bool = True,
        initial_indent: str = "",
        subsequent_indent: str = "",
        fix_sentence_endings: bool = False,
        break_long_words: bool = True,
        break_on_hyphens: bool = True,
        max_lines: int | None = None,
        placeholder: str = " [...]",
    ) -> ABCSeries | ABCIndex:
        result = self._data.array._str_wrap(
            width=width,
            expand_tabs=expand_tabs,
            tabsize=tabsize,
            replace_whitespace=replace_whitespace,
            drop_whitespace=drop_whitespace,
            initial_indent=initial_indent,
            subsequent_indent=subsequent_indent,
            fix_sentence_endings=fix_sentence_endings,
            break_long_words=break_long_words,
            break_on_hyphens=break_on_hyphens,
            max_lines=max_lines,
            placeholder=placeholder,
        )
        return self._wrap_result(result)

    @forbid_nonstring_types(["bytes"])
    def get_dummies(
        self,
        sep: str = "|",
        dtype: NpDtype | None = None,
    ):
        from pandas.core.frame import DataFrame
        if dtype is not None and not (is_numeric_dtype(dtype) or is_bool_dtype(dtype)):
            raise ValueError("Only numeric or boolean dtypes are supported for 'dtype'")
        result, name = self._data.array._str_get_dummies(sep, dtype)
        if is_extension_array_dtype(dtype):
            return self._wrap_result(
                DataFrame(result, columns=name, dtype=dtype),
                name=name,
                returns_string=False,
            )
        return self._wrap_result(
            result,
            name=name,
            expand=True,
            returns_string=False,
        )

    @forbid_nonstring_types(["bytes"])
    def translate(self, table) -> ABCSeries | ABCIndex:
        result = self._data.array._str_translate(table)
        dtype = object if self._data.dtype == "object" else None
        return self._wrap_result(result, dtype=dtype)

    @forbid_nonstring_types(["bytes"])
    def count(self, pat, flags: int = 0) -> ABCSeries | ABCIndex:
        result = self._data.array._str_count(pat, flags)
        return self._wrap_result(result, returns_string=False)

    @forbid_nonstring_types(["bytes"])
    def startswith(
        self, pat: str | tuple[str, ...], na=lib.no_default
    ) -> ABCSeries | ABCIndex:
        if not isinstance(pat, (str, tuple)):
            msg = f"expected a string or tuple, not {type(pat).__name__}"
            raise TypeError(msg)
        result = self._data.array._str_startswith(pat, na=na)
        return self._wrap_result(result, returns_string=False)

    @forbid_nonstring_types(["bytes"])
    def endswith(
        self, pat, na: Scalar | lib.NoDefault = lib.no_default
    ) -> ABCSeries | ABCIndex:
        if not isinstance(pat, (str, tuple)):
            msg = f"expected a string or tuple, not {type(pat).__name__}"
            raise TypeError(msg)
        result = self._data.array._str_endswith(pat, na=na)
        return self._wrap_result(result, returns_string=False)

    @forbid_nonstring_types(["bytes"])
    def findall(self, pat, flags: int = 0) -> ABCSeries | ABCIndex:
        result = self._data.array._str_findall(pat, flags)
        return self._wrap_result(result, returns_string=False)

    @forbid_nonstring_types(["bytes"])
    def extract(
        self, pat: str, flags: int = 0, expand: bool = True
    ) -> Union[DataFrame, ABCSeries, ABCIndex]:
        from pandas import DataFrame
        if not isinstance(expand, bool):
            raise ValueError("expand must be True or False")
        regex = re.compile(pat, flags=flags)
        if regex.groups == 0:
            raise ValueError("pattern contains no capture groups")
        if not expand and regex.groups > 1 and isinstance(self._data, ABCIndex):
            raise ValueError("only one regex group is supported with Index")
        obj = self._data
        result_dtype = _result_dtype(obj)
        returns_df = regex.groups > 1 or expand
        if returns_df:
            name = None
            columns = _get_group_names(regex)
            if obj.array.size == 0:
                result = DataFrame(columns=columns, dtype=result_dtype)
            else:
                result_list = self._data.array._str_extract(
                    pat, flags=flags, expand=returns_df
                )
                result_index: ABCIndex | None
                if isinstance(obj, ABCSeries):
                    result_index = obj.index
                else:
                    result_index = None
                result = DataFrame(
                    result_list, columns=columns, index=result_index, dtype=result_dtype
                )
        else:
            name = _get_single_group_name(regex)
            result = self._data.array._str_extract(pat, flags=flags, expand=returns_df)
        return self._wrap_result(result, name=name, dtype=result_dtype)

    @forbid_nonstring_types(["bytes"])
    def extractall(self, pat, flags: int = 0) -> DataFrame:
        return str_extractall(self._orig, pat, flags)

    @forbid_nonstring_types(["bytes"])
    def find(self, sub, start: int = 0, end=None) -> ABCSeries | ABCIndex:
        if not isinstance(sub, str):
            msg = f"expected a string object, not {type(sub).__name__}"
            raise TypeError(msg)
        result = self._data.array._str_find(sub, start, end)
        return self._wrap_result(result, returns_string=False)

    @forbid_nonstring_types(["bytes"])
    def rfind(self, sub, start: int = 0, end=None) -> ABCSeries | ABCIndex:
        if not isinstance(sub, str):
            msg = f"expected a string object, not {type(sub).__name__}"
            raise TypeError(msg)
        result = self._data.array._str_rfind(sub, start=start, end=end)
        return self._wrap_result(result, returns_string=False)

    @forbid_nonstring_types(["bytes"])
    def index(self, sub, start: int = 0, end=None) -> ABCSeries | ABCIndex:
        if not isinstance(sub, str):
            msg = f"expected a string object, not {type(sub).__name__}"
            raise TypeError(msg)
        result = self._data.array._str_index(sub, start=start, end=end)
        return self._wrap_result(result, returns_string=False)

    @forbid_nonstring_types(["bytes"])
    def rindex(self, sub, start: int = 0, end=None) -> ABCSeries | ABCIndex:
        if not isinstance(sub, str):
            msg = f"expected a string object, not {type(sub).__name__}"
            raise TypeError(msg)
        result = self._data.array._str_rindex(sub, start=start, end=end)
        return self._wrap_result(result, returns_string=False)

    def len(self) -> ABCSeries | ABCIndex:
        result = self._data.array._str_len()
        return self._wrap_result(result, returns_string=False)

    def lower(self) -> ABCSeries | ABCIndex:
        result = self._data.array._str_lower()
        return self._wrap_result(result)

    def upper(self) -> ABCSeries | ABCIndex:
        result = self._data.array._str_upper()
        return self._wrap_result(result)

    def title(self) -> ABCSeries | ABCIndex:
        result = self._data.array._str_title()
        return self._wrap_result(result)

    def capitalize(self) -> ABCSeries | ABCIndex:
        result = self._data.array._str_capitalize()
        return self._wrap_result(result)

    def swapcase(self) -> ABCSeries | ABCIndex:
        result = self._data.array._str_swapcase()
        return self._wrap_result(result)

    def casefold(self) -> ABCSeries | ABCIndex:
        result = self._data.array._str_casefold()
        return self._wrap_result(result)

    isalnum = _map_and_wrap(
        "isalnum",
        docstring="Check whether all characters are alphanumeric." + "\n" + "See Also: isalpha, isnumeric, isdigit, isdecimal, isspace, islower, isascii, isupper, istitle"
    )
    isalpha = _map_and_wrap(
        "isalpha",
        docstring="Check whether all characters are alphabetic." + "\n" + "See Also: isnumeric, isalnum, isdigit, isdecimal, isspace, islower, isascii, isupper, istitle"
    )
    isdigit = _map_and_wrap(
        "isdigit",
        docstring="Check whether all characters are digits." + "\n" + "See Also: isalpha, isnumeric, isalnum, isdecimal, isspace, islower, isascii, isupper, istitle"
    )
    isspace = _map_and_wrap(
        "isspace",
        docstring="Check whether all characters are whitespace." + "\n" + "See Also: isalpha, isnumeric, isalnum, isdigit, isdecimal, islower, isascii, isupper, istitle"
    )
    islower = _map_and_wrap(
        "islower",
        docstring="Check whether all characters are lowercase." + "\n" + "See Also: isalpha, isnumeric, isalnum, isdigit, isdecimal, isspace, isascii, isupper, istitle"
    )
    isascii = _map_and_wrap(
        "isascii",
        docstring="Check whether all characters are ASCII." + "\n" + "See Also: isalpha, isnumeric, isalnum, isdigit, isdecimal, isspace, islower, isupper, istitle"
    )
    isupper = _map_and_wrap(
        "isupper",
        docstring="Check whether all characters are uppercase." + "\n" + "See Also: isalpha, isnumeric, isalnum, isdigit, isdecimal, isspace, islower, isascii, istitle"
    )
    istitle = _map_and_wrap(
        "istitle",
        docstring="Check whether all characters are titlecase." + "\n" + "See Also: isalpha, isnumeric, isalnum, isdigit, isdecimal, isspace, islower, isascii, isupper"
    )
    isnumeric = _map_and_wrap(
        "isnumeric",
        docstring="Check whether all characters are numeric." + "\n" + "See Also: isalpha, isalnum, isdigit, isdecimal, isspace, islower, isascii, isupper, istitle"
    )
    isdecimal = _map_and_wrap(
        "isdecimal",
        docstring="Check whether all characters are decimal." + "\n" + "See Also: isalpha, isnumeric, isalnum, isdigit, isspace, islower, isascii, isupper, istitle"
    )


def cat_safe(list_of_columns: list[npt.NDArray[np.object_]], sep: str) -> npt.NDArray[np.object_]:
    """
    Auxiliary function for :meth:`str.cat`.

    Same signature as cat_core, but handles TypeErrors in concatenation, which
    happen if the arrays in list_of_columns have the wrong dtypes or content.

    Parameters
    ----------
    list_of_columns : list of numpy arrays
        List of arrays to be concatenated with sep; these arrays may not contain NaNs!
    sep : str
        The separator string for concatenating the columns.

    Returns
    -------
    npt.NDArray[np.object_]
        The concatenation of list_of_columns with sep.
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


def cat_core(list_of_columns: list, sep: str) -> npt.NDArray[np.object_]:
    """
    Auxiliary function for :meth:`str.cat`

    Parameters
    ----------
    list_of_columns : list of numpy arrays
        List of arrays to be concatenated with sep; these arrays may not contain NaNs!
    sep : str
        The separator string for concatenating the columns.

    Returns
    -------
    npt.NDArray[np.object_]
        The concatenation of list_of_columns with sep.
    """
    if sep == "":
        arr_of_cols = np.asarray(list_of_columns, dtype=object)
        return np.sum(arr_of_cols, axis=0)
    list_with_sep = [sep] * (2 * len(list_of_columns) - 1)
    list_with_sep[::2] = list_of_columns
    arr_with_sep = np.asarray(list_with_sep, dtype=object)
    return np.sum(arr_with_sep, axis=0)


def _result_dtype(arr: Any) -> Any:
    # workaround #27953
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
    regex : re.Pattern
        Compiled regular expression.

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


def str_extractall(arr: Union[ABCSeries, ABCIndex], pat: str, flags: int = 0) -> DataFrame:
    regex = re.compile(pat, flags=flags)
    if regex.groups == 0:
        raise ValueError("pattern contains no capture groups")
    if isinstance(arr, ABCIndex):
        arr = arr.to_series().reset_index(drop=True).astype(arr.dtype)
    columns = _get_group_names(regex)
    match_list = []
    index_list = []
    is_mi = arr.index.nlevels > 1
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
    from pandas import MultiIndex
    index = MultiIndex.from_tuples(index_list, names=arr.index.names + ["match"])
    dtype = _result_dtype(arr)
    result = arr._constructor_expanddim(
        match_list, index=index, columns=columns, dtype=dtype
    )
    return result
