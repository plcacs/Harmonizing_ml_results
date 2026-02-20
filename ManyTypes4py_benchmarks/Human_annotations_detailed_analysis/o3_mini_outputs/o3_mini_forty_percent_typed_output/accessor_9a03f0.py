from __future__ import annotations

import codecs
from functools import wraps
import re
import warnings
from typing import Any, Callable, Hashable, Iterator, Union

import numpy as np

from pandas._config import get_option
from pandas._libs import lib
from pandas._typing import AlignJoin, DtypeObj, F, Scalar, npt

from pandas.util._decorators import Appender
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import (
    ensure_object,
    is_bool_dtype,
    is_extension_array_dtype,
    is_integer,
    is_list_like,
    is_numeric_dtype,
    is_object_dtype,
    is_re,
)
from pandas.core.dtypes.dtypes import ArrowDtype, CategoricalDtype
from pandas.core.dtypes.generic import ABCDataFrame, ABCIndex, ABCMultiIndex, ABCSeries
from pandas.core.dtypes.missing import isna

from pandas.core.arrays import ExtensionArray
from pandas.core.base import NoNewAttributesMixin
from pandas.core.construction import extract_array

if False:
    # TYPE_CHECKING
    from collections.abc import Callable, Hashable, Iterator
    from pandas._typing import NpDtype
    from pandas import DataFrame, Index, Series

_shared_docs: dict[str, str] = {}
_cpython_optimized_encoders = (
    "utf-8",
    "utf8",
    "latin-1",
    "latin1",
    "iso-8859-1",
    "mbcs",
    "ascii",
)
_cpython_optimized_decoders = _cpython_optimized_encoders + ("utf-16", "utf-32")


def forbid_nonstring_types(
    forbidden, name: str | None = None
) -> Callable[[F], F]:
    """
    Decorator to forbid specific types for a method of StringMethods.
    [docstring truncated for brevity]
    """
    forbidden = [] if forbidden is None else forbidden

    allowed_types = {"string", "empty", "bytes", "mixed", "mixed-integer"} - set(
        forbidden
    )

    def _forbid_nonstring_types(func) -> F:
        func_name = func.__name__ if name is None else name

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if self._inferred_dtype not in allowed_types:
                msg = (
                    f"Cannot use .str.{func_name} with values of "
                    f"inferred dtype '{self._inferred_dtype}'."
                )
                raise TypeError(msg)
            return func(self, *args, **kwargs)

        wrapper.__name__ = func_name
        return wrapper  # type: ignore

    return _forbid_nonstring_types


def _map_and_wrap(name, docstring):
    @forbid_nonstring_types(["bytes"], name=name)
    def wrapper(self):
        result = getattr(self._data.array, f"_str_{name}")()
        return self._wrap_result(
            result, returns_string=name not in ("isnumeric", "isdecimal")
        )

    wrapper.__doc__ = docstring
    return wrapper


class StringMethods(NoNewAttributesMixin):
    """
    Vectorized string functions for Series and Index.
    [docstring truncated for brevity]
    """

    def __init__(self, data) -> None:
        from pandas.core.arrays.string_ import StringDtype

        self._inferred_dtype = self._validate(data)
        self._is_categorical = isinstance(data.dtype, CategoricalDtype)
        self._is_string = isinstance(data.dtype, StringDtype)
        self._data = data

        self._index = self._name = None
        if isinstance(data, ABCSeries):
            self._index = data.index
            self._name = data.name

        self._parent = data._values.categories if self._is_categorical else data
        self._orig = data
        self._freeze()

    @staticmethod
    def _validate(data):
        """
        Auxiliary function for StringMethods, infers and checks dtype of data.
        [docstring truncated for brevity]
        """
        if isinstance(data, ABCMultiIndex):
            raise AttributeError(
                "Can only use .str accessor with Index, not MultiIndex"
            )

        allowed_types = ["string", "empty", "bytes", "mixed", "mixed-integer"]

        data = extract_array(data)

        values = getattr(data, "categories", data)

        inferred_dtype = lib.infer_dtype(values, skipna=True)

        if inferred_dtype not in allowed_types:
            raise AttributeError(
                f"Can only use .str accessor with string values, not {inferred_dtype}"
            )
        return inferred_dtype

    def __getitem__(self, key):
        result = self._data.array._str_getitem(key)
        return self._wrap_result(result)

    def __iter__(self) -> Iterator:
        raise TypeError(f"'{type(self).__name__}' object is not iterable")

    def _wrap_result(
        self,
        result,
        name=None,
        expand: bool | None = None,
        fill_value=np.nan,
        returns_string=True,
        dtype=None,
    ):
        from pandas import Index, MultiIndex

        if not hasattr(result, "ndim") or not hasattr(result, "dtype"):
            if isinstance(result, ABCDataFrame):
                result = result.__finalize__(self._orig, name="str")
            return result
        assert result.ndim < 3

        if expand is None:
            expand = result.ndim != 1
        elif expand is True and not isinstance(self._orig, ABCIndex):
            if isinstance(result.dtype, ArrowDtype):
                import pyarrow as pa
                from pandas.compat import pa_version_under11p0
                from pandas.core.arrays.arrow.array import ArrowExtensionArray

                value_lengths = pa.compute.list_value_length(result._pa_array)
                max_len = pa.compute.max(value_lengths).as_py()
                min_len = pa.compute.min(value_lengths).as_py()
                if result._hasna:
                    result = ArrowExtensionArray(
                        result._pa_array.fill_null([None] * max_len)
                    )
                if min_len < max_len:
                    if not pa_version_under11p0:
                        result = ArrowExtensionArray(
                            pa.compute.list_slice(
                                result._pa_array,
                                start=0,
                                stop=max_len,
                                return_fixed_size_list=True,
                            )
                        )
                    else:
                        all_null = np.full(max_len, fill_value=None, dtype=object)
                        values = result.to_numpy()
                        new_values = []
                        for row in values:
                            if len(row) < max_len:
                                nulls = all_null[: max_len - len(row)]
                                row = np.append(row, nulls)
                            new_values.append(row)
                        pa_type = result._pa_array.type
                        result = ArrowExtensionArray(pa.array(new_values, type=pa_type))
                if name is None:
                    name = range(max_len)
                result = (
                    pa.compute.list_flatten(result._pa_array)
                    .to_numpy()
                    .reshape(len(result), max_len)
                )
                result = {
                    label: ArrowExtensionArray(pa.array(res))
                    for label, res in zip(name, result.T)
                }
            elif is_object_dtype(result):

                def cons_row(x):
                    if is_list_like(x):
                        return x
                    else:
                        return [x]

                result = [cons_row(x) for x in result]
                if result and not self._is_string:
                    max_len = max(len(x) for x in result)
                    result = [
                        x * max_len if len(x) == 0 or x[0] is np.nan else x
                        for x in result
                    ]

        if not isinstance(expand, bool):
            raise ValueError("expand must be True or False")

        if expand is False:
            if name is None:
                name = getattr(result, "name", None)
            if name is None:
                name = self._orig.name

        if isinstance(self._orig, ABCIndex):
            if is_bool_dtype(result):
                return result

            if expand:
                result = list(result)
                out: Index = MultiIndex.from_tuples(result, names=name)
                if out.nlevels == 1:
                    out = out.get_level_values(0)
                return out
            else:
                return Index(result, name=name, dtype=dtype)
        else:
            index = self._orig.index
            _dtype: DtypeObj | str | None = dtype
            vdtype = getattr(result, "dtype", None)
            if _dtype is not None:
                pass
            elif self._is_string:
                if is_bool_dtype(vdtype):
                    _dtype = result.dtype
                elif returns_string:
                    _dtype = self._orig.dtype
                else:
                    _dtype = vdtype
            elif vdtype is not None:
                _dtype = vdtype

            if expand:
                cons = self._orig._constructor_expanddim
                result = cons(result, columns=name, index=index, dtype=_dtype)
            else:
                cons = self._orig._constructor
                result = cons(result, name=name, index=index, dtype=_dtype)
            result = result.__finalize__(self._orig, method="str")
            if name is not None and result.ndim == 1:
                result.name = name
            return result

    def _get_series_list(self, others):
        """
        Auxiliary function for :meth:`str.cat`.
        [docstring truncated for brevity]
        """
        from pandas import DataFrame, Series

        idx = self._orig if isinstance(self._orig, ABCIndex) else self._orig.index

        if isinstance(others, ABCSeries):
            return [others]
        elif isinstance(others, ABCIndex):
            return [Series(others, index=idx, dtype=others.dtype)]
        elif isinstance(others, ABCDataFrame):
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
                    los: list[Series] = []
                    while others:
                        los = los + self._get_series_list(others.pop(0))
                    return los
                elif all(not is_list_like(x) for x in others):
                    return [Series(others, index=idx)]
        raise TypeError(
            "others must be Series, Index, DataFrame, np.ndarray "
            "or list-like (either containing only strings or "
            "containing only objects of type Series/Index/"
            "np.ndarray[1-dim])"
        )

    @forbid_nonstring_types(["bytes", "mixed", "mixed-integer"])
    def cat(
        self,
        others=None,
        sep: str | None = None,
        na_rep=None,
        join="left",
    ) -> str | Series | Index:
        """
        Concatenate strings in the Series/Index with given separator.
        [docstring truncated for brevity]
        """
        from pandas import Index, Series, concat

        if isinstance(others, str):
            raise ValueError("Did you mean to supply a `sep` keyword?")
        if sep is None:
            sep = ""

        if isinstance(self._orig, ABCIndex):
            data = Series(self._orig, index=self._orig, dtype=self._orig.dtype)
        else:
            data = self._orig

        if others is None:
            data = ensure_object(data)  # type: ignore[assignment]
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
                "list-likes without an index), these must all be "
                "of the same length as the calling Series/Index."
            ) from err

        if any(not data.index.equals(x.index) for x in others):
            others = concat(
                others,
                axis=1,
                join=(join if join == "inner" else "outer"),
                keys=range(len(others)),
                sort=False,
            )
            data, others = data.align(others, join=join)
            others = [others[x] for x in others]

        all_cols = [ensure_object(x) for x in [data] + others]
        na_masks = np.array([isna(x) for x in all_cols])
        union_mask = np.logical_or.reduce(na_masks, axis=0)

        if na_rep is None and union_mask.any():
            result = np.empty(len(data), dtype=object)
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

        out: Index | Series
        if isinstance(self._orig.dtype, CategoricalDtype):
            dtype = self._orig.dtype.categories.dtype
        else:
            dtype = self._orig.dtype
        if isinstance(self._orig, ABCIndex):
            if isna(result).all():
                dtype = object
            out = Index(result, dtype=dtype, name=self._orig.name)
        else:
            res_ser = Series(
                result, dtype=dtype, index=data.index, name=self._orig.name, copy=False
            )
            out = res_ser.__finalize__(self._orig, method="str_cat")
        return out

    _shared_docs["str_split"] = r"""
    Split strings around given separator/delimiter.
    [docstring truncated]
    """

    @Appender(
        _shared_docs["str_split"]
        % {
            "side": "beginning",
            "pat_regex": " or compiled regex",
            "pat_description": "String or regular expression to split on",
            "regex_argument": """
    regex : bool, default None
        Determines if the passed-in pattern is a regular expression:
        - If ``True``, assumes the passed-in pattern is a regular expression
        - If ``False``, treats the pattern as a literal string.
        - If ``None`` and `pat` length is 1, treats `pat` as a literal string.
        - If ``None`` and `pat` length is not 1, treats `pat` as a regular expression.
        - Cannot be set to False if `pat` is a compiled regex
         """,
            "raises_split": """
                      Raises
                      ------
                      ValueError
                          * if `regex` is False and `pat` is a compiled regex
                      """,
            "regex_pat_note": """
    Use of `regex =False` with a `pat` as a compiled regex will raise an error.
            """,
            "method": "split",
            "regex_examples": r"""
    [examples truncated]
    """,
        }
    )
    @forbid_nonstring_types(["bytes"])
    def split(
        self,
        pat=None,
        *,
        n=-1,
        expand=False,
        regex=None,
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
        return self._wrap_result(
            result, expand=expand, returns_string=expand, dtype=dtype
        )

    @Appender(
        _shared_docs["str_split"]
        % {
            "side": "end",
            "pat_regex": "",
            "pat_description": "String to split on",
            "regex_argument": "",
            "raises_split": "",
            "regex_pat_note": "",
            "method": "rsplit",
            "regex_examples": "",
        }
    )
    @forbid_nonstring_types(["bytes"])
    def rsplit(self, pat=None, *, n=-1, expand=False):
        result = self._data.array._str_rsplit(pat, n=n)
        dtype = object if self._data.dtype == object else None
        return self._wrap_result(
            result, expand=expand, returns_string=expand, dtype=dtype
        )

    _shared_docs["str_partition"] = """
    Split the string at the %(side)s occurrence of `sep`.
    [docstring truncated]
    """

    @Appender(
        _shared_docs["str_partition"]
        % {
            "side": "first",
            "return": "3 elements containing the string itself, followed by two empty strings",
            "also": "rpartition : Split the string at the last occurrence of `sep`.",
        }
    )
    @forbid_nonstring_types(["bytes"])
    def partition(self, sep=" ", expand: bool = True):
        result = self._data.array._str_partition(sep, expand)
        if self._data.dtype == "category":
            dtype = self._data.dtype.categories.dtype
        else:
            dtype = object if self._data.dtype == object else None
        return self._wrap_result(
            result, expand=expand, returns_string=expand, dtype=dtype
        )

    @Appender(
        _shared_docs["str_partition"]
        % {
            "side": "last",
            "return": "3 elements containing two empty strings, followed by the string itself",
            "also": "partition : Split the string at the first occurrence of `sep`.",
        }
    )
    @forbid_nonstring_types(["bytes"])
    def rpartition(self, sep: str = " ", expand=False):
        result = self._data.array._str_rpartition(sep, expand)
        if self._data.dtype == "category":
            dtype = self._data.dtype.categories.dtype
        else:
            dtype = object if self._data.dtype == object else None
        return self._wrap_result(
            result, expand=expand, returns_string=expand, dtype=dtype
        )

    def get(self, i):
        """
        Extract element from each component at specified position or with specified key.
        [docstring truncated]
        """
        result = self._data.array._str_get(i)
        return self._wrap_result(result)

    @forbid_nonstring_types(["bytes"])
    def join(self, sep: str):
        """
        Join lists contained as elements in the Series/Index with passed delimiter.
        [docstring truncated]
        """
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
        r"""
        Test if pattern or regex is contained within a string of a Series or Index.
        [docstring truncated]
        """
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
    def match(self, pat, case=True, flags: int = 0, na=lib.no_default):
        """
        Determine if each string starts with a match of a regular expression.
        [docstring truncated]
        """
        result = self._data.array._str_match(pat, case=case, flags=flags, na=na)
        return self._wrap_result(result, fill_value=na, returns_string=False)

    @forbid_nonstring_types(["bytes"])
    def fullmatch(self, pat, case: bool = True, flags=0, na=lib.no_default):
        """
        Determine if each string entirely matches a regular expression.
        [docstring truncated]
        """
        result = self._data.array._str_fullmatch(pat, case=case, flags=flags, na=na)
        return self._wrap_result(result, fill_value=na, returns_string=False)

    @forbid_nonstring_types(["bytes"])
    def replace(
        self,
        pat,
        repl: str | Callable | None = None,
        n=-1,
        case=None,
        flags=0,
        regex: bool = False,
    ):
        r"""
        Replace each occurrence of pattern/regex in the Series/Index.
        [docstring truncated]
        """
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
    def repeat(self, repeats):
        """
        Duplicate each string in the Series or Index.
        [docstring truncated]
        """
        result = self._data.array._str_repeat(repeats)
        return self._wrap_result(result)

    @forbid_nonstring_types(["bytes"])
    def pad(
        self,
        width,
        side="left",
        fillchar: str = " ",
    ):
        """
        Pad strings in the Series/Index up to width.
        [docstring truncated]
        """
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

    _shared_docs["str_pad"] = """
    Pad %(side)s side of strings in the Series/Index.
    [docstring truncated]
    """

    @Appender(_shared_docs["str_pad"] % {"side": "left and right", "method": "center"})
    @forbid_nonstring_types(["bytes"])
    def center(self, width: int, fillchar: str = " ") -> Series | Index:
        return self.pad(width, side="both", fillchar=fillchar)

    @Appender(_shared_docs["str_pad"] % {"side": "right", "method": "ljust"})
    @forbid_nonstring_types(["bytes"])
    def ljust(self, width, fillchar: str = " ") -> Series | Index:
        return self.pad(width, side="right", fillchar=fillchar)

    @Appender(_shared_docs["str_pad"] % {"side": "left", "method": "rjust"})
    @forbid_nonstring_types(["bytes"])
    def rjust(self, width, fillchar: str = " ") -> Series | Index:
        return self.pad(width, side="left", fillchar=fillchar)

    @forbid_nonstring_types(["bytes"])
    def zfill(self, width: int):
        """
        Pad strings in the Series/Index by prepending '0' characters.
        [docstring truncated]
        """
        if not is_integer(width):
            msg = f"width must be of integer type, not {type(width).__name__}"
            raise TypeError(msg)
        f = lambda x: x.zfill(width)
        result = self._data.array._str_map(f)
        return self._wrap_result(result)

    def slice(self, start=None, stop=None, step=None):
        """
        Slice substrings from each element in the Series or Index.
        [docstring truncated]
        """
        result = self._data.array._str_slice(start, stop, step)
        return self._wrap_result(result)

    @forbid_nonstring_types(["bytes"])
    def slice_replace(self, start=None, stop=None, repl=None):
        """
        Replace a positional slice of a string with another value.
        [docstring truncated]
        """
        result = self._data.array._str_slice_replace(start, stop, repl)
        return self._wrap_result(result)

    def decode(self, encoding, errors="strict"):
        """
        Decode character string in the Series/Index using indicated encoding.
        [docstring truncated]
        """
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
    def encode(self, encoding, errors="strict"):
        """
        Encode character string in the Series/Index using indicated encoding.
        [docstring truncated]
        """
        result = self._data.array._str_encode(encoding, errors)
        return self._wrap_result(result, returns_string=False)

    _shared_docs["str_strip"] = r"""
    Remove %(position)s characters.
    [docstring truncated]
    """

    @Appender(
        _shared_docs["str_strip"]
        % {
            "side": "left and right sides",
            "method": "strip",
            "position": "leading and trailing",
        }
    )
    @forbid_nonstring_types(["bytes"])
    def strip(self, to_strip: str | None = None):
        result = self._data.array._str_strip(to_strip)
        return self._wrap_result(result)

    @Appender(
        _shared_docs["str_strip"]
        % {"side": "left side", "method": "lstrip", "position": "leading"}
    )
    @forbid_nonstring_types(["bytes"])
    def lstrip(self, to_strip: str | None = None):
        result = self._data.array._str_lstrip(to_strip)
        return self._wrap_result(result)

    @Appender(
        _shared_docs["str_strip"]
        % {"side": "right side", "method": "rstrip", "position": "trailing"}
    )
    @forbid_nonstring_types(["bytes"])
    def rstrip(self, to_strip: str | None = None):
        result = self._data.array._str_rstrip(to_strip)
        return self._wrap_result(result)

    _shared_docs["str_removefix"] = r"""
    Remove a %(side)s from an object series.
    [docstring truncated]
    """

    @Appender(
        _shared_docs["str_removefix"] % {"side": "prefix", "other_side": "suffix"}
    )
    @forbid_nonstring_types(["bytes"])
    def removeprefix(self, prefix: str):
        result = self._data.array._str_removeprefix(prefix)
        return self._wrap_result(result)

    @Appender(
        _shared_docs["str_removefix"] % {"side": "suffix", "other_side": "prefix"}
    )
    @forbid_nonstring_types(["bytes"])
    def removesuffix(self, suffix):
        result = self._data.array._str_removesuffix(suffix)
        return self._wrap_result(result)

    @forbid_nonstring_types(["bytes"])
    def wrap(
        self,
        width,
        expand_tabs=True,
        tabsize=8,
        replace_whitespace=True,
        drop_whitespace: bool = True,
        initial_indent: str = "",
        subsequent_indent: str = "",
        fix_sentence_endings=False,
        break_long_words=True,
        break_on_hyphens: bool = True,
        max_lines=None,
        placeholder: str = " [...]",
    ):
        r"""
        Wrap strings in Series/Index at specified line width.
        [docstring truncated]
        """
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
        dtype=None,
    ):
        """
        Return DataFrame of dummy/indicator variables for Series.
        [docstring truncated]
        """
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
    def translate(self, table):
        """
        Map all characters in the string through the given mapping table.
        [docstring truncated]
        """
        result = self._data.array._str_translate(table)
        dtype = object if self._data.dtype == "object" else None
        return self._wrap_result(result, dtype=dtype)

    @forbid_nonstring_types(["bytes"])
    def count(self, pat, flags=0):
        r"""
        Count occurrences of pattern in each string of the Series/Index.
        [docstring truncated]
        """
        result = self._data.array._str_count(pat, flags)
        return self._wrap_result(result, returns_string=False)

    @forbid_nonstring_types(["bytes"])
    def startswith(
        self, pat: str | tuple[str, ...], na: Scalar | lib.NoDefault = lib.no_default
    ) -> Series | Index:
        """
        Test if the start of each string element matches a pattern.
        [docstring truncated]
        """
        if not isinstance(pat, (str, tuple)):
            msg = f"expected a string or tuple, not {type(pat).__name__}"
            raise TypeError(msg)
        result = self._data.array._str_startswith(pat, na=na)
        return self._wrap_result(result, returns_string=False)

    @forbid_nonstring_types(["bytes"])
    def endswith(
        self, pat, na=lib.no_default
    ) -> Series | Index:
        """
        Test if the end of each string element matches a pattern.
        [docstring truncated]
        """
        if not isinstance(pat, (str, tuple)):
            msg = f"expected a string or tuple, not {type(pat).__name__}"
            raise TypeError(msg)
        result = self._data.array._str_endswith(pat, na=na)
        return self._wrap_result(result, returns_string=False)

    @forbid_nonstring_types(["bytes"])
    def findall(self, pat, flags=0):
        """
        Find all occurrences of pattern or regular expression in the Series/Index.
        [docstring truncated]
        """
        result = self._data.array._str_findall(pat, flags)
        return self._wrap_result(result, returns_string=False)

    @forbid_nonstring_types(["bytes"])
    def extract(
        self, pat, flags=0, expand=True
    ) -> DataFrame | Series | Index:
        r"""
        Extract capture groups in the regex `pat` as columns in a DataFrame.
        [docstring truncated]
        """
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

                result_index: Index | None
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
        r"""
        Extract capture groups in the regex `pat` as columns in DataFrame.
        [docstring truncated]
        """
        return str_extractall(self._orig, pat, flags)


    _shared_docs["find"] = """
    Return %(side)s indexes in each strings in the Series/Index.
    [docstring truncated]
    """

    @Appender(
        _shared_docs["find"]
        % {
            "side": "lowest",
            "method": "find",
            "also": "rfind : Return highest indexes in each strings.",
        }
    )
    @forbid_nonstring_types(["bytes"])
    def find(self, sub, start: int = 0, end=None):
        if not isinstance(sub, str):
            msg = f"expected a string object, not {type(sub).__name__}"
            raise TypeError(msg)
        result = self._data.array._str_find(sub, start, end)
        return self._wrap_result(result, returns_string=False)

    @Appender(
        _shared_docs["find"]
        % {
            "side": "highest",
            "method": "rfind",
            "also": "find : Return lowest indexes in each strings.",
        }
    )
    @forbid_nonstring_types(["bytes"])
    def rfind(self, sub, start: int = 0, end=None):
        if not isinstance(sub, str):
            msg = f"expected a string object, not {type(sub).__name__}"
            raise TypeError(msg)
        result = self._data.array._str_rfind(sub, start=start, end=end)
        return self._wrap_result(result, returns_string=False)

    @forbid_nonstring_types(["bytes"])
    def normalize(self, form):
        """
        Return the Unicode normal form for the strings in the Series/Index.
        [docstring truncated]
        """
        result = self._data.array._str_normalize(form)
        return self._wrap_result(result)

    _shared_docs["index"] = """
    Return %(side)s indexes in each string in Series/Index.
    [docstring truncated]
    """

    @Appender(
        _shared_docs["index"]
        % {
            "side": "lowest",
            "similar": "find",
            "method": "index",
            "also": "rindex : Return highest indexes in each strings.",
        }
    )
    @forbid_nonstring_types(["bytes"])
    def index(self, sub, start=0, end=None):
        if not isinstance(sub, str):
            msg = f"expected a string object, not {type(sub).__name__}"
            raise TypeError(msg)
        result = self._data.array._str_index(sub, start=start, end=end)
        return self._wrap_result(result, returns_string=False)

    @Appender(
        _shared_docs["index"]
        % {
            "side": "highest",
            "similar": "rfind",
            "method": "rindex",
            "also": "index : Return lowest indexes in each strings.",
        }
    )
    @forbid_nonstring_types(["bytes"])
    def rindex(self, sub, start: int = 0, end=None):
        if not isinstance(sub, str):
            msg = f"expected a string object, not {type(sub).__name__}"
            raise TypeError(msg)
        result = self._data.array._str_rindex(sub, start=start, end=end)
        return self._wrap_result(result, returns_string=False)

    def len(self):
        """
        Compute the length of each element in the Series/Index.
        [docstring truncated]
        """
        result = self._data.array._str_len()
        return self._wrap_result(result, returns_string=False)

    _doc_args: dict[str, dict[str, str]] = {}
    _doc_args["lower"] = {"type": "lowercase", "method": "lower", "version": ""}
    _doc_args["upper"] = {"type": "uppercase", "method": "upper", "version": ""}
    _doc_args["title"] = {"type": "titlecase", "method": "title", "version": ""}
    _doc_args["capitalize"] = {
        "type": "be capitalized",
        "method": "capitalize",
        "version": "",
    }
    _doc_args["swapcase"] = {
        "type": "be swapcased",
        "method": "swapcase",
        "version": "",
    }
    _doc_args["casefold"] = {
        "type": "be casefolded",
        "method": "casefold",
        "version": "",
    }

    @Appender(_shared_docs["casemethods"] % _doc_args["lower"])
    @forbid_nonstring_types(["bytes"])
    def lower(self):
        result = self._data.array._str_lower()
        return self._wrap_result(result)

    @Appender(_shared_docs["casemethods"] % _doc_args["upper"])
    @forbid_nonstring_types(["bytes"])
    def upper(self):
        result = self._data.array._str_upper()
        return self._wrap_result(result)

    @Appender(_shared_docs["casemethods"] % _doc_args["title"])
    @forbid_nonstring_types(["bytes"])
    def title(self):
        result = self._data.array._str_title()
        return self._wrap_result(result)

    @Appender(_shared_docs["casemethods"] % _doc_args["capitalize"])
    @forbid_nonstring_types(["bytes"])
    def capitalize(self):
        result = self._data.array._str_capitalize()
        return self._wrap_result(result)

    @Appender(_shared_docs["casemethods"] % _doc_args["swapcase"])
    @forbid_nonstring_types(["bytes"])
    def swapcase(self):
        result = self._data.array._str_swapcase()
        return self._wrap_result(result)

    @Appender(_shared_docs["casemethods"] % _doc_args["casefold"])
    @forbid_nonstring_types(["bytes"])
    def casefold(self):
        result = self._data.array._str_casefold()
        return self._wrap_result(result)

    _shared_docs["ismethods"] = """
    Check whether all characters in each string are %(type)s.
    [docstring truncated]
    """
    _shared_docs["isalpha"] = """
    See Also
    --------
    [docstring truncated]
    """
    _shared_docs["isnumeric"] = """
    See Also
    --------
    [docstring truncated]
    """
    _shared_docs["isalnum"] = """
    See Also
    --------
    [docstring truncated]
    """
    _shared_docs["isdecimal"] = """
    See Also
    --------
    [docstring truncated]
    """
    _shared_docs["isdigit"] = """
    See Also
    --------
    [docstring truncated]
    """
    _shared_docs["isspace"] = """
    See Also
    --------
    [docstring truncated]
    """
    _shared_docs["islower"] = """
    See Also
    --------
    [docstring truncated]
    """
    _shared_docs["isupper"] = """
    See Also
    --------
    [docstring truncated]
    """
    _shared_docs["istitle"] = """
    See Also
    --------
    [docstring truncated]
    """
    _shared_docs["isascii"] = """
    See Also
    --------
    [docstring truncated]
    """

    _doc_args["isalnum"] = {"type": "alphanumeric", "method": "isalnum"}
    _doc_args["isalpha"] = {"type": "alphabetic", "method": "isalpha"}
    _doc_args["isdigit"] = {"type": "digits", "method": "isdigit"}
    _doc_args["isspace"] = {"type": "whitespace", "method": "isspace"}
    _doc_args["islower"] = {"type": "lowercase", "method": "islower"}
    _doc_args["isascii"] = {"type": "ascii", "method": "isascii"}
    _doc_args["isupper"] = {"type": "uppercase", "method": "isupper"}
    _doc_args["istitle"] = {"type": "titlecase", "method": "istitle"}
    _doc_args["isnumeric"] = {"type": "numeric", "method": "isnumeric"}
    _doc_args["isdecimal"] = {"type": "decimal", "method": "isdecimal"}

    isalnum = _map_and_wrap(
        "isalnum",
        docstring=_shared_docs["ismethods"] % _doc_args["isalnum"]
        + _shared_docs["isalnum"],
    )
    isalpha = _map_and_wrap(
        "isalpha",
        docstring=_shared_docs["ismethods"] % _doc_args["isalpha"]
        + _shared_docs["isalpha"],
    )
    isdigit = _map_and_wrap(
        "isdigit",
        docstring=_shared_docs["ismethods"] % _doc_args["isdigit"]
        + _shared_docs["isdigit"],
    )
    isspace = _map_and_wrap(
        "isspace",
        docstring=_shared_docs["ismethods"] % _doc_args["isspace"]
        + _shared_docs["isspace"],
    )
    islower = _map_and_wrap(
        "islower",
        docstring=_shared_docs["ismethods"] % _doc_args["islower"]
        + _shared_docs["islower"],
    )
    isascii = _map_and_wrap(
        "isascii",
        docstring=_shared_docs["ismethods"] % _doc_args["isascii"]
        + _shared_docs["isascii"],
    )
    isupper = _map_and_wrap(
        "isupper",
        docstring=_shared_docs["ismethods"] % _doc_args["isupper"]
        + _shared_docs["isupper"],
    )
    istitle = _map_and_wrap(
        "istitle",
        docstring=_shared_docs["ismethods"] % _doc_args["istitle"]
        + _shared_docs["istitle"],
    )
    isnumeric = _map_and_wrap(
        "isnumeric",
        docstring=_shared_docs["ismethods"] % _doc_args["isnumeric"]
        + _shared_docs["isnumeric"],
    )
    isdecimal = _map_and_wrap(
        "isdecimal",
        docstring=_shared_docs["ismethods"] % _doc_args["isdecimal"]
        + _shared_docs["isdecimal"],
    )


def cat_safe(list_of_columns: list[np.ndarray], sep: str) -> np.ndarray:
    """
    Auxiliary function for :meth:`str.cat`.
    [docstring truncated]
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


def cat_core(list_of_columns: list, sep: str) -> np.ndarray:
    """
    Auxiliary function for :meth:`str.cat`
    [docstring truncated]
    """
    if sep == "":
        arr_of_cols = np.asarray(list_of_columns, dtype=object)
        return np.sum(arr_of_cols, axis=0)
    list_with_sep = [sep] * (2 * len(list_of_columns) - 1)
    list_with_sep[::2] = list_of_columns
    arr_with_sep = np.asarray(list_with_sep, dtype=object)
    return np.sum(arr_with_sep, axis=0)


def _result_dtype(arr: Any) -> Any:
    from pandas.core.arrays.string_ import StringDtype
    if isinstance(arr.dtype, (ArrowDtype, StringDtype)):
        return arr.dtype
    return object


def _get_single_group_name(regex: re.Pattern[str]) -> Hashable:
    if regex.groupindex:
        return next(iter(regex.groupindex))
    else:
        return None


def _get_group_names(regex: re.Pattern[str]) -> list[Hashable] | range:
    rng = range(regex.groups)
    names = {v: k for k, v in regex.groupindex.items()}
    if not names:
        return rng
    result: list[Hashable] = [names.get(1 + i, i) for i in rng]
    arr = np.array(result)
    if arr.dtype.kind == "i" and lib.is_range_indexer(arr, len(arr)):
        return rng
    return result


def str_extractall(arr: Union[ABCIndex, ABCSeries], pat: str, flags: int = 0) -> DataFrame:
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