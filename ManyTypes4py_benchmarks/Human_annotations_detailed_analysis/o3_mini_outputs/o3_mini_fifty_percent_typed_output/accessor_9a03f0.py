from __future__ import annotations

import codecs
import re
import warnings
from functools import wraps
from typing import (
    Any,
    Callable,
    Hashable,
    Iterator,
    Literal,
    Union,
)

import numpy as np
import numpy.typing as npt

from pandas._config import get_option
from pandas._libs import lib
from pandas._typing import DtypeObj, F, Scalar
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
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCIndex,
    ABCMultiIndex,
    ABCSeries,
)
from pandas.core.dtypes.missing import isna

from pandas.core.arrays import ExtensionArray
from pandas.core.base import NoNewAttributesMixin
from pandas.core.construction import extract_array


class StringMethods(NoNewAttributesMixin):
    """
    Vectorized string functions for Series and Index.

    NAs stay NA unless handled otherwise by a particular method.
    Patterned after Python's string methods, with some inspiration from
    R's stringr package.

    Parameters
    ----------
    data : Series or Index
        The content of the Series or Index.

    See Also
    --------
    Series.str : Vectorized string functions for Series.
    Index.str : Vectorized string functions for Index.
    """

    def __init__(self, data: Union[ABCSeries, ABCIndex]) -> None:
        from pandas.core.arrays.string_ import StringDtype

        self._inferred_dtype: str = self._validate(data)
        self._is_categorical: bool = isinstance(data.dtype, CategoricalDtype)
        self._is_string: bool = isinstance(data.dtype, StringDtype)
        self._data = data

        self._index = self._name = None
        if isinstance(data, ABCSeries):
            self._index = data.index
            self._name = data.name

        # ._values.categories works for both Series/Index
        self._parent = data._values.categories if self._is_categorical else data
        # save orig to blow up categoricals to the right type
        self._orig = data
        self._freeze()

    @staticmethod
    def _validate(data: Any) -> str:
        """
        Auxiliary function for StringMethods, infers and checks dtype of data.
        """
        if isinstance(data, ABCMultiIndex):
            raise AttributeError(
                "Can only use .str accessor with Index, not MultiIndex"
            )
        allowed_types = ["string", "empty", "bytes", "mixed", "mixed-integer"]
        data_array = extract_array(data)
        values = getattr(data_array, "categories", data_array)
        inferred_dtype = lib.infer_dtype(values, skipna=True)
        if inferred_dtype not in allowed_types:
            raise AttributeError(
                f"Can only use .str accessor with string values, not {inferred_dtype}"
            )
        return inferred_dtype

    def __getitem__(self, key: Any) -> Any:
        result = self._data.array._str_getitem(key)
        return self._wrap_result(result)

    def __iter__(self) -> Iterator[Any]:
        raise TypeError(f"'{type(self).__name__}' object is not iterable")

    def _wrap_result(
        self,
        result: Any,
        name: Any = None,
        expand: Any = None,
        fill_value: Any = np.nan,
        returns_string: bool = True,
        dtype: Any = None,
    ) -> Any:
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
                        values_np = result.to_numpy()
                        new_values = []
                        for row in values_np:
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

                def cons_row(x: Any) -> list[Any]:
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

    def _get_series_list(self, others: Any) -> list[ABCSeries]:
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
                    los: list[ABCSeries] = []
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
        others: Any = None,
        sep: Union[str, None] = None,
        na_rep: Any = None,
        join: Literal["left", "right", "outer", "inner"] = "left",
    ) -> Union[str, ABCSeries, ABCIndex]:
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
            all_cols = [np.where(nm, na_rep, col) for nm, col in zip(na_masks, all_cols)]
            result = cat_safe(all_cols, sep)
        else:
            result = cat_safe(all_cols, sep)

        if isinstance(self._orig.dtype, CategoricalDtype):
            dtype = self._orig.dtype.categories.dtype
        else:
            dtype = self._orig.dtype
        if isinstance(self._orig, ABCIndex):
            if isna(result).all():
                dtype = object  # type: ignore[assignment]
            out = Index(result, dtype=dtype, name=self._orig.name)
        else:
            res_ser = Series(
                result, dtype=dtype, index=data.index, name=self._orig.name, copy=False
            )
            out = res_ser.__finalize__(self._orig, method="str_cat")
        return out

    _shared_docs: dict[str, str] = {}
    _shared_docs["str_split"] = r"""
    Split strings around given separator/delimiter.
    ...
    """

    @Appender(
        _shared_docs["str_split"]  # noqa: E501
    )
    @forbid_nonstring_types(["bytes"])
    def split(
        self,
        pat: Union[str, re.Pattern, None] = None,
        *,
        n: int = -1,
        expand: bool = False,
        regex: Any = None,
    ) -> Any:
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

    @Appender(
        _shared_docs["str_split"]  # noqa: E501
    )
    @forbid_nonstring_types(["bytes"])
    def rsplit(self, pat: Any = None, *, n: int = -1, expand: bool = False) -> Any:
        result = self._data.array._str_rsplit(pat, n=n)
        dtype = object if self._data.dtype == object else None
        return self._wrap_result(result, expand=expand, returns_string=expand, dtype=dtype)

    _shared_docs["str_partition"] = """
    Split the string at the %(side)s occurrence of `sep`.
    ...
    """

    @Appender(
        _shared_docs["str_partition"] % {
            "side": "first",
            "return": "3 elements containing the string itself, followed by two empty strings",
            "also": "rpartition : Split the string at the last occurrence of `sep`.",
        }
    )
    @forbid_nonstring_types(["bytes"])
    def partition(self, sep: str = " ", expand: bool = True) -> Any:
        result = self._data.array._str_partition(sep, expand)
        if self._data.dtype == "category":
            dtype = self._data.dtype.categories.dtype
        else:
            dtype = object if self._data.dtype == object else None
        return self._wrap_result(result, expand=expand, returns_string=expand, dtype=dtype)

    @Appender(
        _shared_docs["str_partition"] % {
            "side": "last",
            "return": "3 elements containing two empty strings, followed by the string itself",
            "also": "partition : Split the string at the first occurrence of `sep`.",
        }
    )
    @forbid_nonstring_types(["bytes"])
    def rpartition(self, sep: str = " ", expand: bool = True) -> Any:
        result = self._data.array._str_rpartition(sep, expand)
        if self._data.dtype == "category":
            dtype = self._data.dtype.categories.dtype
        else:
            dtype = object if self._data.dtype == object else None
        return self._wrap_result(result, expand=expand, returns_string=expand, dtype=dtype)

    def get(self, i: Any) -> Any:
        result = self._data.array._str_get(i)
        return self._wrap_result(result)

    @forbid_nonstring_types(["bytes"])
    def join(self, sep: str) -> Any:
        result = self._data.array._str_join(sep)
        return self._wrap_result(result)

    @forbid_nonstring_types(["bytes"])
    def contains(
        self,
        pat: Any,
        case: bool = True,
        flags: int = 0,
        na: Any = lib.no_default,
        regex: bool = True,
    ) -> Any:
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
    def match(self, pat: str, case: bool = True, flags: int = 0, na: Any = lib.no_default) -> Any:
        result = self._data.array._str_match(pat, case=case, flags=flags, na=na)
        return self._wrap_result(result, fill_value=na, returns_string=False)

    @forbid_nonstring_types(["bytes"])
    def fullmatch(self, pat: str, case: bool = True, flags: int = 0, na: Any = lib.no_default) -> Any:
        result = self._data.array._str_fullmatch(pat, case=case, flags=flags, na=na)
        return self._wrap_result(result, fill_value=na, returns_string=False)

    @forbid_nonstring_types(["bytes"])
    def replace(
        self,
        pat: Any,
        repl: Any = None,
        n: int = -1,
        case: Union[bool, None] = None,
        flags: int = 0,
        regex: bool = False,
    ) -> Any:
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
            result = res_output.array._str_replace(key, value, n=n, case=case, flags=flags, regex=regex)
            res_output = self._wrap_result(result)
        return res_output

    @forbid_nonstring_types(["bytes"])
    def repeat(self, repeats: Any) -> Any:
        result = self._data.array._str_repeat(repeats)
        return self._wrap_result(result)

    @forbid_nonstring_types(["bytes"])
    def pad(
        self,
        width: int,
        side: Literal["left", "right", "both"] = "left",
        fillchar: str = " ",
    ) -> Any:
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

    @Appender(_shared_docs["str_pad"] % {"side": "left and right", "method": "center"})
    @forbid_nonstring_types(["bytes"])
    def center(self, width: int, fillchar: str = " ") -> Any:
        return self.pad(width, side="both", fillchar=fillchar)

    @Appender(_shared_docs["str_pad"] % {"side": "right", "method": "ljust"})
    @forbid_nonstring_types(["bytes"])
    def ljust(self, width: int, fillchar: str = " ") -> Any:
        return self.pad(width, side="right", fillchar=fillchar)

    @Appender(_shared_docs["str_pad"] % {"side": "left", "method": "rjust"})
    @forbid_nonstring_types(["bytes"])
    def rjust(self, width: int, fillchar: str = " ") -> Any:
        return self.pad(width, side="left", fillchar=fillchar)

    @forbid_nonstring_types(["bytes"])
    def zfill(self, width: int) -> Any:
        if not is_integer(width):
            msg = f"width must be of integer type, not {type(width).__name__}"
            raise TypeError(msg)
        f = lambda x: x.zfill(width)
        result = self._data.array._str_map(f)
        return self._wrap_result(result)

    def slice(self, start: Any = None, stop: Any = None, step: Any = None) -> Any:
        result = self._data.array._str_slice(start, stop, step)
        return self._wrap_result(result)

    @forbid_nonstring_types(["bytes"])
    def slice_replace(self, start: Any = None, stop: Any = None, repl: Any = None) -> Any:
        result = self._data.array._str_slice_replace(start, stop, repl)
        return self._wrap_result(result)

    def decode(self, encoding: str, errors: str = "strict") -> Any:
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
    def encode(self, encoding: str, errors: str = "strict") -> Any:
        result = self._data.array._str_encode(encoding, errors)
        return self._wrap_result(result, returns_string=False)

    @Appender(_shared_docs["str_strip"] % {"side": "left and right", "method": "strip", "position": "leading and trailing"})
    @forbid_nonstring_types(["bytes"])
    def strip(self, to_strip: Any = None) -> Any:
        result = self._data.array._str_strip(to_strip)
        return self._wrap_result(result)

    @Appender(_shared_docs["str_strip"] % {"side": "left side", "method": "lstrip", "position": "leading"})
    @forbid_nonstring_types(["bytes"])
    def lstrip(self, to_strip: Any = None) -> Any:
        result = self._data.array._str_lstrip(to_strip)
        return self._wrap_result(result)

    @Appender(_shared_docs["str_strip"] % {"side": "right side", "method": "rstrip", "position": "trailing"})
    @forbid_nonstring_types(["bytes"])
    def rstrip(self, to_strip: Any = None) -> Any:
        result = self._data.array._str_rstrip(to_strip)
        return self._wrap_result(result)

    @Appender(_shared_docs["str_removefix"] % {"side": "prefix", "other_side": "suffix"})
    @forbid_nonstring_types(["bytes"])
    def removeprefix(self, prefix: str) -> Any:
        result = self._data.array._str_removeprefix(prefix)
        return self._wrap_result(result)

    @Appender(_shared_docs["str_removefix"] % {"side": "suffix", "other_side": "prefix"})
    @forbid_nonstring_types(["bytes"])
    def removesuffix(self, suffix: Any) -> Any:
        result = self._data.array._str_removesuffix(suffix)
        return self._wrap_result(result)

    @forbid_nonstring_types(["bytes"])
    def wrap(
        self,
        width: int,
        expand_tabs: bool = True,
        tabsize: int = 8,
        replace_whitespace: bool = True,
        drop_whitespace: bool = True,
        initial_indent: str = "",
        subsequent_indent: str = "",
        fix_sentence_endings: bool = False,
        break_long_words: bool = True,
        break_on_hyphens: bool = True,
        max_lines: Any = None,
        placeholder: Any = " [...]",
    ) -> Any:
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
        dtype: Any = None,
    ) -> Any:
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
    def translate(self, table: dict[int, Union[int, str, None]]) -> Any:
        result = self._data.array._str_translate(table)
        dtype = object if self._data.dtype == "object" else None
        return self._wrap_result(result, dtype=dtype)

    @forbid_nonstring_types(["bytes"])
    def count(self, pat: str, flags: int = 0) -> Any:
        result = self._data.array._str_count(pat, flags)
        return self._wrap_result(result, returns_string=False)

    @forbid_nonstring_types(["bytes"])
    def startswith(
        self, pat: Union[str, tuple[str, ...]], na: Any = lib.no_default
    ) -> Union[ABCSeries, ABCIndex]:
        if not isinstance(pat, (str, tuple)):
            msg = f"expected a string or tuple, not {type(pat).__name__}"
            raise TypeError(msg)
        result = self._data.array._str_startswith(pat, na=na)
        return self._wrap_result(result, returns_string=False)

    @forbid_nonstring_types(["bytes"])
    def endswith(
        self, pat: Union[str, tuple[str, ...]], na: Any = lib.no_default
    ) -> Union[ABCSeries, ABCIndex]:
        if not isinstance(pat, (str, tuple)):
            msg = f"expected a string or tuple, not {type(pat).__name__}"
            raise TypeError(msg)
        result = self._data.array._str_endswith(pat, na=na)
        return self._wrap_result(result, returns_string=False)

    @forbid_nonstring_types(["bytes"])
    def findall(self, pat: str, flags: int = 0) -> Any:
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
                result_index: Union[ABCIndex, None]
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
    def extractall(self, pat: str, flags: int = 0) -> DataFrame:
        return str_extractall(self._orig, pat, flags)

    _shared_docs["find"] = """
    Return %(side)s indexes in each strings in the Series/Index.
    ...
    """

    @Appender(
        _shared_docs["find"] % {
            "side": "lowest",
            "method": "find",
            "also": "rfind : Return highest indexes in each strings.",
        }
    )
    @forbid_nonstring_types(["bytes"])
    def find(self, sub: str, start: int = 0, end: Any = None) -> Any:
        if not isinstance(sub, str):
            msg = f"expected a string object, not {type(sub).__name__}"
            raise TypeError(msg)
        result = self._data.array._str_find(sub, start, end)
        return self._wrap_result(result, returns_string=False)

    @Appender(
        _shared_docs["find"] % {
            "side": "highest",
            "method": "rfind",
            "also": "find : Return lowest indexes in each strings.",
        }
    )
    @forbid_nonstring_types(["bytes"])
    def rfind(self, sub: str, start: int = 0, end: Any = None) -> Any:
        if not isinstance(sub, str):
            msg = f"expected a string object, not {type(sub).__name__}"
            raise TypeError(msg)
        result = self._data.array._str_rfind(sub, start=start, end=end)
        return self._wrap_result(result, returns_string=False)

    @forbid_nonstring_types(["bytes"])
    def normalize(self, form: str) -> Any:
        result = self._data.array._str_normalize(form)
        return self._wrap_result(result)

    _shared_docs["index"] = """
    Return %(side)s indexes in each string in Series/Index.
    ...
    """

    @Appender(
        _shared_docs["index"] % {
            "side": "lowest",
            "similar": "find",
            "method": "index",
            "also": "rindex : Return highest indexes in each strings.",
        }
    )
    @forbid_nonstring_types(["bytes"])
    def index(self, sub: str, start: int = 0, end: Any = None) -> Any:
        if not isinstance(sub, str):
            msg = f"expected a string object, not {type(sub).__name__}"
            raise TypeError(msg)
        result = self._data.array._str_index(sub, start=start, end=end)
        return self._wrap_result(result, returns_string=False)

    @Appender(
        _shared_docs["index"] % {
            "side": "highest",
            "similar": "rfind",
            "method": "rindex",
            "also": "index : Return lowest indexes in each strings.",
        }
    )
    @forbid_nonstring_types(["bytes"])
    def rindex(self, sub: str, start: int = 0, end: Any = None) -> Any:
        if not isinstance(sub, str):
            msg = f"expected a string object, not {type(sub).__name__}"
            raise TypeError(msg)
        result = self._data.array._str_rindex(sub, start=start, end=end)
        return self._wrap_result(result, returns_string=False)

    def len(self) -> Any:
        result = self._data.array._str_len()
        return self._wrap_result(result, returns_string=False)

    _doc_args: dict[str, dict[str, str]] = {}
    _doc_args["lower"] = {"type": "lowercase", "method": "lower", "version": ""}
    _doc_args["upper"] = {"type": "uppercase", "method": "upper", "version": ""}
    _doc_args["title"] = {"type": "titlecase", "method": "title", "version": ""}
    _doc_args["capitalize"] = {"type": "be capitalized", "method": "capitalize", "version": ""}
    _doc_args["swapcase"] = {"type": "be swapcased", "method": "swapcase", "version": ""}
    _doc_args["casefold"] = {"type": "be casefolded", "method": "casefold", "version": ""}

    @Appender(_shared_docs["casemethods"] % _doc_args["lower"])
    @forbid_nonstring_types(["bytes"])
    def lower(self) -> Any:
        result = self._data.array._str_lower()
        return self._wrap_result(result)

    @Appender(_shared_docs["casemethods"] % _doc_args["upper"])
    @forbid_nonstring_types(["bytes"])
    def upper(self) -> Any:
        result = self._data.array._str_upper()
        return self._wrap_result(result)

    @Appender(_shared_docs["casemethods"] % _doc_args["title"])
    @forbid_nonstring_types(["bytes"])
    def title(self) -> Any:
        result = self._data.array._str_title()
        return self._wrap_result(result)

    @Appender(_shared_docs["casemethods"] % _doc_args["capitalize"])
    @forbid_nonstring_types(["bytes"])
    def capitalize(self) -> Any:
        result = self._data.array._str_capitalize()
        return self._wrap_result(result)

    @Appender(_shared_docs["casemethods"] % _doc_args["swapcase"])
    @forbid_nonstring_types(["bytes"])
    def swapcase(self) -> Any:
        result = self._data.array._str_swapcase()
        return self._wrap_result(result)

    @Appender(_shared_docs["casemethods"] % _doc_args["casefold"])
    @forbid_nonstring_types(["bytes"])
    def casefold(self) -> Any:
        result = self._data.array._str_casefold()
        return self._wrap_result(result)


def cat_safe(list_of_columns: list[npt.NDArray[np.object_]], sep: str) -> npt.NDArray[np.object_]:
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


def cat_core(list_of_columns: list[npt.NDArray[np.object_]], sep: str) -> npt.NDArray[np.object_]:
    if sep == "":
        arr_of_cols = np.asarray(list_of_columns, dtype=object)
        return np.sum(arr_of_cols, axis=0)
    list_with_sep: list[Any] = [sep] * (2 * len(list_of_columns) - 1)
    list_with_sep[::2] = list_of_columns
    arr_with_sep = np.asarray(list_with_sep, dtype=object)
    return np.sum(arr_with_sep, axis=0)


def _result_dtype(arr: Any) -> DtypeObj:
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
    regex: re.Pattern = re.compile(pat, flags=flags)
    if regex.groups == 0:
        raise ValueError("pattern contains no capture groups")
    if isinstance(arr, ABCIndex):
        arr = arr.to_series().reset_index(drop=True).astype(arr.dtype)
    columns = _get_group_names(regex)
    match_list: list[list[Any]] = []
    index_list: list[tuple[Any, ...]] = []
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

_cpython_optimized_decoders = ("ascii", "utf-8", "latin-1")  # placeholder for actual decoders
# End of fully annotated code.
