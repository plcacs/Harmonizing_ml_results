from __future__ import annotations
import functools
import re
import textwrap
from typing import Any, Callable, Optional, Union, TYPE_CHECKING
import unicodedata
import warnings
import numpy as np
from pandas._libs import lib
import pandas._libs.missing as libmissing
import pandas._libs.ops as libops
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import pandas_dtype
from pandas.core.dtypes.missing import isna
from pandas.core.strings.base import BaseStringArrayMethods

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pandas._typing import NpDtype, Scalar


class ObjectStringArrayMixin(BaseStringArrayMethods):
    """
    String Methods operating on object-dtype ndarrays.
    """

    def __len__(self) -> int:
        raise NotImplementedError

    def _str_map(
        self,
        f: Callable[[Any], Any],
        na_value: Any = lib.no_default,
        dtype: Optional[Any] = None,
        convert: bool = True,
    ) -> Any:
        """
        Map a callable over valid elements of the array.
        """
        if dtype is None:
            dtype = np.dtype("object")
        if na_value is lib.no_default:
            na_value = self.dtype.na_value
        if not len(self):
            return np.array([], dtype=dtype)
        arr = np.asarray(self, dtype=object)
        mask = isna(arr)
        map_convert = convert and (not np.all(mask))
        try:
            result = lib.map_infer_mask(arr, f, mask.view(np.uint8), convert=map_convert)
        except (TypeError, AttributeError) as err:
            p_err = "((takes)|(missing)) (?(2)from \\d+ to )?\\d+ (?(3)required )positional arguments?"
            if len(err.args) >= 1 and re.search(p_err, err.args[0]):
                raise err

            def g(x: Any) -> Any:
                try:
                    return f(x)
                except (TypeError, AttributeError):
                    return na_value

            return self._str_map(g, na_value=na_value, dtype=dtype)
        if not isinstance(result, np.ndarray):
            return result
        if na_value is not np.nan:
            np.putmask(result, mask, na_value)
            if convert and result.dtype == object:
                result = lib.maybe_convert_objects(result)
        return result

    def _str_count(self, pat: str, flags: int = 0) -> np.ndarray:
        regex = re.compile(pat, flags=flags)
        f: Callable[[Any], int] = lambda x: len(regex.findall(x))
        return self._str_map(f, dtype="int64")

    def _str_pad(self, width: int, side: Literal["left", "right", "both"] = "left", fillchar: str = " ") -> Any:
        if side == "left":
            f: Callable[[Any], str] = lambda x: x.rjust(width, fillchar)
        elif side == "right":
            f = lambda x: x.ljust(width, fillchar)
        elif side == "both":
            f = lambda x: x.center(width, fillchar)
        else:
            raise ValueError("Invalid side")
        return self._str_map(f)

    def _str_contains(
        self,
        pat: str,
        case: bool = True,
        flags: int = 0,
        na: Any = lib.no_default,
        regex: bool = True,
    ) -> Any:
        if regex:
            if not case:
                flags |= re.IGNORECASE
            pat_compiled: re.Pattern = re.compile(pat, flags=flags)
            f: Callable[[Any], bool] = lambda x: pat_compiled.search(x) is not None
        elif case:
            f = lambda x: pat in x
        else:
            upper_pat: str = pat.upper()
            f = lambda x: upper_pat in x.upper()
        if na is not lib.no_default and (not isna(na)) and (not isinstance(na, bool)):
            warnings.warn(
                "Allowing a non-bool 'na' in obj.str.contains is deprecated and will raise in a future version.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )
        return self._str_map(f, na, dtype=np.dtype("bool"))

    def _str_startswith(self, pat: str, na: Any = lib.no_default) -> Any:
        f: Callable[[Any], bool] = lambda x: x.startswith(pat)
        if na is not lib.no_default and (not isna(na)) and (not isinstance(na, bool)):
            warnings.warn(
                "Allowing a non-bool 'na' in obj.str.startswith is deprecated and will raise in a future version.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )
        return self._str_map(f, na_value=na, dtype=np.dtype(bool))

    def _str_endswith(self, pat: str, na: Any = lib.no_default) -> Any:
        f: Callable[[Any], bool] = lambda x: x.endswith(pat)
        if na is not lib.no_default and (not isna(na)) and (not isinstance(na, bool)):
            warnings.warn(
                "Allowing a non-bool 'na' in obj.str.endswith is deprecated and will raise in a future version.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )
        return self._str_map(f, na_value=na, dtype=np.dtype(bool))

    def _str_replace(
        self, pat: str, repl: Union[str, Callable[[Any], Any]], n: int = -1, case: bool = True, flags: int = 0, regex: bool = True
    ) -> Any:
        if case is False:
            flags |= re.IGNORECASE
        if regex or flags or callable(repl):
            if not isinstance(pat, re.Pattern):
                if regex is False:
                    pat = re.escape(pat)
                pat = re.compile(pat, flags=flags)
            n = n if n >= 0 else 0
            f: Callable[[Any], str] = lambda x: pat.sub(repl=repl, string=x, count=n)
        else:
            f = lambda x: x.replace(pat, repl, n)
        return self._str_map(f, dtype=str)

    def _str_repeat(self, repeats: Any) -> Any:
        if lib.is_integer(repeats):
            rint: int = int(repeats)

            def scalar_rep(x: Any) -> Any:
                try:
                    return bytes.__mul__(x, rint)
                except TypeError:
                    return str.__mul__(x, rint)

            return self._str_map(scalar_rep, dtype=str)
        else:
            from pandas.core.arrays.string_ import BaseStringArray

            def rep(x: Any, r: Any) -> Any:
                if x is libmissing.NA:
                    return x
                try:
                    return bytes.__mul__(x, r)
                except TypeError:
                    return str.__mul__(x, r)

            result = libops.vec_binop(np.asarray(self), np.asarray(repeats, dtype=object), rep)
            if not isinstance(self, BaseStringArray):
                return result
            return type(self)._from_sequence(result, dtype=self.dtype)

    def _str_match(self, pat: str, case: bool = True, flags: int = 0, na: Any = lib.no_default) -> Any:
        if not case:
            flags |= re.IGNORECASE
        regex = re.compile(pat, flags=flags)
        f: Callable[[Any], bool] = lambda x: regex.match(x) is not None
        return self._str_map(f, na_value=na, dtype=np.dtype(bool))

    def _str_fullmatch(self, pat: str, case: bool = True, flags: int = 0, na: Any = lib.no_default) -> Any:
        if not case:
            flags |= re.IGNORECASE
        regex = re.compile(pat, flags=flags)
        f: Callable[[Any], bool] = lambda x: regex.fullmatch(x) is not None
        return self._str_map(f, na_value=na, dtype=np.dtype(bool))

    def _str_encode(self, encoding: str, errors: str = "strict") -> Any:
        f: Callable[[Any], bytes] = lambda x: x.encode(encoding, errors=errors)
        return self._str_map(f, dtype=object)

    def _str_find(self, sub: str, start: int = 0, end: Optional[int] = None) -> Any:
        return self._str_find_(sub, start, end, side="left")

    def _str_rfind(self, sub: str, start: int = 0, end: Optional[int] = None) -> Any:
        return self._str_find_(sub, start, end, side="right")

    def _str_find_(self, sub: str, start: int, end: Optional[int], side: str) -> Any:
        if side == "left":
            method: str = "find"
        elif side == "right":
            method = "rfind"
        else:
            raise ValueError("Invalid side")
        if end is None:
            f: Callable[[Any], int] = lambda x: getattr(x, method)(sub, start)
        else:
            f = lambda x: getattr(x, method)(sub, start, end)
        return self._str_map(f, dtype="int64")

    def _str_findall(self, pat: str, flags: int = 0) -> Any:
        regex = re.compile(pat, flags=flags)
        return self._str_map(regex.findall, dtype="object")

    def _str_get(self, i: Any) -> Any:
        def f(x: Any) -> Any:
            if isinstance(x, dict):
                return x.get(i)
            elif len(x) > i >= -len(x):
                return x[i]
            return self.dtype.na_value

        return self._str_map(f)

    def _str_index(self, sub: str, start: int = 0, end: Optional[int] = None) -> Any:
        if end:
            f: Callable[[Any], int] = lambda x: x.index(sub, start, end)
        else:
            f = lambda x: x.index(sub, start, end)
        return self._str_map(f, dtype="int64")

    def _str_rindex(self, sub: str, start: int = 0, end: Optional[int] = None) -> Any:
        if end:
            f: Callable[[Any], int] = lambda x: x.rindex(sub, start, end)
        else:
            f = lambda x: x.rindex(sub, start, end)
        return self._str_map(f, dtype="int64")

    def _str_join(self, sep: str) -> Any:
        return self._str_map(sep.join)

    def _str_partition(self, sep: str, expand: bool) -> Any:
        result = self._str_map(lambda x: x.partition(sep), dtype="object")
        return result

    def _str_rpartition(self, sep: str, expand: bool) -> Any:
        return self._str_map(lambda x: x.rpartition(sep), dtype="object")

    def _str_len(self) -> Any:
        return self._str_map(len, dtype="int64")

    def _str_slice(self, start: Optional[int] = None, stop: Optional[int] = None, step: Optional[int] = None) -> Any:
        obj = slice(start, stop, step)
        return self._str_map(lambda x: x[obj])

    def _str_slice_replace(
        self, start: Optional[int] = None, stop: Optional[int] = None, repl: Optional[str] = None
    ) -> Any:
        if repl is None:
            repl = ""

        def f(x: str) -> str:
            if x[start:stop] == "":
                local_stop = start  # type: ignore
            else:
                local_stop = stop
            y = ""
            if start is not None:
                y += x[:start]
            y += repl
            if stop is not None:
                y += x[local_stop:]
            return y

        return self._str_map(f)

    def _str_split(self, pat: Optional[str] = None, n: int = -1, expand: bool = False, regex: Optional[bool] = None) -> Any:
        if pat is None:
            if n is None or n == 0:
                n = -1
            f: Callable[[Any], Any] = lambda x: x.split(pat, n)
        else:
            if regex is True or isinstance(pat, re.Pattern):
                new_pat: Union[re.Pattern, str] = re.compile(pat)
            elif regex is False:
                new_pat = pat
            elif len(pat) == 1:
                new_pat = pat
            else:
                new_pat = re.compile(pat)
            if isinstance(new_pat, re.Pattern):
                if n is None or n == -1:
                    n = 0
                f = lambda x: new_pat.split(x, maxsplit=n)
            else:
                if n is None or n == 0:
                    n = -1
                f = lambda x: x.split(pat, n)
        return self._str_map(f, dtype="object")

    def _str_rsplit(self, pat: Optional[str] = None, n: int = -1) -> Any:
        if n is None or n == 0:
            n = -1
        f: Callable[[Any], Any] = lambda x: x.rsplit(pat, n)
        return self._str_map(f, dtype="object")

    def _str_translate(self, table: dict) -> Any:
        return self._str_map(lambda x: x.translate(table))

    def _str_wrap(self, width: int, **kwargs: Any) -> Any:
        kwargs["width"] = width
        tw = textwrap.TextWrapper(**kwargs)
        return self._str_map(lambda s: "\n".join(tw.wrap(s)))

    def _str_get_dummies(self, sep: str = "|", dtype: Optional[Any] = None) -> tuple[Any, list]:
        from pandas import Series
        if dtype is None:
            dtype = np.int64
        arr = Series(self).fillna("")
        try:
            arr = sep + arr + sep
        except (TypeError, NotImplementedError):
            arr = sep + arr.astype(str) + sep
        tags: set[str] = set()
        for ts in Series(arr, copy=False).str.split(sep):
            tags.update(ts)
        tags2 = sorted(tags - {""})
        _dtype = pandas_dtype(dtype)
        if isinstance(_dtype, np.dtype):
            dummies_dtype = _dtype
        else:
            dummies_dtype = np.bool_
        dummies = np.empty((len(arr), len(tags2)), dtype=dummies_dtype, order="F")

        def _isin(test_elements: Any, element: Any) -> bool:
            return element in test_elements

        for i, t in enumerate(tags2):
            pat = sep + t + sep
            dummies[:, i] = lib.map_infer(arr.to_numpy(), functools.partial(_isin, element=pat))
        return (dummies, tags2)

    def _str_upper(self) -> Any:
        return self._str_map(lambda x: x.upper())

    def _str_isalnum(self) -> Any:
        return self._str_map(str.isalnum, dtype="bool")

    def _str_isalpha(self) -> Any:
        return self._str_map(str.isalpha, dtype="bool")

    def _str_isascii(self) -> Any:
        return self._str_map(str.isascii, dtype="bool")

    def _str_isdecimal(self) -> Any:
        return self._str_map(str.isdecimal, dtype="bool")

    def _str_isdigit(self) -> Any:
        return self._str_map(str.isdigit, dtype="bool")

    def _str_islower(self) -> Any:
        return self._str_map(str.islower, dtype="bool")

    def _str_isnumeric(self) -> Any:
        return self._str_map(str.isnumeric, dtype="bool")

    def _str_isspace(self) -> Any:
        return self._str_map(str.isspace, dtype="bool")

    def _str_istitle(self) -> Any:
        return self._str_map(str.istitle, dtype="bool")

    def _str_isupper(self) -> Any:
        return self._str_map(str.isupper, dtype="bool")

    def _str_capitalize(self) -> Any:
        return self._str_map(str.capitalize)

    def _str_casefold(self) -> Any:
        return self._str_map(str.casefold)

    def _str_title(self) -> Any:
        return self._str_map(str.title)

    def _str_swapcase(self) -> Any:
        return self._str_map(str.swapcase)

    def _str_lower(self) -> Any:
        return self._str_map(str.lower)

    def _str_normalize(self, form: str) -> Any:
        f: Callable[[str], str] = lambda x: unicodedata.normalize(form, x)
        return self._str_map(f)

    def _str_strip(self, to_strip: Optional[str] = None) -> Any:
        return self._str_map(lambda x: x.strip(to_strip))

    def _str_lstrip(self, to_strip: Optional[str] = None) -> Any:
        return self._str_map(lambda x: x.lstrip(to_strip))

    def _str_rstrip(self, to_strip: Optional[str] = None) -> Any:
        return self._str_map(lambda x: x.rstrip(to_strip))

    def _str_removeprefix(self, prefix: str) -> Any:
        return self._str_map(lambda x: x.removeprefix(prefix))

    def _str_removesuffix(self, suffix: str) -> Any:
        return self._str_map(lambda x: x.removesuffix(suffix))

    def _str_extract(self, pat: str, flags: int = 0, expand: bool = True) -> Any:
        regex = re.compile(pat, flags=flags)
        na_value = self.dtype.na_value
        if not expand:

            def g(x: Any) -> Any:
                m = regex.search(x)
                return m.groups()[0] if m else na_value

            return self._str_map(g, convert=False)
        empty_row = [na_value] * regex.groups

        def f(x: Any) -> list:
            if not isinstance(x, str):
                return empty_row
            m = regex.search(x)
            if m:
                return [na_value if item is None else item for item in m.groups()]
            else:
                return empty_row

        return [f(val) for val in np.asarray(self)]
