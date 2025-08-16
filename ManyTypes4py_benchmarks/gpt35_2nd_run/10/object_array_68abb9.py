from __future__ import annotations
import functools
import re
import textwrap
from typing import TYPE_CHECKING, Literal, cast
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
    from collections.abc import Callable, Sequence
    from pandas._typing import NpDtype, Scalar

class ObjectStringArrayMixin(BaseStringArrayMethods):
    """
    String Methods operating on object-dtype ndarrays.
    """

    def __len__(self) -> int:
        raise NotImplementedError

    def _str_map(self, f: Callable, na_value: Scalar = lib.no_default, dtype: NpDtype = None, convert: bool = True) -> np.ndarray:
        ...

    def _str_count(self, pat: str, flags: int = 0) -> np.ndarray:
        ...

    def _str_pad(self, width: int, side: Literal['left', 'right', 'both'] = 'left', fillchar: str = ' ') -> np.ndarray:
        ...

    def _str_contains(self, pat: str, case: bool = True, flags: int = 0, na: Scalar = lib.no_default, regex: bool = True) -> np.ndarray:
        ...

    def _str_startswith(self, pat: str, na: Scalar = lib.no_default) -> np.ndarray:
        ...

    def _str_endswith(self, pat: str, na: Scalar = lib.no_default) -> np.ndarray:
        ...

    def _str_replace(self, pat, repl, n: int = -1, case: bool = True, flags: int = 0, regex: bool = True) -> np.ndarray:
        ...

    def _str_repeat(self, repeats) -> np.ndarray:
        ...

    def _str_match(self, pat: str, case: bool = True, flags: int = 0, na: Scalar = lib.no_default) -> np.ndarray:
        ...

    def _str_fullmatch(self, pat: str, case: bool = True, flags: int = 0, na: Scalar = lib.no_default) -> np.ndarray:
        ...

    def _str_encode(self, encoding: str, errors: str = 'strict') -> np.ndarray:
        ...

    def _str_find(self, sub: str, start: int = 0, end: int = None) -> np.ndarray:
        ...

    def _str_rfind(self, sub: str, start: int = 0, end: int = None) -> np.ndarray:
        ...

    def _str_find_(self, sub: str, start: int, end: int, side: Literal['left', 'right']) -> np.ndarray:
        ...

    def _str_findall(self, pat: str, flags: int = 0) -> np.ndarray:
        ...

    def _str_get(self, i: int) -> np.ndarray:
        ...

    def _str_index(self, sub: str, start: int = 0, end: int = None) -> np.ndarray:
        ...

    def _str_rindex(self, sub: str, start: int = 0, end: int = None) -> np.ndarray:
        ...

    def _str_join(self, sep: str) -> np.ndarray:
        ...

    def _str_partition(self, sep: str, expand: bool) -> np.ndarray:
        ...

    def _str_rpartition(self, sep: str, expand: bool) -> np.ndarray:
        ...

    def _str_len(self) -> np.ndarray:
        ...

    def _str_slice(self, start: int = None, stop: int = None, step: int = None) -> np.ndarray:
        ...

    def _str_slice_replace(self, start: int = None, stop: int = None, repl: str = None) -> np.ndarray:
        ...

    def _str_split(self, pat: str = None, n: int = -1, expand: bool = False, regex: bool = None) -> np.ndarray:
        ...

    def _str_rsplit(self, pat: str = None, n: int = -1) -> np.ndarray:
        ...

    def _str_translate(self, table) -> np.ndarray:
        ...

    def _str_wrap(self, width: int, **kwargs) -> np.ndarray:
        ...

    def _str_get_dummies(self, sep: str = '|', dtype: NpDtype = None) -> tuple[np.ndarray, list[str]]:
        ...

    def _str_upper(self) -> np.ndarray:
        ...

    def _str_isalnum(self) -> np.ndarray:
        ...

    def _str_isalpha(self) -> np.ndarray:
        ...

    def _str_isascii(self) -> np.ndarray:
        ...

    def _str_isdecimal(self) -> np.ndarray:
        ...

    def _str_isdigit(self) -> np.ndarray:
        ...

    def _str_islower(self) -> np.ndarray:
        ...

    def _str_isnumeric(self) -> np.ndarray:
        ...

    def _str_isspace(self) -> np.ndarray:
        ...

    def _str_istitle(self) -> np.ndarray:
        ...

    def _str_isupper(self) -> np.ndarray:
        ...

    def _str_capitalize(self) -> np.ndarray:
        ...

    def _str_casefold(self) -> np.ndarray:
        ...

    def _str_title(self) -> np.ndarray:
        ...

    def _str_swapcase(self) -> np.ndarray:
        ...

    def _str_lower(self) -> np.ndarray:
        ...

    def _str_normalize(self, form: str) -> np.ndarray:
        ...

    def _str_strip(self, to_strip: str = None) -> np.ndarray:
        ...

    def _str_lstrip(self, to_strip: str = None) -> np.ndarray:
        ...

    def _str_rstrip(self, to_strip: str = None) -> np.ndarray:
        ...

    def _str_removeprefix(self, prefix: str) -> np.ndarray:
        ...

    def _str_removesuffix(self, suffix: str) -> np.ndarray:
        ...

    def _str_extract(self, pat: str, flags: int = 0, expand: bool = True) -> np.ndarray:
        ...
