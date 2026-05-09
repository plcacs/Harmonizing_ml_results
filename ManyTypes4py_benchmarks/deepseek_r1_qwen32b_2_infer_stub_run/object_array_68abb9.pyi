from __future__ import annotations
import re
from typing import (
    Callable,
    Sequence,
    Union,
    List,
    Optional,
    Tuple,
    Any,
    Dict,
    Set,
    FrozenSet,
    TypeVar,
    overload,
)
from collections.abc import Iterable, Iterator
import numpy as np
from pandas._typing import NpDtype, Scalar
from pandas.core.dtypes.missing import isna

class ObjectStringArrayMixin(BaseStringArrayMethods):
    def __len__(self) -> int:
        ...

    def _str_map(
        self,
        f: Callable,
        na_value: Scalar = lib.no_default,
        dtype: NpDtype = None,
        convert: bool = True,
    ) -> np.ndarray:
        ...

    def _str_count(self, pat: str, flags: int = 0) -> np.ndarray[np.int64]:
        ...

    def _str_pad(
        self,
        width: int,
        side: Literal['left', 'right', 'both'] = 'left',
        fillchar: str = ' ',
    ) -> np.ndarray[object]:
        ...

    def _str_contains(
        self,
        pat: str,
        case: bool = True,
        flags: int = 0,
        na: Scalar = lib.no_default,
        regex: bool = True,
    ) -> np.ndarray[np.bool_]:
        ...

    def _str_startswith(self, pat: str, na: Scalar = lib.no_default) -> np.ndarray[np.bool_]:
        ...

    def _str_endswith(self, pat: str, na: Scalar = lib.no_default) -> np.ndarray[np.bool_]:
        ...

    def _str_replace(
        self,
        pat: Union[str, re.Pattern],
        repl: Union[str, Callable],
        n: int = -1,
        case: bool = True,
        flags: int = 0,
        regex: bool = True,
    ) -> np.ndarray[object]:
        ...

    def _str_repeat(self, repeats: Union[int, np.ndarray]) -> Union[np.ndarray[object], Any]:
        ...

    def _str_match(self, pat: str, case: bool = True, flags: int = 0, na: Scalar = lib.no_default) -> np.ndarray[np.bool_]:
        ...

    def _str_fullmatch(self, pat: str, case: bool = True, flags: int = 0, na: Scalar = lib.no_default) -> np.ndarray[np.bool_]:
        ...

    def _str_encode(self, encoding: str, errors: str = 'strict') -> np.ndarray[object]:
        ...

    def _str_find(self, sub: str, start: int = 0, end: Optional[int] = None) -> np.ndarray[np.int64]:
        ...

    def _str_rfind(self, sub: str, start: int = 0, end: Optional[int] = None) -> np.ndarray[np.int64]:
        ...

    def _str_findall(self, pat: Union[str, re.Pattern], flags: int = 0) -> np.ndarray[object]:
        ...

    def _str_get(self, i: Union[int, str]) -> np.ndarray[object]:
        ...

    def _str_index(self, sub: str, start: int = 0, end: Optional[int] = None) -> np.ndarray[np.int64]:
        ...

    def _str_rindex(self, sub: str, start: int = 0, end: Optional[int] = None) -> np.ndarray[np.int64]:
        ...

    def _str_join(self, sep: str) -> np.ndarray[object]:
        ...

    def _str_partition(self, sep: str, expand: bool) -> np.ndarray[object]:
        ...

    def _str_rpartition(self, sep: str, expand: bool) -> np.ndarray[object]:
        ...

    def _str_len(self) -> np.ndarray[np.int64]:
        ...

    def _str_slice(
        self,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: Optional[int] = None,
    ) -> np.ndarray[object]:
        ...

    def _str_slice_replace(
        self,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        repl: str = '',
    ) -> np.ndarray[object]:
        ...

    def _str_split(
        self,
        pat: Optional[Union[str, re.Pattern]] = None,
        n: int = -1,
        expand: bool = False,
        regex: Optional[bool] = None,
    ) -> np.ndarray[object]:
        ...

    def _str_rsplit(self, pat: Optional[str] = None, n: int = -1) -> np.ndarray[object]:
        ...

    def _str_translate(self, table: Any) -> np.ndarray[object]:
        ...

    def _str_wrap(self, width: int, **kwargs: Any) -> np.ndarray[object]:
        ...

    def _str_get_dummies(self, sep: str = '|', dtype: Optional[Any] = None) -> Tuple[np.ndarray, List[str]]:
        ...

    def _str_upper(self) -> np.ndarray[object]:
        ...

    def _str_isalnum(self) -> np.ndarray[np.bool_]:
        ...

    def _str_isalpha(self) -> np.ndarray[np.bool_]:
        ...

    def _str_isascii(self) -> np.ndarray[np.bool_]:
        ...

    def _str_isdecimal(self) -> np.ndarray[np.bool_]:
        ...

    def _str_isdigit(self) -> np.ndarray[np.bool_]:
        ...

    def _str_islower(self) -> np.ndarray[np.bool_]:
        ...

    def _str_isnumeric(self) -> np.ndarray[np.bool_]:
        ...

    def _str_isspace(self) -> np.ndarray[np.bool_]:
        ...

    def _str_istitle(self) -> np.ndarray[np.bool_]:
        ...

    def _str_isupper(self) -> np.ndarray[np.bool_]:
        ...

    def _str_capitalize(self) -> np.ndarray[object]:
        ...

    def _str_casefold(self) -> np.ndarray[object]:
        ...

    def _str_title(self) -> np.ndarray[object]:
        ...

    def _str_swapcase(self) -> np.ndarray[object]:
        ...

    def _str_lower(self) -> np.ndarray[object]:
        ...

    def _str_normalize(self, form: str) -> np.ndarray[object]:
        ...

    def _str_strip(self, to_strip: Optional[str] = None) -> np.ndarray[object]:
        ...

    def _str_lstrip(self, to_strip: Optional[str] = None) -> np.ndarray[object]:
        ...

    def _str_rstrip(self, to_strip: Optional[str] = None) -> np.ndarray[object]:
        ...

    def _str_removeprefix(self, prefix: str) -> np.ndarray[object]:
        ...

    def _str_removesuffix(self, suffix: str) -> np.ndarray[object]:
        ...

    def _str_extract(
        self,
        pat: Union[str, re.Pattern],
        flags: int = 0,
        expand: bool = True,
    ) -> List[np.ndarray[object]]:
        ...