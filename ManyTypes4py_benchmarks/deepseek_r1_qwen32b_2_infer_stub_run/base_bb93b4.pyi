from __future__ import annotations
from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from pandas._typing import NpDtype, Scalar, Self

class BaseStringArrayMethods(ABC):
    @abstractmethod
    def _str_count(self, pat: str, flags: int = 0) -> NpDtype:
        ...

    @abstractmethod
    def _str_pad(self, width: int, side: Literal['left', 'right', 'both'] = 'left', fillchar: str = ' ') -> NpDtype:
        ...

    @abstractmethod
    def _str_contains(self, pat: str, case: bool = True, flags: int = 0, na: Scalar = None, regex: bool = True) -> NpDtype:
        ...

    @abstractmethod
    def _str_startswith(self, pat: str, na: Scalar = None) -> NpDtype:
        ...

    @abstractmethod
    def _str_endswith(self, pat: str, na: Scalar = None) -> NpDtype:
        ...

    @abstractmethod
    def _str_replace(self, pat: str, repl: str, n: int = -1, case: bool = True, flags: int = 0, regex: bool = True) -> NpDtype:
        ...

    @abstractmethod
    def _str_repeat(self, repeats: int) -> NpDtype:
        ...

    @abstractmethod
    def _str_match(self, pat: str, case: bool = True, flags: int = 0, na: Scalar = lib.no_default) -> NpDtype:
        ...

    @abstractmethod
    def _str_fullmatch(self, pat: str, case: bool = True, flags: int = 0, na: Scalar = lib.no_default) -> NpDtype:
        ...

    @abstractmethod
    def _str_encode(self, encoding: str, errors: str = 'strict') -> NpDtype:
        ...

    @abstractmethod
    def _str_find(self, sub: str, start: int = 0, end: Optional[int] = None) -> NpDtype:
        ...

    @abstractmethod
    def _str_rfind(self, sub: str, start: int = 0, end: Optional[int] = None) -> NpDtype:
        ...

    @abstractmethod
    def _str_findall(self, pat: str, flags: int = 0) -> NpDtype:
        ...

    @abstractmethod
    def _str_get(self, i: int) -> NpDtype:
        ...

    @abstractmethod
    def _str_index(self, sub: str, start: int = 0, end: Optional[int] = None) -> NpDtype:
        ...

    @abstractmethod
    def _str_rindex(self, sub: str, start: int = 0, end: Optional[int] = None) -> NpDtype:
        ...

    @abstractmethod
    def _str_join(self, sep: str) -> NpDtype:
        ...

    @abstractmethod
    def _str_partition(self, sep: str, expand: bool) -> NpDtype:
        ...

    @abstractmethod
    def _str_rpartition(self, sep: str, expand: bool) -> NpDtype:
        ...

    @abstractmethod
    def _str_len(self) -> NpDtype:
        ...

    @abstractmethod
    def _str_slice(self, start: Optional[int] = None, stop: Optional[int] = None, step: Optional[int] = None) -> NpDtype:
        ...

    @abstractmethod
    def _str_slice_replace(self, start: Optional[int] = None, stop: Optional[int] = None, repl: Optional[str] = None) -> NpDtype:
        ...

    @abstractmethod
    def _str_translate(self, table: Dict[str, str]) -> NpDtype:
        ...

    @abstractmethod
    def _str_wrap(self, width: int, **kwargs: Any) -> NpDtype:
        ...

    @abstractmethod
    def _str_get_dummies(self, sep: str = '|', dtype: Optional[type] = None) -> NpDtype:
        ...

    @abstractmethod
    def _str_isalnum(self) -> NpDtype:
        ...

    @abstractmethod
    def _str_isalpha(self) -> NpDtype:
        ...

    @abstractmethod
    def _str_isascii(self) -> NpDtype:
        ...

    @abstractmethod
    def _str_isdecimal(self) -> NpDtype:
        ...

    @abstractmethod
    def _str_isdigit(self) -> NpDtype:
        ...

    @abstractmethod
    def _str_islower(self) -> NpDtype:
        ...

    @abstractmethod
    def _str_isnumeric(self) -> NpDtype:
        ...

    @abstractmethod
    def _str_isspace(self) -> NpDtype:
        ...

    @abstractmethod
    def _str_istitle(self) -> NpDtype:
        ...

    @abstractmethod
    def _str_isupper(self) -> NpDtype:
        ...

    @abstractmethod
    def _str_capitalize(self) -> NpDtype:
        ...

    @abstractmethod
    def _str_casefold(self) -> NpDtype:
        ...

    @abstractmethod
    def _str_title(self) -> NpDtype:
        ...

    @abstractmethod
    def _str_swapcase(self) -> NpDtype:
        ...

    @abstractmethod
    def _str_lower(self) -> NpDtype:
        ...

    @abstractmethod
    def _str_upper(self) -> NpDtype:
        ...

    @abstractmethod
    def _str_normalize(self, form: str) -> NpDtype:
        ...

    @abstractmethod
    def _str_strip(self, to_strip: Optional[str] = None) -> NpDtype:
        ...

    @abstractmethod
    def _str_lstrip(self, to_strip: Optional[str] = None) -> NpDtype:
        ...

    @abstractmethod
    def _str_rstrip(self, to_strip: Optional[str] = None) -> NpDtype:
        ...

    @abstractmethod
    def _str_removeprefix(self, prefix: str) -> NpDtype:
        ...

    @abstractmethod
    def _str_removesuffix(self, suffix: str) -> NpDtype:
        ...

    @abstractmethod
    def _str_split(self, pat: Optional[str] = None, n: int = -1, expand: bool = False, regex: Optional[bool] = None) -> NpDtype:
        ...

    @abstractmethod
    def _str_rsplit(self, pat: Optional[str] = None, n: int = -1) -> NpDtype:
        ...

    @abstractmethod
    def _str_extract(self, pat: str, flags: int = 0, expand: bool = True) -> NpDtype:
        ...

    def _str_getitem(self, key: Union[slice, int]) -> Any:
        ...