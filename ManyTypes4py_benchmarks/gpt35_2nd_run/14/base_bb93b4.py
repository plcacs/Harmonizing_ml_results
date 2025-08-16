from __future__ import annotations
import abc
from typing import TYPE_CHECKING, Literal, Union, Optional
from collections.abc import Callable, Sequence
import re
from pandas._typing import NpDtype, Scalar, Self

class BaseStringArrayMethods(abc.ABC):
    def _str_getitem(self, key: Union[int, slice]) -> Union[str, Sequence[str]]:
        pass

    @abc.abstractmethod
    def _str_count(self, pat: str, flags: int = 0) -> int:
        pass

    @abc.abstractmethod
    def _str_pad(self, width: int, side: Literal['left', 'right'], fillchar: str = ' ') -> str:
        pass

    @abc.abstractmethod
    def _str_contains(self, pat: str, case: bool = True, flags: int = 0, na: Optional[bool] = None, regex: bool = True) -> bool:
        pass

    @abc.abstractmethod
    def _str_startswith(self, pat: str, na: Optional[bool] = None) -> bool:
        pass

    @abc.abstractmethod
    def _str_endswith(self, pat: str, na: Optional[bool] = None) -> bool:
        pass

    @abc.abstractmethod
    def _str_replace(self, pat: str, repl: str, n: int = -1, case: bool = True, flags: int = 0, regex: bool = True) -> str:
        pass

    @abc.abstractmethod
    def _str_repeat(self, repeats: int) -> str:
        pass

    @abc.abstractmethod
    def _str_match(self, pat: str, case: bool = True, flags: int = 0, na: Optional[bool] = None) -> bool:
        pass

    @abc.abstractmethod
    def _str_fullmatch(self, pat: str, case: bool = True, flags: int = 0, na: Optional[bool] = None) -> bool:
        pass

    @abc.abstractmethod
    def _str_encode(self, encoding: str, errors: str = 'strict') -> bytes:
        pass

    @abc.abstractmethod
    def _str_find(self, sub: str, start: int = 0, end: Optional[int] = None) -> int:
        pass

    @abc.abstractmethod
    def _str_rfind(self, sub: str, start: int = 0, end: Optional[int] = None) -> int:
        pass

    @abc.abstractmethod
    def _str_findall(self, pat: str, flags: int = 0) -> Sequence[str]:
        pass

    @abc.abstractmethod
    def _str_get(self, i: int) -> str:
        pass

    @abc.abstractmethod
    def _str_index(self, sub: str, start: int = 0, end: Optional[int] = None) -> int:
        pass

    @abc.abstractmethod
    def _str_rindex(self, sub: str, start: int = 0, end: Optional[int] = None) -> int:
        pass

    @abc.abstractmethod
    def _str_join(self, sep: str) -> str:
        pass

    @abc.abstractmethod
    def _str_partition(self, sep: str, expand: bool) -> Sequence[str]:
        pass

    @abc.abstractmethod
    def _str_rpartition(self, sep: str, expand: bool) -> Sequence[str]:
        pass

    @abc.abstractmethod
    def _str_len(self) -> int:
        pass

    @abc.abstractmethod
    def _str_slice(self, start: Optional[int] = None, stop: Optional[int] = None, step: Optional[int] = None) -> str:
        pass

    @abc.abstractmethod
    def _str_slice_replace(self, start: Optional[int] = None, stop: Optional[int] = None, repl: Optional[str] = None) -> str:
        pass

    @abc.abstractmethod
    def _str_translate(self, table: dict) -> str:
        pass

    @abc.abstractmethod
    def _str_wrap(self, width: int, **kwargs) -> str:
        pass

    @abc.abstractmethod
    def _str_get_dummies(self, sep: str = '|', dtype: Optional[NpDtype] = None) -> 'DataFrame':
        pass

    @abc.abstractmethod
    def _str_isalnum(self) -> bool:
        pass

    @abc.abstractmethod
    def _str_isalpha(self) -> bool:
        pass

    @abc.abstractmethod
    def _str_isascii(self) -> bool:
        pass

    @abc.abstractmethod
    def _str_isdecimal(self) -> bool:
        pass

    @abc.abstractmethod
    def _str_isdigit(self) -> bool:
        pass

    @abc.abstractmethod
    def _str_islower(self) -> bool:
        pass

    @abc.abstractmethod
    def _str_isnumeric(self) -> bool:
        pass

    @abc.abstractmethod
    def _str_isspace(self) -> bool:
        pass

    @abc.abstractmethod
    def _str_istitle(self) -> bool:
        pass

    @abc.abstractmethod
    def _str_isupper(self) -> bool:
        pass

    @abc.abstractmethod
    def _str_capitalize(self) -> str:
        pass

    @abc.abstractmethod
    def _str_casefold(self) -> str:
        pass

    @abc.abstractmethod
    def _str_title(self) -> str:
        pass

    @abc.abstractmethod
    def _str_swapcase(self) -> str:
        pass

    @abc.abstractmethod
    def _str_lower(self) -> str:
        pass

    @abc.abstractmethod
    def _str_upper(self) -> str:
        pass

    @abc.abstractmethod
    def _str_normalize(self, form: str) -> str:
        pass

    @abc.abstractmethod
    def _str_strip(self, to_strip: Optional[str] = None) -> str:
        pass

    @abc.abstractmethod
    def _str_lstrip(self, to_strip: Optional[str] = None) -> str:
        pass

    @abc.abstractmethod
    def _str_rstrip(self, to_strip: Optional[str] = None) -> str:
        pass

    @abc.abstractmethod
    def _str_removeprefix(self, prefix: str) -> str:
        pass

    @abc.abstractmethod
    def _str_removesuffix(self, suffix: str) -> str:
        pass

    @abc.abstractmethod
    def _str_split(self, pat: Optional[str] = None, n: int = -1, expand: bool = False, regex: Optional[bool] = None) -> Sequence[str]:
        pass

    @abc.abstractmethod
    def _str_rsplit(self, pat: Optional[str] = None, n: int = -1) -> Sequence[str]:
        pass

    @abc.abstractmethod
    def _str_extract(self, pat: str, flags: int = 0, expand: bool = True) -> Union[str, Sequence[str]]:
        pass
