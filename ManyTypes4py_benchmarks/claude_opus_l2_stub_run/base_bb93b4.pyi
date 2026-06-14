from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Literal

from pandas._libs import lib

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    import re
    from pandas._typing import NpDtype, Scalar, Self

class BaseStringArrayMethods(abc.ABC):
    def _str_getitem(self, key: slice | int) -> Self: ...

    @abc.abstractmethod
    def _str_count(self, pat: str | re.Pattern, flags: int = 0) -> Self: ...

    @abc.abstractmethod
    def _str_pad(
        self,
        width: int,
        side: Literal["left", "right", "both"] = "left",
        fillchar: str = " ",
    ) -> Self: ...

    @abc.abstractmethod
    def _str_contains(
        self,
        pat: str | re.Pattern,
        case: bool = True,
        flags: int = 0,
        na: Scalar = None,
        regex: bool = True,
    ) -> Self: ...

    @abc.abstractmethod
    def _str_startswith(self, pat: str | tuple[str, ...], na: Scalar = None) -> Self: ...

    @abc.abstractmethod
    def _str_endswith(self, pat: str | tuple[str, ...], na: Scalar = None) -> Self: ...

    @abc.abstractmethod
    def _str_replace(
        self,
        pat: str | re.Pattern,
        repl: str | Callable,
        n: int = -1,
        case: bool = True,
        flags: int = 0,
        regex: bool = True,
    ) -> Self: ...

    @abc.abstractmethod
    def _str_repeat(self, repeats: int | Sequence[int]) -> Self: ...

    @abc.abstractmethod
    def _str_match(
        self,
        pat: str,
        case: bool = True,
        flags: int = 0,
        na: Scalar = ...,
    ) -> Self: ...

    @abc.abstractmethod
    def _str_fullmatch(
        self,
        pat: str,
        case: bool = True,
        flags: int = 0,
        na: Scalar = ...,
    ) -> Self: ...

    @abc.abstractmethod
    def _str_encode(self, encoding: str, errors: str = "strict") -> Self: ...

    @abc.abstractmethod
    def _str_find(self, sub: str, start: int = 0, end: int | None = None) -> Self: ...

    @abc.abstractmethod
    def _str_rfind(self, sub: str, start: int = 0, end: int | None = None) -> Self: ...

    @abc.abstractmethod
    def _str_findall(self, pat: str | re.Pattern, flags: int = 0) -> Self: ...

    @abc.abstractmethod
    def _str_get(self, i: int) -> Self: ...

    @abc.abstractmethod
    def _str_index(self, sub: str, start: int = 0, end: int | None = None) -> Self: ...

    @abc.abstractmethod
    def _str_rindex(self, sub: str, start: int = 0, end: int | None = None) -> Self: ...

    @abc.abstractmethod
    def _str_join(self, sep: str) -> Self: ...

    @abc.abstractmethod
    def _str_partition(self, sep: str, expand: bool) -> Self: ...

    @abc.abstractmethod
    def _str_rpartition(self, sep: str, expand: bool) -> Self: ...

    @abc.abstractmethod
    def _str_len(self) -> Self: ...

    @abc.abstractmethod
    def _str_slice(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> Self: ...

    @abc.abstractmethod
    def _str_slice_replace(
        self,
        start: int | None = None,
        stop: int | None = None,
        repl: str | None = None,
    ) -> Self: ...

    @abc.abstractmethod
    def _str_translate(self, table: dict[int, str | int | None]) -> Self: ...

    @abc.abstractmethod
    def _str_wrap(self, width: int, **kwargs: object) -> Self: ...

    @abc.abstractmethod
    def _str_get_dummies(self, sep: str = "|", dtype: NpDtype | None = None) -> Self: ...

    @abc.abstractmethod
    def _str_isalnum(self) -> Self: ...

    @abc.abstractmethod
    def _str_isalpha(self) -> Self: ...

    @abc.abstractmethod
    def _str_isascii(self) -> Self: ...

    @abc.abstractmethod
    def _str_isdecimal(self) -> Self: ...

    @abc.abstractmethod
    def _str_isdigit(self) -> Self: ...

    @abc.abstractmethod
    def _str_islower(self) -> Self: ...

    @abc.abstractmethod
    def _str_isnumeric(self) -> Self: ...

    @abc.abstractmethod
    def _str_isspace(self) -> Self: ...

    @abc.abstractmethod
    def _str_istitle(self) -> Self: ...

    @abc.abstractmethod
    def _str_isupper(self) -> Self: ...

    @abc.abstractmethod
    def _str_capitalize(self) -> Self: ...

    @abc.abstractmethod
    def _str_casefold(self) -> Self: ...

    @abc.abstractmethod
    def _str_title(self) -> Self: ...

    @abc.abstractmethod
    def _str_swapcase(self) -> Self: ...

    @abc.abstractmethod
    def _str_lower(self) -> Self: ...

    @abc.abstractmethod
    def _str_upper(self) -> Self: ...

    @abc.abstractmethod
    def _str_normalize(self, form: str) -> Self: ...

    @abc.abstractmethod
    def _str_strip(self, to_strip: str | None = None) -> Self: ...

    @abc.abstractmethod
    def _str_lstrip(self, to_strip: str | None = None) -> Self: ...

    @abc.abstractmethod
    def _str_rstrip(self, to_strip: str | None = None) -> Self: ...

    @abc.abstractmethod
    def _str_removeprefix(self, prefix: str) -> Self: ...

    @abc.abstractmethod
    def _str_removesuffix(self, suffix: str) -> Self: ...

    @abc.abstractmethod
    def _str_split(
        self,
        pat: str | re.Pattern | None = None,
        n: int = -1,
        expand: bool = False,
        regex: bool | None = None,
    ) -> Self: ...

    @abc.abstractmethod
    def _str_rsplit(self, pat: str | None = None, n: int = -1) -> Self: ...

    @abc.abstractmethod
    def _str_extract(self, pat: str, flags: int = 0, expand: bool = True) -> Self: ...