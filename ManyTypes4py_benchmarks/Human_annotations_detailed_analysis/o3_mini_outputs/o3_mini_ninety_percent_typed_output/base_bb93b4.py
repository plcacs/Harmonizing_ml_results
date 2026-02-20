from __future__ import annotations

import abc
from typing import (
    TYPE_CHECKING,
    Literal,
)

from pandas._libs import lib

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Sequence,
    )
    import re

    from pandas._typing import (
        NpDtype,
        Scalar,
        Self,
    )


class BaseStringArrayMethods(abc.ABC):
    """
    Base class for extension arrays implementing string methods.

    This is where our ExtensionArrays can override the implementation of
    Series.str.<method>. We don't expect this to work with
    3rd-party extension arrays.

    * User calls Series.str.<method>
    * pandas extracts the extension array from the Series
    * pandas calls ``extension_array._str_<method>(*args, **kwargs)``
    * pandas wraps the result, to return to the user.

    See :ref:`Series.str` for the docstring of each method.
    """

    def _str_getitem(self, key: int | slice) -> Self | Scalar:
        if isinstance(key, slice):
            return self._str_slice(start=key.start, stop=key.stop, step=key.step)
        else:
            return self._str_get(key)

    @abc.abstractmethod
    def _str_count(self, pat: str | re.Pattern, flags: int = 0) -> object:
        pass

    @abc.abstractmethod
    def _str_pad(
        self,
        width: int,
        side: Literal["left", "right", "both"] = "left",
        fillchar: str = " ",
    ) -> object:
        pass

    @abc.abstractmethod
    def _str_contains(
        self,
        pat: str | re.Pattern,
        case: bool = True,
        flags: int = 0,
        na: object = None,
        regex: bool = True,
    ) -> object:
        pass

    @abc.abstractmethod
    def _str_startswith(self, pat: str, na: object = None) -> object:
        pass

    @abc.abstractmethod
    def _str_endswith(self, pat: str, na: object = None) -> object:
        pass

    @abc.abstractmethod
    def _str_replace(
        self,
        pat: str | re.Pattern,
        repl: str | Callable,
        n: int = -1,
        case: bool = True,
        flags: int = 0,
        regex: bool = True,
    ) -> object:
        pass

    @abc.abstractmethod
    def _str_repeat(self, repeats: int | Sequence[int]) -> object:
        pass

    @abc.abstractmethod
    def _str_match(
        self,
        pat: str,
        case: bool = True,
        flags: int = 0,
        na: Scalar | lib.NoDefault = lib.no_default,
    ) -> object:
        pass

    @abc.abstractmethod
    def _str_fullmatch(
        self,
        pat: str | re.Pattern,
        case: bool = True,
        flags: int = 0,
        na: Scalar | lib.NoDefault = lib.no_default,
    ) -> object:
        pass

    @abc.abstractmethod
    def _str_encode(self, encoding: str, errors: str = "strict") -> object:
        pass

    @abc.abstractmethod
    def _str_find(self, sub: str, start: int = 0, end: int | None = None) -> object:
        pass

    @abc.abstractmethod
    def _str_rfind(self, sub: str, start: int = 0, end: int | None = None) -> object:
        pass

    @abc.abstractmethod
    def _str_findall(self, pat: str | re.Pattern, flags: int = 0) -> object:
        pass

    @abc.abstractmethod
    def _str_get(self, i: int) -> object:
        pass

    @abc.abstractmethod
    def _str_index(self, sub: str, start: int = 0, end: int | None = None) -> object:
        pass

    @abc.abstractmethod
    def _str_rindex(self, sub: str, start: int = 0, end: int | None = None) -> object:
        pass

    @abc.abstractmethod
    def _str_join(self, sep: str) -> object:
        pass

    @abc.abstractmethod
    def _str_partition(self, sep: str, expand: bool) -> object:
        pass

    @abc.abstractmethod
    def _str_rpartition(self, sep: str, expand: bool) -> object:
        pass

    @abc.abstractmethod
    def _str_len(self) -> object:
        pass

    @abc.abstractmethod
    def _str_slice(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> object:
        pass

    @abc.abstractmethod
    def _str_slice_replace(
        self,
        start: int | None = None,
        stop: int | None = None,
        repl: str | None = None,
    ) -> object:
        pass

    @abc.abstractmethod
    def _str_translate(self, table: dict[int, int]) -> object:
        pass

    @abc.abstractmethod
    def _str_wrap(self, width: int, **kwargs: object) -> object:
        pass

    @abc.abstractmethod
    def _str_get_dummies(
        self, sep: str = "|", dtype: NpDtype | None = None
    ) -> object:
        pass

    @abc.abstractmethod
    def _str_isalnum(self) -> object:
        pass

    @abc.abstractmethod
    def _str_isalpha(self) -> object:
        pass

    @abc.abstractmethod
    def _str_isascii(self) -> object:
        pass

    @abc.abstractmethod
    def _str_isdecimal(self) -> object:
        pass

    @abc.abstractmethod
    def _str_isdigit(self) -> object:
        pass

    @abc.abstractmethod
    def _str_islower(self) -> object:
        pass

    @abc.abstractmethod
    def _str_isnumeric(self) -> object:
        pass

    @abc.abstractmethod
    def _str_isspace(self) -> object:
        pass

    @abc.abstractmethod
    def _str_istitle(self) -> object:
        pass

    @abc.abstractmethod
    def _str_isupper(self) -> object:
        pass

    @abc.abstractmethod
    def _str_capitalize(self) -> object:
        pass

    @abc.abstractmethod
    def _str_casefold(self) -> object:
        pass

    @abc.abstractmethod
    def _str_title(self) -> object:
        pass

    @abc.abstractmethod
    def _str_swapcase(self) -> object:
        pass

    @abc.abstractmethod
    def _str_lower(self) -> object:
        pass

    @abc.abstractmethod
    def _str_upper(self) -> object:
        pass

    @abc.abstractmethod
    def _str_normalize(self, form: str) -> object:
        pass

    @abc.abstractmethod
    def _str_strip(self, to_strip: str | None = None) -> object:
        pass

    @abc.abstractmethod
    def _str_lstrip(self, to_strip: str | None = None) -> object:
        pass

    @abc.abstractmethod
    def _str_rstrip(self, to_strip: str | None = None) -> object:
        pass

    @abc.abstractmethod
    def _str_removeprefix(self, prefix: str) -> Self:
        pass

    @abc.abstractmethod
    def _str_removesuffix(self, suffix: str) -> Self:
        pass

    @abc.abstractmethod
    def _str_split(
        self, pat: str | None = None, n: int = -1, expand: bool = False, regex: bool | None = None
    ) -> object:
        pass

    @abc.abstractmethod
    def _str_rsplit(self, pat: str | None = None, n: int = -1) -> object:
        pass

    @abc.abstractmethod
    def _str_extract(self, pat: str, flags: int = 0, expand: bool = True) -> object:
        pass