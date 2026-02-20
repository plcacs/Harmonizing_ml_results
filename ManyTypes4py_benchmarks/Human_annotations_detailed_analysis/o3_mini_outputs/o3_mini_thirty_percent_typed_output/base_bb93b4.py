from __future__ import annotations

import abc
from typing import (
    TYPE_CHECKING,
    Literal,
    Any,
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

    def _str_getitem(self, key: int | slice) -> Any:
        if isinstance(key, slice):
            return self._str_slice(start=key.start, stop=key.stop, step=key.step)
        else:
            return self._str_get(key)

    @abc.abstractmethod
    def _str_count(self, pat: str | re.Pattern, flags: int = 0) -> Any:
        pass

    @abc.abstractmethod
    def _str_pad(
        self,
        width: int,
        side: str = "left",
        fillchar: str = " ",
    ) -> Any:
        pass

    @abc.abstractmethod
    def _str_contains(
        self,
        pat: str | re.Pattern,
        case: bool = True,
        flags: int = 0,
        na: Any = None,
        regex: bool = True,
    ) -> Any:
        pass

    @abc.abstractmethod
    def _str_startswith(self, pat: str, na: Any = None) -> Any:
        pass

    @abc.abstractmethod
    def _str_endswith(self, pat: str, na: Any = None) -> Any:
        pass

    @abc.abstractmethod
    def _str_replace(
        self,
        pat: str,
        repl: str,
        n: int = -1,
        case: bool = True,
        flags: int = 0,
        regex: bool = True,
    ) -> Any:
        pass

    @abc.abstractmethod
    def _str_repeat(self, repeats: int) -> Any:
        pass

    @abc.abstractmethod
    def _str_match(
        self,
        pat: str,
        case: bool = True,
        flags: int = 0,
        na: Any = lib.no_default,
    ) -> Any:
        pass

    @abc.abstractmethod
    def _str_fullmatch(
        self,
        pat: str,
        case: bool = True,
        flags: int = 0,
        na: Scalar | lib.NoDefault = lib.no_default,
    ) -> Any:
        pass

    @abc.abstractmethod
    def _str_encode(self, encoding: str, errors: str = "strict") -> Any:
        pass

    @abc.abstractmethod
    def _str_find(self, sub: str, start: int = 0, end: int | None = None) -> Any:
        pass

    @abc.abstractmethod
    def _str_rfind(self, sub: str, start: int = 0, end: int | None = None) -> Any:
        pass

    @abc.abstractmethod
    def _str_findall(self, pat: str, flags: int = 0) -> Any:
        pass

    @abc.abstractmethod
    def _str_get(self, i: int) -> Any:
        pass

    @abc.abstractmethod
    def _str_index(self, sub: str, start: int = 0, end: int | None = None) -> Any:
        pass

    @abc.abstractmethod
    def _str_rindex(self, sub: str, start: int = 0, end: int | None = None) -> Any:
        pass

    @abc.abstractmethod
    def _str_join(self, sep: str) -> Any:
        pass

    @abc.abstractmethod
    def _str_partition(self, sep: str, expand: bool) -> Any:
        pass

    @abc.abstractmethod
    def _str_rpartition(self, sep: str, expand: bool) -> Any:
        pass

    @abc.abstractmethod
    def _str_len(self) -> Any:
        pass

    @abc.abstractmethod
    def _str_slice(
        self, start: int | None = None, stop: int | None = None, step: int | None = None
    ) -> Any:
        pass

    @abc.abstractmethod
    def _str_slice_replace(
        self, start: int | None = None, stop: int | None = None, repl: str | None = None
    ) -> Any:
        pass

    @abc.abstractmethod
    def _str_translate(self, table: dict[int, Any]) -> Any:
        pass

    @abc.abstractmethod
    def _str_wrap(self, width: int, **kwargs: Any) -> Any:
        pass

    @abc.abstractmethod
    def _str_get_dummies(self, sep: str = "|", dtype: NpDtype | None = None) -> Any:
        pass

    @abc.abstractmethod
    def _str_isalnum(self) -> Any:
        pass

    @abc.abstractmethod
    def _str_isalpha(self) -> Any:
        pass

    @abc.abstractmethod
    def _str_isascii(self) -> Any:
        pass

    @abc.abstractmethod
    def _str_isdecimal(self) -> Any:
        pass

    @abc.abstractmethod
    def _str_isdigit(self) -> Any:
        pass

    @abc.abstractmethod
    def _str_islower(self) -> Any:
        pass

    @abc.abstractmethod
    def _str_isnumeric(self) -> Any:
        pass

    @abc.abstractmethod
    def _str_isspace(self) -> Any:
        pass

    @abc.abstractmethod
    def _str_istitle(self) -> Any:
        pass

    @abc.abstractmethod
    def _str_isupper(self) -> Any:
        pass

    @abc.abstractmethod
    def _str_capitalize(self) -> Any:
        pass

    @abc.abstractmethod
    def _str_casefold(self) -> Any:
        pass

    @abc.abstractmethod
    def _str_title(self) -> Any:
        pass

    @abc.abstractmethod
    def _str_swapcase(self) -> Any:
        pass

    @abc.abstractmethod
    def _str_lower(self) -> Any:
        pass

    @abc.abstractmethod
    def _str_upper(self) -> Any:
        pass

    @abc.abstractmethod
    def _str_normalize(self, form: str) -> Any:
        pass

    @abc.abstractmethod
    def _str_strip(self, to_strip: str | None = None) -> Any:
        pass

    @abc.abstractmethod
    def _str_lstrip(self, to_strip: str | None = None) -> Any:
        pass

    @abc.abstractmethod
    def _str_rstrip(self, to_strip: str | None = None) -> Any:
        pass

    @abc.abstractmethod
    def _str_removeprefix(self, prefix: str) -> Self:
        pass

    @abc.abstractmethod
    def _str_removesuffix(self, suffix: str) -> Self:
        pass

    @abc.abstractmethod
    def _str_split(
        self,
        pat: str | None = None,
        n: int = -1,
        expand: bool = False,
        regex: bool | None = None,
    ) -> Any:
        pass

    @abc.abstractmethod
    def _str_rsplit(self, pat: str | None = None, n: int = -1) -> Any:
        pass

    @abc.abstractmethod
    def _str_extract(self, pat: str, flags: int = 0, expand: bool = True) -> Any:
        pass