from __future__ import annotations
import abc
from typing import TYPE_CHECKING, Literal, Optional, Union, Any
from pandas._libs import lib

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    import re
    from pandas._typing import NpDtype, Scalar, Self

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

    def _str_getitem(self, key: Union[slice, int]) -> Any: ...

    @abc.abstractmethod
    def _str_count(self, pat: str, flags: int = 0) -> Any: ...

    @abc.abstractmethod
    def _str_pad(self, width: int, side: Literal['left', 'right', 'both'] = 'left', fillchar: str = ' ') -> Any: ...

    @abc.abstractmethod
    def _str_contains(self, pat: str, case: bool = True, flags: int = 0, na: Optional[bool] = None, regex: bool = True) -> Any: ...

    @abc.abstractmethod
    def _str_startswith(self, pat: str, na: Optional[bool] = None) -> Any: ...

    @abc.abstractmethod
    def _str_endswith(self, pat: str, na: Optional[bool] = None) -> Any: ...

    @abc.abstractmethod
    def _str_replace(self, pat: str, repl: str, n: int = -1, case: bool = True, flags: int = 0, regex: bool = True) -> Any: ...

    @abc.abstractmethod
    def _str_repeat(self, repeats: int) -> Any: ...

    @abc.abstractmethod
    def _str_match(self, pat: str, case: bool = True, flags: int = 0, na: Any = lib.no_default) -> Any: ...

    @abc.abstractmethod
    def _str_fullmatch(self, pat: str, case: bool = True, flags: int = 0, na: Any = lib.no_default) -> Any: ...

    @abc.abstractmethod
    def _str_encode(self, encoding: str, errors: str = 'strict') -> Any: ...

    @abc.abstractmethod
    def _str_find(self, sub: str, start: int = 0, end: Optional[int] = None) -> Any: ...

    @abc.abstractmethod
    def _str_rfind(self, sub: str, start: int = 0, end: Optional[int] = None) -> Any: ...

    @abc.abstractmethod
    def _str_findall(self, pat: str, flags: int = 0) -> Any: ...

    @abc.abstractmethod
    def _str_get(self, i: int) -> Any: ...

    @abc.abstractmethod
    def _str_index(self, sub: str, start: int = 0, end: Optional[int] = None) -> Any: ...

    @abc.abstractmethod
    def _str_rindex(self, sub: str, start: int = 0, end: Optional[int] = None) -> Any: ...

    @abc.abstractmethod
    def _str_join(self, sep: str) -> Any: ...

    @abc.abstractmethod
    def _str_partition(self, sep: str, expand: bool) -> Any: ...

    @abc.abstractmethod
    def _str_rpartition(self, sep: str, expand: bool) -> Any: ...

    @abc.abstractmethod
    def _str_len(self) -> Any: ...

    @abc.abstractmethod
    def _str_slice(self, start: Optional[int] = None, stop: Optional[int] = None, step: Optional[int] = None) -> Any: ...

    @abc.abstractmethod
    def _str_slice_replace(self, start: Optional[int] = None, stop: Optional[int] = None, repl: Optional[str] = None) -> Any: ...

    @abc.abstractmethod
    def _str_translate(self, table: Union[dict[int, Any], Any]) -> Any: ...

    @abc.abstractmethod
    def _str_wrap(self, width: int, **kwargs: Any) -> Any: ...

    @abc.abstractmethod
    def _str_get_dummies(self, sep: str = '|', dtype: Optional[NpDtype] = None) -> Any: ...

    @abc.abstractmethod
    def _str_isalnum(self) -> Any: ...

    @abc.abstractmethod
    def _str_isalpha(self) -> Any: ...

    @abc.abstractmethod
    def _str_isascii(self) -> Any: ...

    @abc.abstractmethod
    def _str_isdecimal(self) -> Any: ...

    @abc.abstractmethod
    def _str_isdigit(self) -> Any: ...

    @abc.abstractmethod
    def _str_islower(self) -> Any: ...

    @abc.abstractmethod
    def _str_isnumeric(self) -> Any: ...

    @abc.abstractmethod
    def _str_isspace(self) -> Any: ...

    @abc.abstractmethod
    def _str_istitle(self) -> Any: ...

    @abc.abstractmethod
    def _str_isupper(self) -> Any: ...

    @abc.abstractmethod
    def _str_capitalize(self) -> Any: ...

    @abc.abstractmethod
    def _str_casefold(self) -> Any: ...

    @abc.abstractmethod
    def _str_title(self) -> Any: ...

    @abc.abstractmethod
    def _str_swapcase(self) -> Any: ...

    @abc.abstractmethod
    def _str_lower(self) -> Any: ...

    @abc.abstractmethod
    def _str_upper(self) -> Any: ...

    @abc.abstractmethod
    def _str_normalize(self, form: str) -> Any: ...

    @abc.abstractmethod
    def _str_strip(self, to_strip: Optional[str] = None) -> Any: ...

    @abc.abstractmethod
    def _str_lstrip(self, to_strip: Optional[str] = None) -> Any: ...

    @abc.abstractmethod
    def _str_rstrip(self, to_strip: Optional[str] = None) -> Any: ...

    @abc.abstractmethod
    def _str_removeprefix(self, prefix: str) -> Any: ...

    @abc.abstractmethod
    def _str_removesuffix(self, suffix: str) -> Any: ...

    @abc.abstractmethod
    def _str_split(self, pat: Optional[str] = None, n: int = -1, expand: bool = False, regex: Optional[bool] = None) -> Any: ...

    @abc.abstractmethod
    def _str_rsplit(self, pat: Optional[str] = None, n: int = -1) -> Any: ...

    @abc.abstractmethod
    def _str_extract(self, pat: str, flags: int = 0, expand: bool = True) -> Any: ...