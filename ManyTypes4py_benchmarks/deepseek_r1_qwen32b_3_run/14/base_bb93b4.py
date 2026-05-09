from __future__ import annotations
import abc
from typing import TYPE_CHECKING, Literal, Optional, Union, Dict, Any, Tuple, Sequence, Callable
from pandas._libs import lib
import numpy as np
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

    def _str_getitem(self, key: Union[int, slice]) -> Self:
        if isinstance(key, slice):
            return self._str_slice(start=key.start, stop=key.stop, step=key.step)
        else:
            return self._str_get(key)

    @abc.abstractmethod
    def _str_count(self, pat: str, flags: int = 0) -> NpDtype[np.int64]:
        pass

    @abc.abstractmethod
    def _str_pad(self, width: int, side: Literal['left', 'right', 'both'] = 'left', fillchar: str = ' ') -> Self:
        pass

    @abc.abstractmethod
    def _str_contains(self, pat: str, case: bool = True, flags: int = 0, na: Optional[Scalar] = None, regex: bool = True) -> NpDtype[np.bool_]:
        pass

    @abc.abstractmethod
    def _str_startswith(self, pat: str, na: Optional[Scalar] = None) -> NpDtype[np.bool_]:
        pass

    @abc.abstractmethod
    def _str_endswith(self, pat: str, na: Optional[Scalar] = None) -> NpDtype[np.bool_]:
        pass

    @abc.abstractmethod
    def _str_replace(self, pat: str, repl: str, n: int = -1, case: bool = True, flags: int = 0, regex: bool = True) -> Self:
        pass

    @abc.abstractmethod
    def _str_repeat(self, repeats: int) -> Self:
        pass

    @abc.abstractmethod
    def _str_match(self, pat: str, case: bool = True, flags: int = 0, na: Union[Scalar, Literal[lib.no_default]] = lib.no_default) -> NpDtype[np.bool_]:
        pass

    @abc.abstractmethod
    def _str_fullmatch(self, pat: str, case: bool = True, flags: int = 0, na: Union[Scalar, Literal[lib.no_default]] = lib.no_default) -> NpDtype[np.bool_]:
        pass

    @abc.abstractmethod
    def _str_encode(self, encoding: str, errors: str = 'strict') -> Self:
        pass

    @abc.abstractmethod
    def _str_find(self, sub: str, start: int = 0, end: Optional[int] = None) -> NpDtype[np.int64]:
        pass

    @abc.abstractmethod
    def _str_rfind(self, sub: str, start: int = 0, end: Optional[int] = None) -> NpDtype[np.int64]:
        pass

    @abc.abstractmethod
    def _str_findall(self, pat: str, flags: int = 0) -> Self:
        pass

    @abc.abstractmethod
    def _str_get(self, i: int) -> Self:
        pass

    @abc.abstractmethod
    def _str_index(self, sub: str, start: int = 0, end: Optional[int] = None) -> NpDtype[np.int64]:
        pass

    @abc.abstractmethod
    def _str_rindex(self, sub: str, start: int = 0, end: Optional[int] = None) -> NpDtype[np.int64]:
        pass

    @abc.abstractmethod
    def _str_join(self, sep: str) -> Self:
        pass

    @abc.abstractmethod
    def _str_partition(self, sep: str, expand: bool) -> Self:
        pass

    @abc.abstractmethod
    def _str_rpartition(self, sep: str, expand: bool) -> Self:
        pass

    @abc.abstractmethod
    def _str_len(self) -> NpDtype[np.int64]:
        pass

    @abc.abstractmethod
    def _str_slice(self, start: Optional[int] = None, stop: Optional[int] = None, step: Optional[int] = None) -> Self:
        pass

    @abc.abstractmethod
    def _str_slice_replace(self, start: Optional[int] = None, stop: Optional[int] = None, repl: Optional[str] = None) -> Self:
        pass

    @abc.abstractmethod
    def _str_translate(self, table: Dict[int, str]) -> Self:
        pass

    @abc.abstractmethod
    def _str_wrap(self, width: int, **kwargs: Any) -> Self:
        pass

    @abc.abstractmethod
    def _str_get_dummies(self, sep: str = '|', dtype: Optional[NpDtype] = None) -> Self:
        pass

    @abc.abstractmethod
    def _str_isalnum(self) -> NpDtype[np.bool_]:
        pass

    @abc.abstractmethod
    def _str_isalpha(self) -> NpDtype[np.bool_]:
        pass

    @abc.abstractmethod
    def _str_isascii(self) -> NpDtype[np.bool_]:
        pass

    @abc.abstractmethod
    def _str_isdecimal(self) -> NpDtype[np.bool_]:
        pass

    @abc.abstractmethod
    def _str_isdigit(self) -> NpDtype[np.bool_]:
        pass

    @abc.abstractmethod
    def _str_islower(self) -> NpDtype[np.bool_]:
        pass

    @abc.abstractmethod
    def _str_isnumeric(self) -> NpDtype[np.bool_]:
        pass

    @abc.abstractmethod
    def _str_isspace(self) -> NpDtype[np.bool_]:
        pass

    @abc.abstractmethod
    def _str_istitle(self) -> NpDtype[np.bool_]:
        pass

    @abc.abstractmethod
    def _str_isupper(self) -> NpDtype[np.bool_]:
        pass

    @abc.abstractmethod
    def _str_capitalize(self) -> Self:
        pass

    @abc.abstractmethod
    def _str_casefold(self) -> Self:
        pass

    @abc.abstractmethod
    def _str_title(self) -> Self:
        pass

    @abc.abstractmethod
    def _str_swapcase(self) -> Self:
        pass

    @abc.abstractmethod
    def _str_lower(self) -> Self:
        pass

    @abc.abstractmethod
    def _str_upper(self) -> Self:
        pass

    @abc.abstractmethod
    def _str_normalize(self, form: str) -> Self:
        pass

    @abc.abstractmethod
    def _str_strip(self, to_strip: Optional[str] = None) -> Self:
        pass

    @abc.abstractmethod
    def _str_lstrip(self, to_strip: Optional[str] = None) -> Self:
        pass

    @abc.abstractmethod
    def _str_rstrip(self, to_strip: Optional[str] = None) -> Self:
        pass

    @abc.abstractmethod
    def _str_removeprefix(self, prefix: str) -> Self:
        pass

    @abc.abstractmethod
    def _str_removesuffix(self, suffix: str) -> Self:
        pass

    @abc.abstractmethod
    def _str_split(self, pat: Optional[str] = None, n: int = -1, expand: bool = False, regex: Optional[bool] = None) -> Self:
        pass

    @abc.abstractmethod
    def _str_rsplit(self, pat: Optional[str] = None, n: int = -1) -> Self:
        pass

    @abc.abstractmethod
    def _str_extract(self, pat: str, flags: int = 0, expand: bool = True) -> Self:
        pass