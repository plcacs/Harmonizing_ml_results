from __future__ import annotations
import abc
from typing import TYPE_CHECKING, Literal
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

    def _str_getitem(self, key: Union[int, slice]) -> str:
        if isinstance(key, slice):
            return self._str_slice(start=key.start, stop=key.stop, step=key.step)
        else:
            return self._str_get(key)

    @abc.abstractmethod
    def _str_count(self, pat: Union[str, bool, list[str]], flags: int=0) -> None:
        pass

    @abc.abstractmethod
    def _str_pad(self, width: Union[bool, str, None, list[str]], side: typing.Text='left', fillchar: typing.Text=' ') -> None:
        pass

    @abc.abstractmethod
    def _str_contains(self, pat: Union[bool, str, typing.Sequence[int]], case: bool=True, flags: int=0, na: Union[None, bool, str, typing.Sequence[int]]=None, regex: bool=True) -> None:
        pass

    @abc.abstractmethod
    def _str_startswith(self, pat: str, na: Union[None, str]=None) -> None:
        pass

    @abc.abstractmethod
    def _str_endswith(self, pat: Union[str, typing.Iterable[typing.Any]], na: Union[None, str, typing.Iterable[typing.Any]]=None) -> None:
        pass

    @abc.abstractmethod
    def _str_replace(self, pat: Union[bool, str, typing.Sequence[int]], repl: Union[bool, str, typing.Sequence[int]], n: int=-1, case: bool=True, flags: int=0, regex: bool=True) -> None:
        pass

    @abc.abstractmethod
    def _str_repeat(self, repeats: Union[bool, None]) -> None:
        pass

    @abc.abstractmethod
    def _str_match(self, pat: bool, case: bool=True, flags: int=0, na: Any=lib.no_default) -> None:
        pass

    @abc.abstractmethod
    def _str_fullmatch(self, pat: bool, case: bool=True, flags: int=0, na: Any=lib.no_default) -> None:
        pass

    @abc.abstractmethod
    def _str_encode(self, encoding: Union[str, dict, bool], errors: typing.Text='strict') -> None:
        pass

    @abc.abstractmethod
    def _str_find(self, sub: Union[str, int], start: int=0, end: Union[None, str, int]=None) -> None:
        pass

    @abc.abstractmethod
    def _str_rfind(self, sub: Union[str, bool], start: int=0, end: Union[None, str, bool]=None) -> None:
        pass

    @abc.abstractmethod
    def _str_findall(self, pat: Union[str, bool, typing.Sequence], flags: int=0) -> None:
        pass

    @abc.abstractmethod
    def _str_get(self, i: Union[int, slice, str]) -> None:
        pass

    @abc.abstractmethod
    def _str_index(self, sub: Union[str, bool, dict[str, typing.Any]], start: int=0, end: Union[None, str, bool, dict[str, typing.Any]]=None) -> None:
        pass

    @abc.abstractmethod
    def _str_rindex(self, sub: Union[str, bool], start: int=0, end: Union[None, str, bool]=None) -> None:
        pass

    @abc.abstractmethod
    def _str_join(self, sep: Union[str, typing.Mapping]) -> None:
        pass

    @abc.abstractmethod
    def _str_partition(self, sep: Union[bool, str, list[str]], expand: Union[bool, str, list[str]]) -> None:
        pass

    @abc.abstractmethod
    def _str_rpartition(self, sep: Union[bool, str], expand: Union[bool, str]) -> None:
        pass

    @abc.abstractmethod
    def _str_len(self) -> None:
        pass

    @abc.abstractmethod
    def _str_slice(self, start: Union[None, int, float]=None, stop: Union[None, int, float]=None, step: Union[None, int, float]=None) -> None:
        pass

    @abc.abstractmethod
    def _str_slice_replace(self, start: Union[None, int, str]=None, stop: Union[None, int, str]=None, repl: Union[None, int, str]=None) -> None:
        pass

    @abc.abstractmethod
    def _str_translate(self, table: Union[str, dict[str, typing.Sequence[str]], bool]) -> None:
        pass

    @abc.abstractmethod
    def _str_wrap(self, width: Union[int, None], **kwargs) -> None:
        pass

    @abc.abstractmethod
    def _str_get_dummies(self, sep: typing.Text='|', dtype: Union[None, str]=None) -> None:
        pass

    @abc.abstractmethod
    def _str_isalnum(self) -> None:
        pass

    @abc.abstractmethod
    def _str_isalpha(self) -> None:
        pass

    @abc.abstractmethod
    def _str_isascii(self) -> None:
        pass

    @abc.abstractmethod
    def _str_isdecimal(self) -> None:
        pass

    @abc.abstractmethod
    def _str_isdigit(self) -> None:
        pass

    @abc.abstractmethod
    def _str_islower(self) -> None:
        pass

    @abc.abstractmethod
    def _str_isnumeric(self) -> None:
        pass

    @abc.abstractmethod
    def _str_isspace(self) -> None:
        pass

    @abc.abstractmethod
    def _str_istitle(self) -> None:
        pass

    @abc.abstractmethod
    def _str_isupper(self) -> None:
        pass

    @abc.abstractmethod
    def _str_capitalize(self) -> None:
        pass

    @abc.abstractmethod
    def _str_casefold(self) -> None:
        pass

    @abc.abstractmethod
    def _str_title(self) -> None:
        pass

    @abc.abstractmethod
    def _str_swapcase(self) -> None:
        pass

    @abc.abstractmethod
    def _str_lower(self) -> None:
        pass

    @abc.abstractmethod
    def _str_upper(self) -> None:
        pass

    @abc.abstractmethod
    def _str_normalize(self, form: Any) -> None:
        pass

    @abc.abstractmethod
    def _str_strip(self, to_strip: Union[None, str, bool]=None) -> None:
        pass

    @abc.abstractmethod
    def _str_lstrip(self, to_strip: Union[None, str, bool]=None) -> None:
        pass

    @abc.abstractmethod
    def _str_rstrip(self, to_strip: Union[None, str, bool, list[str]]=None) -> None:
        pass

    @abc.abstractmethod
    def _str_removeprefix(self, prefix: Union[str, None]) -> None:
        pass

    @abc.abstractmethod
    def _str_removesuffix(self, suffix: Union[str, list[str]]) -> None:
        pass

    @abc.abstractmethod
    def _str_split(self, pat: Union[None, str, int]=None, n: int=-1, expand: bool=False, regex: Union[None, str, int]=None) -> None:
        pass

    @abc.abstractmethod
    def _str_rsplit(self, pat: Union[None, str, int]=None, n: int=-1) -> None:
        pass

    @abc.abstractmethod
    def _str_extract(self, pat: bool, flags: int=0, expand: bool=True) -> None:
        pass